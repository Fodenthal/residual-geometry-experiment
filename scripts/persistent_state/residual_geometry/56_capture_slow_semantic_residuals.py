#!/usr/bin/env python
"""Capture only the frozen layer/positions and local embedding summaries."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from residual_geometry.slow_semantic.protocol import PROTOCOL, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = Path(args.run_dir)
    labels = run / "labels" / "document_labels.parquet"
    tokens_path = run / "labels" / "token_ids.npy"
    output = run / "capture" / "residuals.npy"
    local_output = run / "capture" / "local_embedding_means.npy"
    if output.exists() and local_output.exists() and not args.overwrite:
        print("capture already complete")
        return 0
    if not labels.exists() or not tokens_path.exists():
        raise FileNotFoundError("labels must be frozen before residual access")
    if not (run / "decisions" / "label_health.json").exists():
        raise FileNotFoundError("label-health gate missing")

    tokens = np.load(tokens_path, mmap_mode="r")
    n_docs, n_tokens = tokens.shape
    positions = np.asarray(PROTOCOL["positions"], dtype=np.int64)
    if n_docs != PROTOCOL["documents"] or n_tokens != PROTOCOL["context_length"]:
        raise ValueError(f"unexpected token matrix {tokens.shape}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_source = os.environ.get("SLOW_SEMANTIC_MODEL_PATH", PROTOCOL["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, revision=None if model_source != PROTOCOL["model"] else PROTOCOL["model_revision"],
        local_files_only=model_source != PROTOCOL["model"],
    )
    # TransformerLens requires an official model identifier as its first argument,
    # even when the weights are supplied from a pinned local Hugging Face snapshot.
    # Load the authoritative snapshot explicitly, then hand that exact model to TL.
    hf_model = None
    if model_source != PROTOCOL["model"]:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=True,
            torch_dtype=dtype,
        )
    model = HookedTransformer.from_pretrained(
        PROTOCOL["model"],
        revision=PROTOCOL["model_revision"] if hf_model is None else None,
        tokenizer=tokenizer,
        hf_model=hf_model,
        local_files_only=hf_model is not None,
        device=device,
        dtype=dtype,
    ).eval()
    del hf_model
    if int(model.cfg.d_model) != PROTOCOL["d_model"]:
        raise ValueError("model width mismatch")
    residuals = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float16,
        shape=(n_docs, len(positions), PROTOCOL["d_model"]),
    )
    local = np.lib.format.open_memmap(
        local_output, mode="w+", dtype=np.float16,
        shape=(n_docs, len(positions), PROTOCOL["d_model"]),
    )
    gpu_positions = torch.as_tensor(positions, device=device)
    times: list[dict[str, float]] = []
    for batch_i, start in enumerate(range(0, n_docs, args.batch_size)):
        stop = min(n_docs, start + args.batch_size)
        batch_tokens = torch.as_tensor(np.asarray(tokens[start:stop], dtype=np.int64), device=device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                batch_tokens, names_filter=lambda name: name == PROTOCOL["hook"], return_type=None
            )
            selected = cache[PROTOCOL["hook"]].index_select(1, gpu_positions)
        forward_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.inference_mode():
            embedded = model.W_E[batch_tokens]
            summaries = torch.stack([
                embedded[:, max(0, int(p) - 16) : min(n_tokens, int(p) + 17)].mean(dim=1)
                for p in positions
            ], dim=1)
            residuals[start:stop] = selected.float().cpu().numpy().astype(np.float16)
            local[start:stop] = summaries.float().cpu().numpy().astype(np.float16)
        transfer_seconds = time.perf_counter() - t1
        row = {"batch": batch_i, "start": start, "stop": stop,
               "forward_seconds": forward_seconds, "transfer_and_local_seconds": transfer_seconds}
        times.append(row)
        print(json.dumps(row), flush=True)
        if args.profile_batches and batch_i + 1 >= args.profile_batches:
            residuals.flush(); local.flush()
            mean_seconds = float(np.mean([x["forward_seconds"] + x["transfer_and_local_seconds"] for x in times]))
            selected_bytes = (stop - start) * len(positions) * PROTOCOL["d_model"] * 2 * 2
            save_json(run / "capture" / "profile.json", {
                "status": "PROFILE_ONLY", "batches": len(times),
                "mean_seconds_per_batch": mean_seconds,
                "projected_seconds": mean_seconds * math.ceil(n_docs / args.batch_size),
                "selected_transfer_bytes_per_batch_two_arrays": selected_bytes,
                "selected_fraction": len(positions) / n_tokens,
            })
            output.unlink(missing_ok=True); local_output.unlink(missing_ok=True)
            return 4
    residuals.flush(); local.flush()
    save_json(run / "capture" / "manifest.json", {
        "status": "PASS", "documents": n_docs, "positions": positions.tolist(),
        "residual_shape": list(residuals.shape), "dtype": "float16",
        "hook": PROTOCOL["hook"], "model": PROTOCOL["model"],
        "selected_fraction": len(positions) / n_tokens,
        "total_forward_passes": math.ceil(n_docs / args.batch_size),
        "full_forward_passes_per_batch": 1,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
