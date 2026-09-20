#!/usr/bin/env python
"""Evaluate the exact canonical signed Q31 object with the current runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.persistent_state.autocorr.estimators import AutocorrAccumulator, build_timescale_table, compute_document_autocorr
from src.persistent_state.subspace.fat_subspace import load_residual_basis_artifact, sample_unit_directions_in_span


EXPECTED_BASIS_SHA256 = "966736e0346dd354559f1354efa86f71c4e6c01cbb929ff70b18f7a9420cce6c"
MODEL = "google/gemma-2-2b"
REVISION = "main"
HOOK = "blocks.12.hook_resid_post"
COUNT = 512
SEED = 31
MAX_LAG = 512


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--context-pool", type=Path, required=True)
    parser.add_argument("--context-split", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile-batches", type=int)
    args = parser.parse_args()

    basis_hash = hashlib.sha256(args.basis.read_bytes()).hexdigest()
    if basis_hash != EXPECTED_BASIS_SHA256:
        raise ValueError(f"canonical basis hash mismatch: {basis_hash}")
    artifact = load_residual_basis_artifact(str(args.basis))
    if artifact.k != 31 or artifact.basis.shape != (2304, 31):
        raise ValueError("canonical basis rank or shape changed")
    families, family_counts = np.unique(artifact.source_probe_family.astype(str), return_counts=True)
    composition = dict(zip(families.tolist(), family_counts.astype(int).tolist()))
    if composition != {"pca": 1, "time_lagged": 30}:
        raise ValueError(f"canonical source composition changed: {composition}")
    orthonormality_error = float(np.linalg.norm(artifact.basis.T @ artifact.basis - np.eye(31)))
    if orthonormality_error > 1e-5:
        raise ValueError(f"canonical basis is not orthonormal: {orthonormality_error}")
    directions = sample_unit_directions_in_span(artifact.basis, count=COUNT, seed=SEED)

    pool = pd.read_parquet(args.context_pool)
    split = pd.read_parquet(args.context_split)
    documents = pool.merge(split, on="context_id", how="inner")
    frames = {name: documents.loc[documents.split == name].reset_index(drop=True) for name in ("val", "test")}
    if any(len(frame) != 500 for frame in frames.values()):
        raise ValueError("canonical validation/test splits must each contain 500 documents")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    source = os.environ.get("SLOW_SEMANTIC_MODEL_PATH", MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=None if source != MODEL else REVISION, local_files_only=source != MODEL
    )
    hf_model = None
    if source != MODEL:
        hf_model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, torch_dtype=dtype)
    model = HookedTransformer.from_pretrained(
        MODEL,
        revision=REVISION if hf_model is None else None,
        tokenizer=tokenizer,
        hf_model=hf_model,
        local_files_only=hf_model is not None,
        device=device,
        dtype=dtype,
    ).eval()
    del hf_model
    gpu_directions = torch.as_tensor(directions.T, dtype=torch.float32, device=device)

    output = args.run_dir / "canonical_q31_audit"
    timings = []
    split_tables = []
    split_profiles = {}
    trace_payload = {}
    for split_name, frame in frames.items():
        limit = len(frame)
        if args.profile_batches is not None:
            limit = min(limit, args.profile_batches * args.batch_size)
        tokens = np.asarray([list(value) for value in frame.token_ids.iloc[:limit]], dtype=np.int64)
        accumulator = AutocorrAccumulator(n_features=COUNT, max_lag=MAX_LAG)
        for batch_index, start in enumerate(range(0, limit, args.batch_size)):
            stop = min(limit, start + args.batch_size)
            t0 = time.perf_counter()
            batch = torch.as_tensor(tokens[start:stop], device=device)
            with torch.inference_mode():
                _, cache = model.run_with_cache(batch, names_filter=lambda name: name == HOOK, return_type=None)
                projections = torch.einsum("btd,dp->btp", cache[HOOK].float(), gpu_directions)
                cpu_projections = projections.cpu().numpy().astype(np.float32)
            if not np.isfinite(cpu_projections).all():
                raise FloatingPointError(f"non-finite projections in {split_name} batch {batch_index}")
            accumulator.update(compute_document_autocorr(cpu_projections, max_lag=MAX_LAG, estimator="within"))
            if batch_index == 0:
                trace_payload[f"{split_name}_context_ids"] = frame.context_id.iloc[: min(5, stop)].astype(str).to_numpy()
                trace_payload[f"{split_name}_projections"] = cpu_projections[:5, :, :5]
            timings.append({
                "split": split_name,
                "batch": batch_index,
                "documents": stop - start,
                "seconds": time.perf_counter() - t0,
                "projection_bytes_transferred": int(projections.numel() * 4),
                "raw_residual_bytes_transferred": 0,
            })
        if args.profile_batches is not None:
            continue
        result = accumulator.finalize()
        table = build_timescale_table(
            feature_indices=np.arange(COUNT),
            estimator_results={"within": result},
            max_lag=MAX_LAG,
            min_valid_docs=50,
            min_valid_lag_fraction=0.8,
            smoothing_width=5,
        )
        table["split"] = split_name
        table["observable"] = "signed"
        table["probe_family"] = "canonical_q31_random_in_span"
        split_tables.append(table)
        split_profiles[split_name] = result.profiles.astype(np.float32)

    if args.profile_batches is not None:
        mean_seconds = float(np.mean([row["seconds"] for row in timings]))
        projected_batches = 2 * math.ceil(500 / args.batch_size)
        save_json(output / "profile.json", {
            "status": "PROFILE_ONLY",
            "profile_batches_per_split": args.profile_batches,
            "mean_seconds_per_batch": mean_seconds,
            "projected_seconds_excluding_model_load": mean_seconds * projected_batches,
            "timings": timings,
        })
        return 4

    output.mkdir(parents=True, exist_ok=True)
    timescales = pd.concat(split_tables, ignore_index=True)
    timescales.to_parquet(output / "canonical_q31_signed_timescales.parquet", index=False)
    np.savez_compressed(
        output / "canonical_q31_signed_profiles.npz",
        validation=split_profiles["val"],
        test=split_profiles["test"],
    )
    np.savez_compressed(output / "fixed_trace_sample.npz", **trace_payload)
    medians = {
        name: float(timescales.loc[timescales.split == name, "tau_within"].median())
        for name in ("val", "test")
    }
    reproduced = abs(medians["val"] - 24.0) <= 3.0 and abs(medians["test"] - 25.0) <= 3.0
    decision = "CANONICAL_Q31_REPRODUCED" if reproduced else "CANONICAL_Q31_EVALUATOR_MISMATCH"
    summary = {
        "decision": decision,
        "basis_sha256": basis_hash,
        "basis_rank": artifact.k,
        "basis_orthonormality_error": orthonormality_error,
        "source_family_counts": composition,
        "observable": "signed",
        "random_in_span_count": COUNT,
        "random_in_span_seed": SEED,
        "canonical_reference_medians": {"val": 24.0, "test": 25.0},
        "current_medians": medians,
        "raw_residual_bytes_transferred": 0,
        "full_forward_passes": 2 * math.ceil(500 / args.batch_size),
        "full_forward_passes_per_batch": 1,
        "timings": timings,
    }
    save_json(output / "decision.json", summary)
    (output / "report.md").write_text(
        "# Canonical Q31 Signed Provenance Audit\n\n"
        f"Decision: `{decision}`\n\n"
        f"Canonical reference val/test medians: 24/25. Current medians: {medians['val']:.1f}/{medians['test']:.1f}.\n\n"
        f"Basis SHA-256: `{basis_hash}`. Observable: signed. Directions: 512, seed 31.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
