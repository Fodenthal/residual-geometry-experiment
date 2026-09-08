#!/usr/bin/env python
"""Build the fresh C4 pool and freeze activation-blind document labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from residual_geometry.slow_semantic.protocol import PROTOCOL, deterministic_document_split, save_json


PROMPT = """Classify this web document using exactly one topic and one register from the frozen lists.
Topic: {topics}
Register: {registers}
Return exactly: topic=<label>;register=<label>
If genuinely ambiguous, use other_or_unclear. Do not explain.

DOCUMENT:
{text}"""


def stable_id(text: str, url: str) -> str:
    return hashlib.sha256((url + "\n" + text).encode("utf-8", errors="replace")).hexdigest()


def formatting_label(text: str) -> str:
    lines = text.splitlines() or [text]
    chars = max(len(text), 1)
    code = sum(text.count(x) for x in ("{", "}", "</", ";", "```")) / chars
    list_lines = sum(bool(re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line)) for line in lines) / len(lines)
    table = sum(line.count("|") >= 2 or "\t" in line for line in lines) / len(lines)
    quote = sum(line.lstrip().startswith((">", '"', "“")) for line in lines) / len(lines)
    newline = len(lines) / chars
    repeated = 1.0 - len(set(line.strip() for line in lines if line.strip())) / max(sum(bool(line.strip()) for line in lines), 1)
    scores = {
        "code_or_markup": code,
        "list_or_table": max(list_lines, table),
        "quotation_or_dialogue": quote,
        "newline_heavy": newline * 80,
        "boilerplate_repetition": repeated,
        "plain_prose": 0.035,
    }
    return max(scores, key=scores.get)


def parse_label(output: str, choices: list[str], key: str) -> str:
    match = re.search(rf"{key}\s*=\s*([a-z0-9_]+)", output.lower())
    if match and match.group(1) in choices:
        return match.group(1)
    found = [choice for choice in choices if choice in output.lower()]
    return found[0] if len(found) == 1 else "other_or_unclear"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = Path(args.run_dir)
    if not (run / "decisions" / "s0.json").exists():
        raise FileNotFoundError("S0 decision missing")
    output = run / "labels" / "document_labels.parquet"
    tokens_output = run / "labels" / "token_ids.npy"
    if output.exists() and tokens_output.exists() and not args.overwrite:
        print("labels already frozen")
        return 0

    label_manifest = {
        "status": "FROZEN_BEFORE_RESIDUAL_ACCESS",
        "model": PROTOCOL["labeler_model"],
        "revision": PROTOCOL["labeler_revision"],
        "prompt": PROMPT,
        "topic_classes": PROTOCOL["topic_classes"],
        "register_classes": PROTOCOL["register_classes"],
        "abstention_rule": PROTOCOL["labeler_abstention_rule"],
        "labeler_input": "raw document text only; no activations or slow-space result",
    }
    save_json(run / "labels" / "labeler_manifest.json", label_manifest)

    model_source = os.environ.get("SLOW_SEMANTIC_MODEL_PATH", PROTOCOL["model"])
    labeler_source = os.environ.get("SLOW_SEMANTIC_LABELER_PATH", PROTOCOL["labeler_model"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, revision=None if model_source != PROTOCOL["model"] else PROTOCOL["tokenizer_revision"],
        local_files_only=model_source != PROTOCOL["model"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset(
        PROTOCOL["corpus"], PROTOCOL["corpus_config"], split=PROTOCOL["corpus_split"], streaming=True
    )
    needed = PROTOCOL["documents"]
    records: list[dict[str, object]] = []
    token_rows: list[np.ndarray] = []
    seen: set[str] = set()
    for row in dataset:
        text = str(row.get("text", ""))
        url = str(row.get("url", ""))
        doc_id = stable_id(text, url)
        if doc_id in seen:
            continue
        # Frozen hash thinning makes selection independent of stream batching.
        if int(doc_id[:8], 16) % 7 != PROTOCOL["sample_seed"] % 7:
            continue
        encoded = tokenizer(
            text, add_special_tokens=True, truncation=True,
            max_length=PROTOCOL["context_length"],
        )["input_ids"]
        if len(encoded) < PROTOCOL["context_length"]:
            continue
        encoded = np.asarray(encoded[: PROTOCOL["context_length"]], dtype=np.int32)
        seen.add(doc_id)
        records.append({
            "document_id": doc_id,
            "url": url,
            "host": (urlparse(url).hostname or "").lower(),
            "text": text,
            "formatting": formatting_label(text),
        })
        token_rows.append(encoded)
        if len(records) == needed:
            break
    if len(records) != needed:
        raise RuntimeError(f"only found {len(records)} eligible documents, required {needed}")
    splits = deterministic_document_split([str(r["document_id"]) for r in records])
    for record, split in zip(records, splits):
        record["split"] = split

    label_tokenizer = AutoTokenizer.from_pretrained(
        labeler_source,
        revision=None if labeler_source != PROTOCOL["labeler_model"] else PROTOCOL["labeler_revision"],
        local_files_only=labeler_source != PROTOCOL["labeler_model"],
    )
    if label_tokenizer.pad_token_id is None:
        label_tokenizer.pad_token_id = label_tokenizer.eos_token_id
    label_model = AutoModelForCausalLM.from_pretrained(
        labeler_source,
        revision=None if labeler_source != PROTOCOL["labeler_model"] else PROTOCOL["labeler_revision"],
        local_files_only=labeler_source != PROTOCOL["labeler_model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    topics = PROTOCOL["topic_classes"]
    registers = PROTOCOL["register_classes"]
    completed = 0
    batch_times: list[float] = []
    for start in range(0, needed, args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts = [PROMPT.format(
            topics=", ".join(topics), registers=", ".join(registers),
            text=str(row["text"])[:12000],
        ) for row in batch]
        chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
        rendered = [label_tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        encoded = label_tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=True,
            max_length=PROTOCOL["labeler_max_input_tokens"],
        ).to(label_model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = label_model.generate(
                **encoded, max_new_tokens=32, do_sample=False,
                pad_token_id=label_tokenizer.pad_token_id,
            )
        batch_times.append(time.perf_counter() - t0)
        for i, row in enumerate(batch):
            new_tokens = generated[i, encoded["input_ids"].shape[1] :]
            raw = label_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            row["topic"] = parse_label(raw, topics, "topic")
            row["register"] = parse_label(raw, registers, "register")
            row["labeler_raw"] = raw
            completed += 1
        print(json.dumps({"labeled": completed, "batch_seconds": batch_times[-1]}), flush=True)
        if args.profile_batches and len(batch_times) >= args.profile_batches:
            save_json(run / "labels" / "profile.json", {
                "batches": len(batch_times), "documents": completed,
                "mean_seconds_per_batch": float(np.mean(batch_times)),
                "projected_label_seconds": float(np.mean(batch_times) * np.ceil(needed / args.batch_size)),
            })
            return 4

    del label_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    hosts = Counter(str(r["host"]) for r in records if r["host"])
    for row in records:
        host = str(row["host"])
        row["source_template"] = host if hosts[host] >= 20 else "OTHER_LOW_SUPPORT"
    frame = pd.DataFrame(records)
    # Fail closed on pathological labeler collapse, but retain OTHER as a valid abstention.
    health = {
        "topic_counts": frame["topic"].value_counts().to_dict(),
        "register_counts": frame["register"].value_counts().to_dict(),
        "formatting_counts": frame["formatting"].value_counts().to_dict(),
        "source_template_counts": frame["source_template"].value_counts().to_dict(),
        "document_count": len(frame),
        "split_counts": frame["split"].value_counts().to_dict(),
    }
    if frame["topic"].nunique() < 4 or frame["register"].nunique() < 4:
        save_json(run / "decisions" / "label_health.json", {**health, "status": "FAIL"})
        raise RuntimeError("external labeler collapsed below four supported classes")
    frame.to_parquet(output, index=False)
    np.save(tokens_output, np.stack(token_rows), allow_pickle=False)
    save_json(run / "decisions" / "label_health.json", {**health, "status": "PASS"})
    print(f"froze {len(frame)} labels before residual access")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
