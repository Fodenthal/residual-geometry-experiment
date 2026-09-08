from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from datasets import load_dataset

from residual_geometry.config.schema import ContextProcessingConfig, ContextSplitConfig, DatasetConfig
from residual_geometry.utils.seed import get_rng


def stable_context_id(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def preprocess_document_text(
    text: str,
    tokenizer: Any,
    processing: ContextProcessingConfig,
) -> dict | None:
    if not text or not text.strip():
        return None
    encoded = tokenizer(
        text,
        add_special_tokens=processing.add_special_tokens,
        truncation=False,
        return_attention_mask=False,
    )
    token_ids = list(encoded["input_ids"])
    if len(token_ids) < processing.min_tokens:
        return None
    token_ids = token_ids[: processing.max_tokens]
    if len(token_ids) != processing.max_tokens:
        return None
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {
        "context_id": stable_context_id(text),
        "text": decoded_text,
        "raw_text_sha256": stable_context_id(text),
        "token_length": len(token_ids),
        "token_ids": token_ids,
    }


def build_context_pool(
    dataset_config: DatasetConfig,
    tokenizer: Any,
    processing: ContextProcessingConfig,
) -> pd.DataFrame:
    dataset = load_dataset(
        dataset_config.dataset_name,
        dataset_config.dataset_config_name,
        split=dataset_config.split,
        streaming=dataset_config.streaming,
        trust_remote_code=True,
    )

    rows: list[dict] = []
    seen_ids: set[str] = set()
    scanned_rows = 0
    for row in dataset:
        scanned_rows += 1
        if scanned_rows > dataset_config.max_scan_rows:
            break
        processed = preprocess_document_text(row.get("text", ""), tokenizer, processing)
        if processed is None:
            continue
        context_id = processed["context_id"]
        if context_id in seen_ids:
            continue
        seen_ids.add(context_id)
        rows.append(
            {
                "context_id": context_id,
                "text": processed["text"],
                "token_ids": processed["token_ids"],
                "raw_text_sha256": processed["raw_text_sha256"],
                "token_length": processed["token_length"],
                "source_dataset": dataset_config.dataset_name,
                "source_config": dataset_config.dataset_config_name or "",
                "source_split": dataset_config.split,
                "scan_row": scanned_rows,
            }
        )
        if len(rows) >= dataset_config.document_count:
            break

    if len(rows) < dataset_config.document_count:
        raise ValueError(
            f"Only {len(rows)} documents available after preprocessing, "
            f"but document_count={dataset_config.document_count} was requested."
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def assign_context_splits(
    contexts_df: pd.DataFrame,
    split_config: ContextSplitConfig,
) -> pd.DataFrame:
    n = len(contexts_df)
    rng = get_rng(split_config.split_seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = round(split_config.train_fraction * n)
    n_val = round(split_config.val_fraction * n)
    labels = [""] * n
    for idx in indices[:n_train]:
        labels[idx] = "train"
    for idx in indices[n_train : n_train + n_val]:
        labels[idx] = "val"
    for idx in indices[n_train + n_val :]:
        labels[idx] = "test"

    out = contexts_df[["context_id"]].copy().reset_index(drop=True)
    out["split"] = labels
    out["split_seed"] = split_config.split_seed
    return out
