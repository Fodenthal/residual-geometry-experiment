#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from _common import artifact_metadata, load_config_and_store
from residual_geometry.data.contexts import assign_context_splits, build_context_pool
from residual_geometry.utils.io import save_json, save_parquet
from residual_geometry.utils.logging import get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Build residual-geometry context pool.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logger = get_logger(__name__)
    config, store = load_config_and_store(args.config)
    if not Path(store.resolved_model_architecture_path).exists():
        raise FileNotFoundError(
            f"Missing {store.resolved_model_architecture_path}; run 00_validate_env.py without --skip-model-load first."
        )
    if Path(store.context_pool_path).exists() and Path(store.context_split_path).exists() and not args.overwrite:
        logger.info("Context artifacts already exist; use --overwrite to rebuild.")
        return 0
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pool = build_context_pool(config.dataset, tokenizer, config.context_processing)
    split = assign_context_splits(pool, config.context_split)
    save_parquet(pool, store.context_pool_path)
    save_parquet(split, store.context_split_path)
    save_json(
        {
            **artifact_metadata(config, "residual_geometry_01_build_context_pool"),
            "stage": "residual_geometry_01_build_context_pool",
            "status": "PASS",
            "document_count": len(pool),
            "token_length": int(pool["token_length"].iloc[0]) if len(pool) else None,
            "split_counts": split["split"].value_counts().to_dict(),
        },
        store.qc_summary_path("residual_geometry_stage_01"),
    )
    logger.info("Saved context pool to %s", store.contexts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
