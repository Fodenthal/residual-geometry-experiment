from __future__ import annotations

import pandas as pd

from residual_geometry.config.schema import ContextProcessingConfig
from residual_geometry.data.contexts import assign_context_splits, preprocess_document_text, stable_context_id
from residual_geometry.config.schema import ContextSplitConfig


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False, return_attention_mask=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(f"tok{i}" for i in token_ids)


def test_stable_context_id_uses_raw_text() -> None:
    assert stable_context_id("a b c") != stable_context_id("a b c ")


def test_preprocess_requires_exact_fixed_length_after_truncation() -> None:
    tokenizer = DummyTokenizer()
    processing = ContextProcessingConfig(min_tokens=512, max_tokens=512)
    assert preprocess_document_text("x " * 511, tokenizer, processing) is None
    record = preprocess_document_text("x " * 600, tokenizer, processing)
    assert record is not None
    assert record["token_length"] == 512


def test_assign_context_splits_expected_counts() -> None:
    contexts = pd.DataFrame({"context_id": [f"c{i}" for i in range(50)]})
    split = assign_context_splits(contexts, ContextSplitConfig())
    assert split["split"].value_counts().to_dict() == {"train": 40, "val": 5, "test": 5}

