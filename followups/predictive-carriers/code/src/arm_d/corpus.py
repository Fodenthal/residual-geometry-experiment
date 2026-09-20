"""Document pools, splits, and frozen document-level nuisance strata (R1 §5).

Two pools, and the reason they differ is structural rather than bookkeeping.
Every legacy residual-geometry run drew from ``c4/en split=validation``.  The
confirmatory pool here draws from ``split=train``, so disjointness from all
previously inspected contexts follows from the split name and is verifiable
after the fact (R1 §5.3, addition A3).  The development pool reproduces the
legacy recipe on ``split=validation`` and may never carry a confirmatory number.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DEV_SPLIT = "validation"
CONFIRMATORY_SPLIT = "train"
CONTEXT_LENGTH = 1024


@dataclass(frozen=True)
class PoolSpec:
    name: str
    dataset_split: str
    document_count: int
    max_scan_rows: int
    split_seed: int
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    skip_documents: int = 0
    skip_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dataset_name"] = "allenai/c4"
        payload["dataset_config"] = "en"
        payload["context_length"] = CONTEXT_LENGTH
        payload["add_special_tokens"] = False
        return payload


DEV_POOL = PoolSpec(
    name="dev",
    dataset_split=DEV_SPLIT,
    document_count=1200,
    max_scan_rows=200_000,
    split_seed=2,
)

SMOKE_INSPECTED_DOCUMENTS = 200
"""Confirmatory documents inspected during development, excluded from production.

Two pipeline smoke runs (64 and 200 documents) read outcome-bearing statistics --
the sham-versus-main ordering, the split-half overlap, the composition error --
off the head of the deterministic confirmatory stream.  Those documents are
therefore development data.  ``build_pool`` skips the first
:data:`SMOKE_INSPECTED_DOCUMENTS` qualifying documents of the confirmatory
stream so that none of them reaches the confirmatory train, validation, or
sealed test split, and records their identifiers in the run manifest.
"""


CONFIRMATORY_POOL = PoolSpec(
    name="confirmatory",
    dataset_split=CONFIRMATORY_SPLIT,
    skip_documents=SMOKE_INSPECTED_DOCUMENTS,
    skip_reason=(
        "inspected during two development smoke runs of the analysis pipeline "
        "(64 and 200 documents) before the confirmatory pool was sealed"
    ),
    document_count=2400,
    max_scan_rows=200_000,
    split_seed=42,
)


def stable_context_id(raw_text: str) -> str:
    """sha256 of the raw document text, matching the legacy corpus contract."""

    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def build_pool(spec: PoolSpec, tokenizer) -> pd.DataFrame:
    """Stream C4 and keep documents of exactly ``CONTEXT_LENGTH`` tokens.

    Contract inherited verbatim from the original residual-geometry
    specification: drop empty documents, tokenize with
    ``add_special_tokens=False`` (no BOS), keep documents with at least 1024
    tokens, truncate to exactly 1024.
    """

    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split=spec.dataset_split,
        streaming=True,
    )
    rows: list[dict] = []
    quarantined: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for record in dataset:
        scanned += 1
        if scanned > spec.max_scan_rows:
            break
        text = record.get("text", "")
        if not text or not text.strip():
            continue
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
        token_ids = list(encoded["input_ids"])
        if len(token_ids) < CONTEXT_LENGTH:
            continue
        # Every admitted document is truncated to exactly CONTEXT_LENGTH, so the
        # retained length is constant and useless as a stratification variable.
        # The length that varies, and that the length_band scheme is about, is
        # the document's length before truncation.
        full_token_length = len(token_ids)
        token_ids = token_ids[:CONTEXT_LENGTH]
        context_id = stable_context_id(text)
        if context_id in seen:
            continue
        seen.add(context_id)
        if len(quarantined) < int(spec.skip_documents):
            # Documents already inspected during development are removed from the
            # confirmatory pool before the split assignment, so an inspected
            # document cannot re-enter the sealed test split.
            quarantined.append(context_id)
            continue
        rows.append(
            {
                "context_id": context_id,
                "token_ids": np.asarray(token_ids, dtype=np.int32),
                "token_length": len(token_ids),
                "full_token_length": int(full_token_length),
                "source_split": spec.dataset_split,
                "scan_row": scanned,
                # Carried for the R1.1 §45 stratification schemes only.  Adding
                # these fields cannot change which documents are selected or in
                # what order: the context id is a hash of the text, and neither
                # field participates in selection, so the pool is identical to
                # the one already sealed.
                "url": str(record.get("url", "") or ""),
                "text_prefix": text[:2000],
            }
        )
        if len(rows) >= spec.document_count:
            break
    if len(rows) < spec.document_count:
        raise ValueError(
            f"pool {spec.name!r}: only {len(rows)} documents after {scanned} scanned rows, "
            f"needed {spec.document_count}"
        )
    frame = pd.DataFrame(rows).reset_index(drop=True)
    frame["pool"] = spec.name
    frame.attrs["quarantined_context_ids"] = quarantined
    frame.attrs["quarantine_reason"] = spec.skip_reason
    return frame


def assign_splits(frame: pd.DataFrame, spec: PoolSpec) -> pd.DataFrame:
    """Document-level train/val/test assignment; the document is the unit."""

    n = len(frame)
    rng = np.random.default_rng(spec.split_seed)
    order = rng.permutation(n)
    n_train = int(round(spec.train_fraction * n))
    n_val = int(round(spec.val_fraction * n))
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train : n_train + n_val]] = "val"
    labels[order[n_train + n_val :]] = "test"
    out = frame.copy()
    out["split"] = labels
    out["split_seed"] = spec.split_seed
    return out


NEWLINE_MARKERS = ("\n",)


def document_features(frame: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Document-level features used for donor stratification.

    Deliberately cheap and text-only: they must be computable before any
    activation is captured, so donor matching can never depend on the
    quantity under test.
    """

    newline_ids = set()
    vocabulary = tokenizer.get_vocab()
    for token, index in vocabulary.items():
        if "\n" in token or token in {"<0x0A>"}:
            newline_ids.add(int(index))

    distinct_ratio = []
    newline_density = []
    for token_ids in frame["token_ids"]:
        array = np.asarray(token_ids)
        distinct_ratio.append(float(np.unique(array).size / array.size))
        newline_density.append(float(np.isin(array, list(newline_ids)).mean()) if newline_ids else 0.0)
    out = frame.copy()
    out["distinct_token_ratio"] = distinct_ratio
    out["newline_density"] = newline_density
    return out


def freeze_strata(frame: pd.DataFrame, n_bins: int = 4) -> dict[str, list[float]]:
    """Quantile edges frozen on the train documents only."""

    train = frame[frame["split"] == "train"]
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return {
        "distinct_token_ratio": [float(value) for value in np.quantile(train["distinct_token_ratio"], quantiles)],
        "newline_density": [float(value) for value in np.quantile(train["newline_density"], quantiles)],
    }


def apply_strata(frame: pd.DataFrame, edges: dict[str, list[float]]) -> pd.DataFrame:
    out = frame.copy()
    for column, cut_points in edges.items():
        out[f"{column}_bin"] = np.digitize(out[column].to_numpy(), np.asarray(cut_points))
    out["nuisance_stratum"] = (
        out["distinct_token_ratio_bin"].astype(str) + "_" + out["newline_density_bin"].astype(str)
    )
    return out


def hashed_token_profile(token_ids: np.ndarray, upper: int, buckets: int = 2048) -> np.ndarray:
    """L2-normalized hashed bag-of-tokens over the first ``upper`` tokens.

    Used only to find a maximally content-similar donor for the matched-content
    splice control (R1 §16.3); it never touches activations.
    """

    array = np.asarray(token_ids)[:upper]
    counts = np.bincount(array % buckets, minlength=buckets).astype(np.float64)
    norm = float(np.linalg.norm(counts))
    return counts / norm if norm > 0 else counts
