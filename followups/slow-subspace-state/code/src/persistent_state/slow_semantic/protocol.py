from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL: dict[str, Any] = {
    "protocol_revision": "slow_subspace_semantic_audit_r1",
    "model": "google/gemma-2-2b",
    "model_revision": "main",
    "tokenizer_revision": "main",
    "hook": "blocks.12.hook_resid_post",
    "layer": 12,
    "d_model": 2304,
    "corpus": "allenai/c4",
    "corpus_config": "en",
    "corpus_split": "train",
    "context_length": 1024,
    "documents": 1200,
    "split_counts": {"fit": 800, "validation": 200, "test": 200},
    "split_seed": 730241,
    "sample_seed": 730239,
    "positions": [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960],
    "slow_ranks": [31],
    "pca_parent_rank": 256,
    "random_bases_per_rank": 20,
    "random_basis_seed": 730251,
    "bootstrap_replicates": 1000,
    "bootstrap_seed": 730253,
    "regularization_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    "cv_folds": 5,
    "cv_seed": 730257,
    "local_embedding_pca_rank": 32,
    "topic_classes": [
        "technology_and_software", "science_and_medicine", "business_and_finance",
        "politics_and_law", "arts_and_entertainment", "sports", "travel_and_places",
        "home_food_and_lifestyle", "education_and_reference", "other_or_unclear",
    ],
    "register_classes": [
        "encyclopedic_or_expository", "news_or_reporting", "technical_or_reference",
        "instructional", "legal_or_administrative", "conversational_or_forum",
        "narrative_or_creative", "other_or_unclear",
    ],
    "labeler_model": "google/gemma-2-9b-it",
    "labeler_revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "labeler_max_input_tokens": 768,
    "labeler_abstention_rule": "emit other_or_unclear unless exactly one frozen label is returned",
    "primary_effect_health_nats": 0.01,
    "pca_competitive_point_margin_nats": -0.002,
    "pca_competitive_ci_lower_nats": -0.005,
    "semantic_retention": 0.90,
    "semantic_ranks": [1, 2, 4, 8],
    "semantic_shuffle_replicates": 50,
    "original_basis_sha256": "966736e0346dd354559f1354efa86f71c4e6c01cbb929ff70b18f7a9420cce6c",
    "original_basis_rank": 31,
    "sae_policy": "skip unless a pretrained layer-12 resid_post SAE with exact normalization provenance is supplied",
    "minimum_compute_amendment": {
        "status": "PROSPECTIVE_BEFORE_ANY_SLURM_SUBMISSION_OR_RESULT_ACCESS",
        "fitted_targets": ["topic", "register"],
        "primary_rank": 31,
        "matched_controls": ["pca_31", "20_random_31_in_fit_pca256"],
        "rank8_policy": "separate extension only after a positive or boundary primary result",
        "random_extension_policy": "extend beyond 20 only after a boundary primary result",
        "sample_extension_policy": "add documents only after validation demonstrates inadequate precision",
        "bootstrap_policy": "cache held-out per-document losses; never refit inside bootstrap",
    },
    "labeler_checkpoint_amendment": {
        "status": "PROSPECTIVE_BEFORE_DOCUMENT_SELECTION_LABELING_OR_RESIDUAL_ACCESS",
        "reason": "the originally proposed 2B instruction checkpoint was not cached or authenticated on Beehive",
        "replacement": "existing provenance-addressed local Gemma-2-9B-IT snapshot",
        "scientific_constants_unchanged": ["prompt", "topic ontology", "register ontology", "abstention rule"],
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_basis(path: str | Path) -> tuple[np.ndarray, str]:
    """Load a basis without guessing silently among incompatible arrays."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as artifact:
        candidates = []
        for key in ("directions", "basis", "Q", "components"):
            if key in artifact:
                array = np.asarray(artifact[key])
                if array.ndim == 2:
                    candidates.append((key, array))
        if len(candidates) != 1:
            keys = sorted(artifact.files)
            raise ValueError(f"Expected one recognized 2D basis array, found {candidates}; keys={keys}")
        key, basis = candidates[0]
    if basis.shape == (31, 2304):
        basis = basis.T
    if basis.shape != (2304, 31):
        raise ValueError(f"Slow basis has shape {basis.shape}, expected (2304, 31)")
    basis = np.asarray(basis, dtype=np.float64)
    error = float(np.linalg.norm(basis.T @ basis - np.eye(31), ord=2))
    if error > 1e-5:
        raise ValueError(f"Slow basis orthonormality error {error:.3e} exceeds 1e-5")
    return basis, key


def protocol_hash() -> str:
    payload = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def deterministic_document_split(document_ids: list[str]) -> list[str]:
    counts = PROTOCOL["split_counts"]
    if len(document_ids) != sum(counts.values()):
        raise ValueError("document count does not match frozen split contract")
    rng = np.random.default_rng(PROTOCOL["split_seed"])
    order = rng.permutation(len(document_ids))
    labels = np.empty(len(document_ids), dtype=object)
    start = 0
    for split, count in counts.items():
        labels[order[start : start + count]] = split
        start += count
    return labels.tolist()


def orthonormal_random_controls(parent: np.ndarray, rank: int, count: int, seed: int) -> list[np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    if parent.ndim != 2 or rank > parent.shape[1]:
        raise ValueError("invalid parent/rank")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        q, _ = np.linalg.qr(rng.standard_normal((parent.shape[1], rank)))
        out.append(parent @ q[:, :rank])
    return out
