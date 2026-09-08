#!/usr/bin/env python
"""M1: GQA-safe post-W_O head ablation test.

Question: do aligned heads causally support high-τ persistence?

Ablates per-query-head post-W_O residual writes by zeroing hook_z at the ablated
layer and head. Measures collapse in J_lag-heldout probe timescales relative to
a fresh clean baseline.

For head (ℓ, h), zeroing hook_z[:, :, h, :] at layer ℓ is equivalent to zeroing
the post-W_O head write o_{ℓ,h,t} = z_{ℓ,h,t} W_O^h. This is GQA-safe: hook_z
is per-query-head and independent of KV-group structure.

Metric:
  C_head = 1 − median_j(τ_j(ablated)) / median_j(τ_j(clean))
  where τ_j is the within-document demeaned autocorrelation timescale of
  J_lag-heldout probe j, computed fresh from a forward pass through the val split.

Positive causal evidence: C_aligned_heads > C_matched_controls.
Global positive status requires aligned heads to beat ALL matched controls
under head-level comparison (C_head is already one value per head).

Bootstrap CIs use paired document resampling — same doc indices for clean and
ablated each replicate — so that correlated noise cancels in the C_head estimate.

A level-8 causal claim additionally requires M0 attention-specific geometry
(pass --m0-summary to gate the claim).

Prerequisites:
  Stage 00: resolved_model_architecture.json
  Stage 02: probe direction files
  Stage 03: probe timescales
  Stage 05: heldout_projection_eval_probes.npz (J_lag-heldout probes)
  Stage 06: alignment and PCA parquets (for auto head selection)

Head groups (auto-selected from 06 results unless --heads is specified):
  raw_top     — highest mean max_abs_alignment (ctrl_k=0)
  resid_top   — highest at max ctrl_k, deduped vs raw_top
  low_align   — lowest-alignment head in same layer
  random_ctrl — randomly chosen head in same layer (seed=0)
  high_var    — head with highest eigenvalue at component 0 (from 06 PCA)

Exploratory M1-lite:
  If locked M3 failed but per-head M3b diagnostics identify transport candidates,
  pass --exploratory-m1-lite. This uses validation-informed, post-hoc head sets
  and writes separate m1_lite artifacts. It cannot support claim level 8.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    artifact_metadata,
    load_config_and_store,
    require_residual_geometry_config,
    split_contexts,
    token_matrix,
)
from residual_geometry.autocorr.estimators import (
    compute_document_autocorr_matrix,
    extract_tau,
)
from residual_geometry.residuals.probes import load_probe_set
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.utils.io import save_json, save_parquet


# ---------------------------------------------------------------------------
# Head selection — mirrors M3 exactly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int
    head_group: str
    alignment_score: float = float("nan")


def _select_heads(
    alignment_parquet: str,
    pca_parquet: str,
    n_top: int,
    rng_seed: int,
) -> list[HeadSpec]:
    """Auto-select head groups from 06 alignment and PCA outputs."""
    align = pd.read_parquet(alignment_parquet)
    pca   = pd.read_parquet(pca_parquet)
    specs: list[HeadSpec] = []

    raw_df = (
        align[
            (align["direction_group"] == "persistent")
            & (align["resid_pca_k"] == 0)
            & align.get("residualized_alignment_valid", pd.Series(True, index=align.index)).astype(bool)
        ]
        .groupby(["layer", "head"])["max_abs_alignment"]
        .mean()
        .sort_values(ascending=False)
        .head(n_top)
        .reset_index()
    )
    raw_set: set[tuple[int, int]] = set()
    for _, row in raw_df.iterrows():
        l, h = int(row["layer"]), int(row["head"])
        specs.append(HeadSpec(l, h, "raw_top", float(row["max_abs_alignment"])))
        raw_set.add((l, h))

    max_ctrl_k = int(align["resid_pca_k"].max())
    resid_df = (
        align[
            (align["direction_group"] == "persistent")
            & (align["resid_pca_k"] == max_ctrl_k)
            & align.get("residualized_alignment_valid", pd.Series(True, index=align.index)).astype(bool)
        ]
        .groupby(["layer", "head"])["max_abs_alignment"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    n_added = 0
    for _, row in resid_df.iterrows():
        if n_added >= n_top:
            break
        l, h = int(row["layer"]), int(row["head"])
        if (l, h) in raw_set:
            continue
        specs.append(HeadSpec(l, h, "resid_top", float(row["max_abs_alignment"])))
        n_added += 1

    selected_layers = sorted({s.layer for s in specs if s.head_group in ("raw_top", "resid_top")})
    selected_set = {(s.layer, s.head) for s in specs}
    n_heads_total = int(align["head"].nunique())

    low_df = (
        align[
            (align["direction_group"] == "persistent")
            & (align["resid_pca_k"] == 0)
        ]
        .groupby(["layer", "head"])["max_abs_alignment"]
        .mean()
        .reset_index()
    )
    for layer in selected_layers:
        layer_heads = low_df[low_df["layer"] == layer].sort_values("max_abs_alignment")
        for _, row in layer_heads.iterrows():
            h = int(row["head"])
            if (layer, h) not in selected_set:
                specs.append(HeadSpec(layer, h, "low_align", float(row["max_abs_alignment"])))
                selected_set.add((layer, h))
                break

    rng = np.random.default_rng(rng_seed)
    for layer in selected_layers:
        available = [h for h in range(n_heads_total) if (layer, h) not in selected_set]
        if available:
            h = int(rng.choice(available))
            score_rows = low_df[(low_df["layer"] == layer) & (low_df["head"] == h)]["max_abs_alignment"]
            score = float(score_rows.iloc[0]) if len(score_rows) else float("nan")
            specs.append(HeadSpec(layer, h, "random_ctrl", score))
            selected_set.add((layer, h))

    hv_df = pca[pca["component"] == 0].sort_values("eigenvalue", ascending=False)
    for layer in selected_layers:
        layer_hv = hv_df[hv_df["layer"] == layer]
        for _, row in layer_hv.iterrows():
            h = int(row["head"])
            if (layer, h) not in selected_set:
                score_rows = low_df[(low_df["layer"] == layer) & (low_df["head"] == h)]["max_abs_alignment"]
                align_score = float(score_rows.iloc[0]) if len(score_rows) else float("nan")
                specs.append(HeadSpec(layer, h, "high_var", align_score))
                selected_set.add((layer, h))
                break

    return specs


def _parse_head_override(specs_str: list[str]) -> list[HeadSpec]:
    """Parse 'L12H7:raw_top' or 'L12H7' strings into HeadSpec objects."""
    result = []
    for s in specs_str:
        m = re.match(r"L(\d+)H(\d+)(?::(\w+))?", s)
        if not m:
            raise ValueError(f"Cannot parse head spec {s!r}; expected LxxHyy or LxxHyy:group")
        result.append(HeadSpec(int(m.group(1)), int(m.group(2)), m.group(3) or "manual"))
    return result


def _select_exploratory_m1_lite_heads() -> list[HeadSpec]:
    """Post-hoc head sets from M3b diagnostics.

    These are intentionally hard-coded to make the validation-informed selection explicit
    and reproducible. Do not use this mode for locked claim-level M1 evidence.
    """
    return [
        HeadSpec(12, 7, "positive_transport_candidate"),
        HeadSpec(10, 6, "positive_transport_candidate"),
        HeadSpec(11, 0, "positive_transport_candidate"),
        HeadSpec(7, 2, "positive_transport_candidate"),
        HeadSpec(12, 6, "negative_transport_candidate"),
        HeadSpec(11, 2, "transport_positive_control"),
        HeadSpec(8, 4, "transport_positive_control"),
        HeadSpec(12, 1, "transport_positive_control"),
        HeadSpec(11, 4, "low_negative_control"),
        HeadSpec(8, 6, "low_negative_control"),
        HeadSpec(12, 2, "low_negative_control"),
    ]


def _head_specs_from_summary(summary_path: str) -> list[HeadSpec]:
    with open(summary_path) as fh:
        summary = json.load(fh)
    specs = []
    for item in summary.get("head_specs", []):
        specs.append(
            HeadSpec(
                layer=int(item["layer"]),
                head=int(item["head"]),
                head_group=str(item.get("group") or item.get("head_group") or "manual"),
                alignment_score=float(item.get("alignment_score", float("nan"))),
            )
        )
    if not specs:
        raise ValueError(f"No head_specs found in existing summary: {summary_path}")
    return specs


def _resolve_m0_status(store, explicit_path: str | None) -> tuple[str, str | None]:
    """Return (status, path). Prefer explicit path; otherwise use the default M0 artifact."""
    m0_path = explicit_path
    if m0_path is None:
        candidate = os.path.join(store.subspace_dir, "block_output_subspace_summary.json")
        if Path(candidate).exists():
            m0_path = candidate
    if m0_path is None:
        return "missing", None
    if not Path(m0_path).exists():
        raise FileNotFoundError(f"--m0-summary path not found: {m0_path}")
    with open(m0_path) as fh:
        return str(json.load(fh).get("status", "missing")), m0_path


# ---------------------------------------------------------------------------
# Per-document autocorrelation profile accumulator
# ---------------------------------------------------------------------------

class PerDocProfileAccum:
    """Stores per-document within-demeaned autocorr profiles across forward-pass batches.

    Shape: (n_docs, n_probes, k_max+1) float32. NaN marks invalid (low-variance) entries.
    Supports paired-bootstrap resampling by keeping all per-document profiles.

    Memory: n_docs × n_probes × (k_max+1) × 4 bytes
      pilot (5000 docs, 128 probes, K=512): ~1.3 GB — acceptable for one accum at a time.
    """

    def __init__(self, n_docs: int, n_probes: int, k_max: int) -> None:
        self.n_docs   = n_docs
        self.n_probes = n_probes
        self.k_max    = k_max
        self.profiles = np.full((n_docs, n_probes, k_max + 1), np.nan, dtype=np.float32)

    def update(self, doc_start: int, corr_matrix: np.ndarray) -> None:
        """corr_matrix: (B, n_probes, k_max+1) float32 from compute_document_autocorr_matrix."""
        b = corr_matrix.shape[0]
        self.profiles[doc_start:doc_start + b] = corr_matrix.astype(np.float32)

    def mean_profile(self, doc_idx: np.ndarray | None = None) -> np.ndarray:
        """Mean autocorr profile. Returns (n_probes, k_max+1) float64."""
        src = self.profiles if doc_idx is None else self.profiles[doc_idx]
        return np.nanmean(src.astype(np.float64), axis=0)

    def valid_counts(self, doc_idx: np.ndarray | None = None) -> np.ndarray:
        """Number of valid (non-NaN) documents per (probe, lag). Returns (n_probes, k_max+1) int64."""
        src = self.profiles if doc_idx is None else self.profiles[doc_idx]
        return (~np.isnan(src)).sum(axis=0).astype(np.int64)


# ---------------------------------------------------------------------------
# Tau extraction
# ---------------------------------------------------------------------------

def _extract_tau_all(
    mean_profile: np.ndarray,   # (n_probes, k_max+1)
    valid_cnts: np.ndarray,     # (n_probes, k_max+1)
    k_max: int,
    min_valid_docs: int,
    min_valid_lag_frac: float,
) -> np.ndarray:
    """Extract τ_within for each probe. Returns (n_probes,) float64; NaN for invalid probes."""
    n_probes = mean_profile.shape[0]
    taus = np.full(n_probes, np.nan, dtype=np.float64)
    for j in range(n_probes):
        result = extract_tau(
            mean_profile[j],
            max_lag=k_max,
            valid_counts=valid_cnts[j],
            min_valid_docs=min_valid_docs,
            min_valid_lag_fraction=min_valid_lag_frac,
        )
        if result["tau_valid"]:
            taus[j] = float(result["tau"])
    return taus


# ---------------------------------------------------------------------------
# Forward pass with optional ablation
# ---------------------------------------------------------------------------

def _validate_hooks(model, device: torch.device, layers: list[int], resid_hook: str) -> None:
    """Probe once that hook_z at each ablated layer and the residual hook are accessible."""
    probe = torch.zeros((1, 8), dtype=torch.long, device=device)
    needed = {f"blocks.{l}.attn.hook_z" for l in layers} | {resid_hook}
    with torch.inference_mode():
        _, cache = model.run_with_cache(probe, names_filter=lambda n: n in needed, return_type=None)
    missing = [n for n in needed if n not in cache]
    if missing:
        raise RuntimeError(
            f"M1 requires hooks not in cache: {missing[:3]}{'...' if len(missing) > 3 else ''}. "
            "Check TransformerLens version."
        )


def _run_collection_pass(
    model,
    device: torch.device,
    tokens: np.ndarray,
    resid_hook: str,
    eval_dirs: torch.Tensor,    # (n_eval, d_model) float32 on device
    k_max: int,
    batch_size: int,
    n_docs: int,
    ablate_hook: str | None = None,
    ablate_head: int | None = None,
    profile_batches: int | None = None,
) -> PerDocProfileAccum:
    """One forward pass over tokens, collecting per-doc within-demeaned autocorr profiles.

    Clean pass: ablate_hook=None. Ablated pass: zeros hook_z[:, :, ablate_head, :].
    Uses model.run_with_hooks (not run_with_cache) to combine capture and ablation.
    """
    n_eval = eval_dirs.shape[0]
    d_model = eval_dirs.shape[1]
    accum = PerDocProfileAccum(n_docs, n_eval, k_max)
    resid_storage: dict[str, torch.Tensor] = {}

    def _capture(resid: torch.Tensor, hook) -> torch.Tensor:
        resid_storage["r"] = resid.detach()
        return resid

    if ablate_hook is not None and ablate_head is not None:
        _head = ablate_head

        def _ablate(z: torch.Tensor, hook) -> torch.Tensor:
            z = z.clone()
            z[:, :, _head, :] = 0.0
            return z

        fwd_hooks = [(ablate_hook, _ablate), (resid_hook, _capture)]
    else:
        fwd_hooks = [(resid_hook, _capture)]

    doc_offset = 0
    for batch_i, start in enumerate(range(0, len(tokens), batch_size)):
        batch = tokens[start: start + batch_size]
        b = batch.shape[0]
        batch_tok = torch.as_tensor(batch, dtype=torch.long, device=device)

        t_fwd = time.perf_counter()
        with torch.inference_mode():
            model.run_with_hooks(batch_tok, return_type=None, fwd_hooks=fwd_hooks)
        t_fwd = time.perf_counter() - t_fwd

        t_cpu = time.perf_counter()
        resid = resid_storage["r"].float()                            # (b, seq, d_model)
        seq_len = resid.shape[1]
        proj_gpu = torch.mm(
            resid.reshape(-1, d_model), eval_dirs.T
        ).reshape(b, seq_len, n_eval)                                 # (b, seq, n_eval)
        proj = proj_gpu.cpu().numpy()
        del resid, proj_gpu

        corr_matrix, _ = compute_document_autocorr_matrix(proj, max_lag=k_max, estimator="within")
        # corr_matrix: (b, n_eval, k_max+1)
        accum.update(doc_offset, corr_matrix)
        doc_offset += b
        del proj, corr_matrix
        t_cpu = time.perf_counter() - t_cpu

        if profile_batches is not None:
            print(f"  batch {batch_i}: fwd={t_fwd:.2f}s cpu={t_cpu:.2f}s")
            if batch_i + 1 >= profile_batches:
                break

    return accum


# ---------------------------------------------------------------------------
# C_head and bootstrap
# ---------------------------------------------------------------------------

def _c_head_from_accums(
    clean_accum: PerDocProfileAccum,
    ablated_accum: PerDocProfileAccum,
    k_max: int,
    min_valid_docs: int,
    min_valid_lag_frac: float,
    doc_idx: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute C_head, clean median τ, ablated median τ for a given doc sample."""
    clean_tau = _extract_tau_all(
        clean_accum.mean_profile(doc_idx), clean_accum.valid_counts(doc_idx),
        k_max, min_valid_docs, min_valid_lag_frac,
    )
    ablated_tau = _extract_tau_all(
        ablated_accum.mean_profile(doc_idx), ablated_accum.valid_counts(doc_idx),
        k_max, min_valid_docs, min_valid_lag_frac,
    )
    clean_med   = float(np.nanmedian(clean_tau))
    ablated_med = float(np.nanmedian(ablated_tau))
    c_head = float(1.0 - ablated_med / clean_med) if clean_med > 0 else float("nan")
    return c_head, clean_med, ablated_med


def _bootstrap_c_head(
    clean_accum: PerDocProfileAccum,
    ablated_accum: PerDocProfileAccum,
    k_max: int,
    min_valid_docs: int,
    min_valid_lag_frac: float,
    seed: int,
    reps: int,
) -> tuple[float, float]:
    """Paired-bootstrap 95% CI for C_head. Returns (ci_low, ci_high)."""
    n_docs = clean_accum.n_docs
    rng = np.random.default_rng(seed)
    boot_c = np.full(reps, np.nan, dtype=np.float64)
    for rep in range(reps):
        idx = rng.integers(0, n_docs, size=n_docs)
        c, _, _ = _c_head_from_accums(
            clean_accum, ablated_accum, k_max, min_valid_docs, min_valid_lag_frac, idx
        )
        boot_c[rep] = c
    valid = ~np.isnan(boot_c)
    if valid.sum() < 10:
        return float("nan"), float("nan")
    return float(np.nanquantile(boot_c, 0.025)), float(np.nanquantile(boot_c, 0.975))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _compute_summary(
    rows: pd.DataFrame,
    head_specs: list[HeadSpec],
    metadata: dict[str, object],
    m0_status: str = "missing",
) -> dict[str, object]:
    """Head-group summary with M0-gated level-8 claim.

    C_head is already one value per (split, layer, head); no further per-probe
    reduction needed. Group comparison uses head-level medians.
    Positive status requires aligned > ALL matched controls.
    """
    ALIGNED  = {"raw_top", "resid_top"}
    CONTROLS = ["low_align", "random_ctrl", "high_var"]

    summary: dict[str, object] = {
        **metadata,
        "metric": "C_head = 1 - median_j(tau_j_ablated) / median_j(tau_j_clean)",
        "eval_probe_family": "lag_heldout",
        "bootstrap_method": "paired document resampling (same indices for clean and ablated)",
        "by_head_group": {},
        "aligned_vs_control": {},
    }

    # One C_head per (split, layer, head) — take first row since it's constant within the group
    head_level = (
        rows.groupby(["split", "layer", "head", "head_group"])["c_head"]
        .first()
        .reset_index()
    )

    for group in sorted(rows["head_group"].dropna().unique()):
        grp = head_level[head_level["head_group"] == group]
        summary["by_head_group"][str(group)] = {
            "head_count": int(len(grp)),
            "median_c_head": float(grp["c_head"].median()) if len(grp) else float("nan"),
            "q90_c_head":    float(grp["c_head"].quantile(0.9)) if len(grp) else float("nan"),
        }

    aligned_c = head_level[head_level["head_group"].isin(ALIGNED)]["c_head"].dropna()
    all_controls_positive = True
    for ctrl_grp in CONTROLS:
        ctrl_c = head_level[head_level["head_group"] == ctrl_grp]["c_head"].dropna()
        if len(aligned_c) and len(ctrl_c):
            diff = float(aligned_c.median() - ctrl_c.median())
            positive = bool(diff > 0)
        else:
            diff = float("nan")
            positive = False
        summary["aligned_vs_control"][ctrl_grp] = {
            "aligned_median_c": float(aligned_c.median()) if len(aligned_c) else float("nan"),
            "control_median_c": float(ctrl_c.median()) if len(ctrl_c) else float("nan"),
            "difference": diff,
            f"positive_vs_{ctrl_grp}": positive,
        }
        if not positive:
            all_controls_positive = False

    causal_statistic_positive = bool(all_controls_positive and len(aligned_c) > 0)
    summary["causal_statistic_positive"] = causal_statistic_positive
    summary["positive_vs_all_controls"] = causal_statistic_positive

    # M0-gated level-8 claim
    summary["m0_status"] = m0_status
    claim_supported = causal_statistic_positive and m0_status == "attention_specific"
    summary["claim_level_8_supported"] = claim_supported

    if claim_supported:
        interpretation = (
            "Ablating aligned heads reduces J_lag-heldout timescales more than matched controls, "
            "with M0 attention-specific geometry: level-8 causal head-support claim supported."
        )
    elif causal_statistic_positive and m0_status == "missing":
        interpretation = (
            "Causal statistic positive, but M0 is missing; "
            "do not claim attention-specific causal support. Provide --m0-summary to evaluate."
        )
    elif causal_statistic_positive:
        interpretation = (
            f"Causal statistic positive, but M0 geometry is not attention-specific "
            f"(M0 status: {m0_status}); ablation effect may reflect generic residual structure."
        )
    else:
        interpretation = "Causal statistic not positive."
    summary["interpretation"] = interpretation
    summary["status"] = "positive_causal" if causal_statistic_positive else "not_positive_or_not_evaluable"

    return summary


def _effect_direction(c_head: float) -> str:
    if np.isnan(c_head):
        return "not_evaluable"
    if c_head > 0:
        return "ablation_reduced_persistence"
    if c_head < 0:
        return "ablation_increased_persistence"
    return "no_median_change"


def _compute_exploratory_m1_lite_summary(
    rows: pd.DataFrame,
    head_specs: list[HeadSpec],
    metadata: dict[str, object],
    m0_status: str = "missing",
) -> dict[str, object]:
    """Exploratory post-hoc M1-lite summary.

    This is deliberately separate from the locked M1 summary: the head sets are selected
    from validation-stage M3b diagnostics, so claim_level_8_supported is always false.
    """
    head_level = (
        rows.groupby(["split", "layer", "head", "head_group"])
        .agg(
            c_head=("c_head", "first"),
            clean_median_tau=("clean_median_tau", "first"),
            ablated_median_tau=("ablated_median_tau", "first"),
            median_delta_tau=("delta_tau", "median"),
            bootstrap_ci_low_c_head=("bootstrap_ci_low_c_head", "first"),
            bootstrap_ci_high_c_head=("bootstrap_ci_high_c_head", "first"),
        )
        .reset_index()
    )
    head_level["effect_direction"] = head_level["c_head"].map(_effect_direction)

    summary: dict[str, object] = {
        **metadata,
        "status": "exploratory_completed",
        "exploratory": True,
        "post_hoc_head_selection": True,
        "head_selection_source": "M3 validation per-head transport diagnostics",
        "claim_level_8_supported": False,
        "m0_status": m0_status,
        "metric": "C_head = 1 - median_j(tau_j_ablated) / median_j(tau_j_clean)",
        "eval_probe_family": "lag_heldout",
        "bootstrap_method": "paired document resampling (same indices for clean and ablated)",
        "interpretation": (
            "Exploratory M1-lite only. Head sets were selected post hoc from M3b diagnostics; "
            "results cannot support locked claim level 8."
        ),
        "by_head_group": {},
        "by_head": [],
        "exploratory_comparisons": {},
    }

    for group in sorted(head_level["head_group"].dropna().unique()):
        grp = head_level[head_level["head_group"] == group]
        summary["by_head_group"][str(group)] = {
            "head_count": int(len(grp)),
            "median_c_head": float(grp["c_head"].median()) if len(grp) else float("nan"),
            "q90_c_head": float(grp["c_head"].quantile(0.9)) if len(grp) else float("nan"),
            "effect_direction_counts": {
                str(k): int(v) for k, v in grp["effect_direction"].value_counts().to_dict().items()
            },
        }

    def _median_for(group: str) -> float:
        values = head_level[head_level["head_group"] == group]["c_head"].dropna()
        return float(values.median()) if len(values) else float("nan")

    pos_med = _median_for("positive_transport_candidate")
    for ctrl in ["transport_positive_control", "low_negative_control"]:
        ctrl_med = _median_for(ctrl)
        summary["exploratory_comparisons"][f"positive_transport_candidate_vs_{ctrl}"] = {
            "candidate_median_c": pos_med,
            "control_median_c": ctrl_med,
            "difference": float(pos_med - ctrl_med) if not np.isnan(pos_med) and not np.isnan(ctrl_med) else float("nan"),
            "candidate_gt_control": bool(pos_med > ctrl_med) if not np.isnan(pos_med) and not np.isnan(ctrl_med) else False,
        }

    for row in head_level.sort_values(["head_group", "layer", "head"]).itertuples(index=False):
        item = {
            "split": str(row.split),
            "layer": int(row.layer),
            "head": int(row.head),
            "group": str(row.head_group),
            "c_head": float(row.c_head),
            "clean_median_tau": float(row.clean_median_tau),
            "ablated_median_tau": float(row.ablated_median_tau),
            "median_delta_tau": float(row.median_delta_tau),
            "bootstrap_ci_low_c_head": float(row.bootstrap_ci_low_c_head),
            "bootstrap_ci_high_c_head": float(row.bootstrap_ci_high_c_head),
            "effect_direction": str(row.effect_direction),
        }
        summary["by_head"].append(item)
        if int(row.layer) == 12 and int(row.head) == 6:
            summary["l12h6_signed_effect"] = {
                **item,
                "interpretation_rule": (
                    "If ablation increases persistence (negative C_head), treat as exploratory "
                    "suppressive-head evidence; if it reduces persistence, treat as support-like "
                    "despite negative M3 transport sign."
                ),
            }

    summary["head_specs"] = [
        {"layer": s.layer, "head": s.head, "group": s.head_group, "alignment_score": s.alignment_score}
        for s in head_specs
    ]
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="M1: GQA-safe post-W_O head ablation test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-top-heads", type=int, default=5,
                        help="Number of top heads per group to select from 06 results.")
    parser.add_argument("--heads", nargs="+", default=None,
                        help="Manual head override, e.g. L12H7:raw_top L11H0:resid_top. "
                             "Bypasses auto-selection.")
    parser.add_argument("--exploratory-m1-lite", action="store_true",
                        help="Use post-hoc M3b head sets and write separate m1_lite artifacts. "
                             "This mode cannot support claim_level_8_supported.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Docs per forward pass. M1 runs one full pass per head, so "
                             "memory per pass is low; the cost is total forward-pass count.")
    parser.add_argument("--k-max", type=int, default=512,
                        help="Maximum autocorr lag for ablated-residual timescale extraction.")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test"])
    parser.add_argument("--m0-summary", default=None,
                        help="Path to M0 summary JSON. Required to gate claim_level_8_supported.")
    parser.add_argument("--min-valid-lag-fraction", type=float, default=0.8,
                        help="Minimum fraction of lags with valid doc counts for tau_valid=True.")
    parser.add_argument("--profile-batches", type=int, default=None,
                        help="Stop each pass after this many batches (for timing profiling).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Recompute the M1 summary JSON from the existing parquet without "
                             "rerunning model forwards. Use with --exploratory-m1-lite for "
                             "M1-lite artifacts.")
    args = parser.parse_args()

    config, store = load_config_and_store(args.config)
    residual_cfg  = require_residual_geometry_config(config)
    if args.exploratory_m1_lite:
        output_path = os.path.join(store.subspace_dir, "head_ablation_m1_lite_autocorr.parquet")
        summary_path = os.path.join(store.subspace_dir, "head_ablation_m1_lite_summary.json")
        stage_name = "residual_geometry_M1_lite_head_ablation"
    else:
        output_path = os.path.join(store.subspace_dir, "head_ablation_autocorr.parquet")
        summary_path = os.path.join(store.subspace_dir, "head_ablation_summary.json")
        stage_name = "residual_geometry_M1_head_ablation"
    metadata = artifact_metadata(config, stage_name)

    if args.summary_only:
        if not Path(output_path).exists():
            raise FileNotFoundError(f"Missing existing M1 parquet: {output_path}")
        if not Path(summary_path).exists():
            raise FileNotFoundError(f"Missing existing M1 summary for head_specs: {summary_path}")
        result = pd.read_parquet(output_path)
        head_specs = _head_specs_from_summary(summary_path)
        m0_status, m0_path = _resolve_m0_status(store, args.m0_summary)
        if args.exploratory_m1_lite:
            summary = _compute_exploratory_m1_lite_summary(
                result, head_specs, metadata, m0_status=m0_status
            )
        else:
            summary = _compute_summary(result, head_specs, metadata, m0_status=m0_status)
        summary["m0_summary_path"] = m0_path
        summary["summary_only_recompute"] = True
        previous = json.load(open(summary_path))
        for key in ["residual_hook", "eval_probe_count", "k_max", "gqa_architecture"]:
            if key in previous:
                summary[key] = previous[key]
        save_json(summary, summary_path)
        return 0

    if Path(output_path).exists() and Path(summary_path).exists() and not args.overwrite:
        return 0

    alignment_path = os.path.join(store.subspace_dir, "residual_direction_head_alignment.parquet")
    pca_path       = os.path.join(store.subspace_dir, "head_output_pca.parquet")
    heldout_path   = os.path.join(store.subspace_dir, "heldout_projection_eval_probes.npz")
    arch_path      = store.resolved_model_architecture_path

    # Upfront prerequisite checks — fail before loading the model
    if not Path(heldout_path).exists():
        raise FileNotFoundError(
            f"Missing J_lag-heldout probe file at {heldout_path}. "
            "Run 05_projection_collapse.py first."
        )
    if not Path(arch_path).exists():
        raise FileNotFoundError(
            f"Missing architecture metadata at {arch_path}. "
            "Run 00_validate_env.py first."
        )

    if args.exploratory_m1_lite and args.heads:
        raise ValueError("--exploratory-m1-lite uses fixed post-hoc head sets; do not pass --heads.")
    if args.exploratory_m1_lite:
        head_specs = _select_exploratory_m1_lite_heads()
    elif args.heads:
        head_specs = _parse_head_override(args.heads)
    else:
        if not Path(alignment_path).exists():
            raise FileNotFoundError(
                f"Missing Stage 06 alignment parquet at {alignment_path}; run 06 first."
            )
        if not Path(pca_path).exists():
            raise FileNotFoundError(
                f"Missing Stage 06 head-output PCA parquet at {pca_path}; run 06 first."
            )
        head_specs = _select_heads(alignment_path, pca_path, args.n_top_heads, rng_seed=0)

    # Architecture metadata (recorded in summary, not used for logic)
    with open(arch_path) as fh:
        arch = json.load(fh)

    # Load J_lag-heldout evaluation probes
    heldout_all = load_probe_set(heldout_path)
    lag_mask = heldout_all.probe_family.astype(str) == "lag_heldout"
    if not lag_mask.any():
        # Legacy probe IDs use the raw family letter; fall back to prefix match
        lag_mask = np.char.startswith(heldout_all.probe_ids.astype(str), "heldout_time_lagged_residual_")
    if not lag_mask.any():
        raise ValueError(
            f"No lag_heldout probes found in {heldout_path}. "
            f"Families present: {sorted(set(heldout_all.probe_family.astype(str).tolist()))}"
        )
    eval_dirs_np  = heldout_all.directions[lag_mask].astype(np.float32)  # (n_eval, d_model)
    eval_probe_ids = heldout_all.probe_ids[lag_mask].astype(str)
    n_eval = len(eval_probe_ids)

    # M0 status for claim gating. Defaults to the M0 artifact in this run if present.
    m0_status, m0_path = _resolve_m0_status(store, args.m0_summary)

    provider = ResidualStreamProvider(config)
    provider.load()
    model  = provider.model
    device = provider.device
    n_model_heads = int(model.cfg.n_heads)
    resid_hook    = config.model.hook_name   # e.g. "blocks.12.hook_resid_post"

    # Validate head indices
    invalid = [(s.layer, s.head) for s in head_specs if s.head >= n_model_heads]
    if invalid:
        raise ValueError(f"Head index out of range for n_heads={n_model_heads}: {invalid}")

    unique_ablate_layers = sorted({s.layer for s in head_specs})
    _validate_hooks(model, device, unique_ablate_layers, resid_hook)

    eval_dirs_gpu = torch.as_tensor(eval_dirs_np, dtype=torch.float32, device=device)
    batch_size = args.batch_size or residual_cfg.projections.batch_size
    k_max      = args.k_max

    all_rows: list[dict] = []

    for split in args.splits:
        tokens = token_matrix(split_contexts(store, split))
        n_docs = len(tokens)
        min_valid_docs = max(1, n_docs // 10)

        # Clean baseline — one forward pass, no ablation
        clean_accum = _run_collection_pass(
            model=model, device=device, tokens=tokens,
            resid_hook=resid_hook, eval_dirs=eval_dirs_gpu,
            k_max=k_max, batch_size=batch_size, n_docs=n_docs,
            profile_batches=args.profile_batches,
        )
        clean_tau_all = _extract_tau_all(
            clean_accum.mean_profile(), clean_accum.valid_counts(),
            k_max, min_valid_docs, args.min_valid_lag_fraction,
        )
        clean_median_tau = float(np.nanmedian(clean_tau_all))

        for spec in head_specs:
            ablate_hook = f"blocks.{spec.layer}.attn.hook_z"
            seed = spec.layer * 1000 + spec.head

            ablated_accum = _run_collection_pass(
                model=model, device=device, tokens=tokens,
                resid_hook=resid_hook, eval_dirs=eval_dirs_gpu,
                k_max=k_max, batch_size=batch_size, n_docs=n_docs,
                ablate_hook=ablate_hook, ablate_head=spec.head,
                profile_batches=args.profile_batches,
            )
            ablated_tau_all = _extract_tau_all(
                ablated_accum.mean_profile(), ablated_accum.valid_counts(),
                k_max, min_valid_docs, args.min_valid_lag_fraction,
            )
            ablated_median_tau = float(np.nanmedian(ablated_tau_all))
            c_head = (
                float(1.0 - ablated_median_tau / clean_median_tau)
                if clean_median_tau > 0 else float("nan")
            )
            signed_effect_direction = _effect_direction(c_head)

            ci_lo, ci_hi = _bootstrap_c_head(
                clean_accum, ablated_accum,
                k_max=k_max, min_valid_docs=min_valid_docs,
                min_valid_lag_frac=args.min_valid_lag_fraction,
                seed=seed, reps=args.bootstrap_reps,
            )

            for probe_i, probe_id in enumerate(eval_probe_ids):
                all_rows.append({
                    "split":               split,
                    "layer":               int(spec.layer),
                    "head":                int(spec.head),
                    "head_group":          spec.head_group,
                    "head_alignment_score": float(spec.alignment_score),
                    "probe_id":            str(probe_id),
                    "tau_within_clean":    float(clean_tau_all[probe_i]),
                    "tau_within_ablated":  float(ablated_tau_all[probe_i]),
                    "delta_tau":           float(clean_tau_all[probe_i] - ablated_tau_all[probe_i]),
                    "c_head":              float(c_head),
                    "bootstrap_ci_low_c_head":  float(ci_lo),
                    "bootstrap_ci_high_c_head": float(ci_hi),
                    "clean_median_tau":    float(clean_median_tau),
                    "ablated_median_tau":  float(ablated_median_tau),
                    "signed_effect_direction": signed_effect_direction,
                    "post_hoc_head_selection": bool(args.exploratory_m1_lite),
                    "head_selection_source": (
                        "M3 validation per-head transport diagnostics"
                        if args.exploratory_m1_lite else "locked M1 selection"
                    ),
                    "n_docs":              int(n_docs),
                    "k_max":               int(k_max),
                })

    result = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    for key, value in metadata.items():
        result[key] = value
    save_parquet(result, output_path)

    if args.exploratory_m1_lite:
        summary = _compute_exploratory_m1_lite_summary(
            result, head_specs, metadata, m0_status=m0_status
        )
    else:
        summary = _compute_summary(result, head_specs, metadata, m0_status=m0_status)
    summary["residual_hook"]    = resid_hook
    summary["m0_summary_path"]  = m0_path
    summary["eval_probe_count"] = int(n_eval)
    summary["k_max"]            = int(k_max)
    summary["gqa_architecture"] = {
        "n_query_heads":             arch.get("n_heads"),
        "n_kv_groups":               arch.get("n_key_value_heads") or arch.get("query_groups"),
        "per_layer_attention_type":  arch.get("per_layer_attention_type"),
        "target_layer_attention_type": arch.get("target_layer_attention_type"),
        "sliding_window_size":       arch.get("sliding_window_size"),
        "target_layer_full_1024_context": arch.get(
            "target_layer_full_1024_context_if_window_ge_1024"
        ),
    }
    summary["head_specs"] = [
        {"layer": s.layer, "head": s.head, "group": s.head_group, "alignment_score": s.alignment_score}
        for s in head_specs
    ]
    save_json(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
