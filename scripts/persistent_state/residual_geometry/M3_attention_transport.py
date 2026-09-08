#!/usr/bin/env python
"""M3: Attention transport test.

Question: do aligned heads route scalar persistent state from prior source positions into
destination-position head writes?

For a persistent direction v and head (ℓ, h):
  source state:  s_s = (r_pre[s])ᵀ v               # residual pre-attention at layer ℓ

  Primary transport estimate (no-self):
    ŝ_t^{no-self} = Σ_{s<t} A^{ℓ,h}_{t,s} s_s / Σ_{s<t} A^{ℓ,h}_{t,s}
                                                   # renormalized over prior positions only

  Sensitivity estimate (include-self):
    ŝ_t^{incl}   = Σ_{s≤t} A^{ℓ,h}_{t,s} s_s     # raw attention-weighted; may conflate
                                                   # same-token dependence with routing

  Head write onto v:
    o_t = (z^{ℓ,h}_t W_O^h)ᵀ v

Primary metric:     Pearson r(ŝ_t^{no-self}, o_t) — cross-position routing only.
Sensitivity metric: Pearson r(ŝ_t^{incl},   o_t) — reported alongside primary.

Both pooled over (document, destination_position) pairs with t ≥ min_destination_position.
Bootstrap CI is document-level (resample documents).

Positive evidence: corr(aligned heads) > corr(matched controls), under head-level median
aggregation (median over persistent directions per head, then compare group medians).
Global positive status requires aligned heads to beat ALL matched controls.

A level-7b attention-routing claim additionally requires M0 to show attention-specific
geometry beyond residual PCA (pass --m0-summary to gate the claim).

Head groups (auto-selected from 06 alignment results unless --heads is specified):
  raw_top       — highest mean max_abs_alignment (ctrl_k=0, persistent directions)
  resid_top     — highest mean max_abs_alignment at ctrl_k=128 (deduped vs raw_top)
  low_align     — lowest-alignment head in same layer as each selected head
  random_ctrl   — randomly chosen head in same layer (seed=0, not in selected)
  high_var      — head with highest eigenvalue at component 0 in same layer (from 06 PCA)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
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
from residual_geometry.residuals.probes import ResidualProbeSet, load_many_probe_sets
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.subspace.residual_geometry import deduplicate_ranked_probes
from residual_geometry.utils.io import save_json, save_parquet


# ---------------------------------------------------------------------------
# Head selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int
    head_group: str   # raw_top | resid_top | low_align | random_ctrl | high_var
    alignment_score: float = float("nan")


def _select_heads(
    alignment_parquet: str,
    pca_parquet: str,
    n_top: int,
    rng_seed: int,
) -> list[HeadSpec]:
    """Auto-select five head groups from 06 alignment and PCA outputs."""
    align = pd.read_parquet(alignment_parquet)
    pca   = pd.read_parquet(pca_parquet)
    specs: list[HeadSpec] = []

    # --- raw_top: top N heads by mean max_abs_alignment (ctrl_k=0, persistent) ---
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

    # --- resid_top: top N by mean max_abs_alignment at highest ctrl_k, deduped ---
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

    # Unique layers spanned by selected heads
    selected_layers = sorted({s.layer for s in specs if s.head_group in ("raw_top", "resid_top")})
    selected_set = {(s.layer, s.head) for s in specs}
    n_heads_total = int(align["head"].nunique())

    # --- low_align: lowest-alignment head at each selected layer ---
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

    # --- random_ctrl: random head at each selected layer not already chosen ---
    rng = np.random.default_rng(rng_seed)
    for layer in selected_layers:
        available = [h for h in range(n_heads_total) if (layer, h) not in selected_set]
        if available:
            h = int(rng.choice(available))
            score_rows = low_df[(low_df["layer"] == layer) & (low_df["head"] == h)]["max_abs_alignment"]
            score = float(score_rows.iloc[0]) if len(score_rows) else float("nan")
            specs.append(HeadSpec(layer, h, "random_ctrl", score))
            selected_set.add((layer, h))

    # --- high_var: head with highest eigenvalue (component=0) at each selected layer ---
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
# Direction sets (same as M0/06)
# ---------------------------------------------------------------------------

def _load_all_probes(store) -> ResidualProbeSet:
    paths = [p for p in [
        store.time_lagged_residual_directions_path,
        store.residual_pca_directions_path,
        store.random_residual_directions_path,
    ] if Path(p).exists()]
    if not paths:
        raise FileNotFoundError("No probe direction files found; run 02 first.")
    return load_many_probe_sets(paths)


def _validation_timescales(store) -> pd.DataFrame:
    ts = pd.read_parquet(store.residual_probe_timescales_path)
    ts = ts[(ts["split"] == "val") & (ts["control"] == "real")].copy()
    if "tau_valid_within" in ts:
        ts = ts[ts["tau_valid_within"].astype(bool)].copy()
    return ts


def _direction_sets(store, probes: ResidualProbeSet, timescales: pd.DataFrame) -> pd.DataFrame:
    import json
    id_to_idx = {str(pid): idx for idx, pid in enumerate(probes.probe_ids.astype(str))}
    k = 32
    dim_path = os.path.join(store.subspace_dir, "dimensionality_summary.json")
    if os.path.exists(dim_path):
        with open(dim_path) as fh:
            k = int(json.load(fh).get("k_80pct_lifetime_excess", k) or k)
    k = max(1, min(k, 128))
    tl_filtered = timescales.copy()
    if "gk_positive_validation_persistence" in tl_filtered:
        tl_filtered = tl_filtered[
            (tl_filtered["probe_family"] != "time_lagged")
            | tl_filtered["gk_positive_validation_persistence"].astype(bool)
        ].copy()
    persistent = deduplicate_ranked_probes(probes, tl_filtered, threshold=0.95).head(k)
    random = timescales[timescales["probe_family"] == "random"].sort_values("feature_index").head(len(persistent))
    low = timescales[timescales["probe_family"] == "time_lagged"].sort_values("tau_within").head(len(persistent))
    rows = []
    for group, table in [("persistent", persistent), ("random", random), ("low_lifetime_time_lagged", low)]:
        for rank, row in enumerate(table.itertuples(index=False)):
            probe_id = str(getattr(row, "probe_id"))
            idx = id_to_idx.get(probe_id)
            if idx is None:
                continue
            rows.append({
                "direction_group": group, "direction_rank": rank, "probe_id": probe_id,
                "probe_family": str(getattr(row, "probe_family")),
                "tau_within": float(getattr(row, "tau_within", np.nan)),
                "direction_index": int(idx),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-document Pearson sufficient statistics
# ---------------------------------------------------------------------------

@dataclass
class PearsonAccum:
    """Accumulates per-document sufficient statistics for Pearson r.

    Supports document-level bootstrap by keeping stats per document.
    """
    n_docs: int
    n_dirs: int
    # Per-doc stats, shape (n_docs, n_dirs):
    counts: np.ndarray = field(init=False)
    sx:     np.ndarray = field(init=False)
    sy:     np.ndarray = field(init=False)
    sxx:    np.ndarray = field(init=False)
    syy:    np.ndarray = field(init=False)
    sxy:    np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        z = np.zeros((self.n_docs, self.n_dirs), dtype=np.float64)
        self.counts = z.copy(); self.sx = z.copy(); self.sy = z.copy()
        self.sxx = z.copy(); self.syy = z.copy(); self.sxy = z.copy()

    def update(self, doc_start: int, x: np.ndarray, y: np.ndarray) -> None:
        """x, y: (batch, n_valid_pos, n_dirs) float32."""
        b = x.shape[0]
        xd = x.astype(np.float64); yd = y.astype(np.float64)
        self.counts[doc_start:doc_start + b] += xd.shape[1]
        self.sx [doc_start:doc_start + b] += xd.sum(axis=1)
        self.sy [doc_start:doc_start + b] += yd.sum(axis=1)
        self.sxx[doc_start:doc_start + b] += (xd * xd).sum(axis=1)
        self.syy[doc_start:doc_start + b] += (yd * yd).sum(axis=1)
        self.sxy[doc_start:doc_start + b] += (xd * yd).sum(axis=1)

    def pearson_pooled(self, doc_mask: np.ndarray | None = None) -> np.ndarray:
        """Pooled Pearson r over all (or masked) documents, shape (n_dirs,)."""
        mask = doc_mask if doc_mask is not None else np.ones(self.n_docs, dtype=bool)
        n   = self.counts[mask].sum(axis=0).astype(np.float64)
        sx  = self.sx [mask].sum(axis=0)
        sy  = self.sy [mask].sum(axis=0)
        sxx = self.sxx[mask].sum(axis=0)
        syy = self.syy[mask].sum(axis=0)
        sxy = self.sxy[mask].sum(axis=0)
        num = n * sxy - sx * sy
        den = np.sqrt(np.maximum(n * sxx - sx**2, 0.0) * np.maximum(n * syy - sy**2, 0.0))
        return np.where(den > 0, num / den, np.nan)

    def bootstrap_ci(self, seed: int, reps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Document-level bootstrap CI. Returns (ci_low, ci_high), shape (n_dirs,)."""
        rng = np.random.default_rng(seed)
        sample_rs = np.zeros((reps, self.n_dirs), dtype=np.float64)
        for rep in range(reps):
            doc_idx = rng.integers(0, self.n_docs, size=self.n_docs)
            sample_rs[rep] = self._pearson_from_indices(doc_idx)
        return np.quantile(sample_rs, 0.025, axis=0), np.quantile(sample_rs, 0.975, axis=0)

    def _pearson_from_indices(self, doc_idx: np.ndarray) -> np.ndarray:
        # Accumulate sufficient stats for the resampled document set
        n   = self.counts[doc_idx].sum(axis=0).astype(np.float64)
        sx  = self.sx [doc_idx].sum(axis=0)
        sy  = self.sy [doc_idx].sum(axis=0)
        sxx = self.sxx[doc_idx].sum(axis=0)
        syy = self.syy[doc_idx].sum(axis=0)
        sxy = self.sxy[doc_idx].sum(axis=0)
        num = n * sxy - sx * sy
        den = np.sqrt(np.maximum(n * sxx - sx**2, 0.0) * np.maximum(n * syy - sy**2, 0.0))
        return np.where(den > 0, num / den, np.nan)


# ---------------------------------------------------------------------------
# Forward pass + transport computation
# ---------------------------------------------------------------------------

def _validate_transport_hooks(provider: ResidualStreamProvider, layers: list[int]) -> None:
    model = provider.model
    probe = torch.zeros((1, 8), dtype=torch.long, device=provider.device)
    needed = set()
    for layer in layers:
        needed.add(f"blocks.{layer}.hook_resid_pre")
        needed.add(f"blocks.{layer}.attn.hook_pattern")
        needed.add(f"blocks.{layer}.attn.hook_z")
    with torch.inference_mode():
        _, cache = model.run_with_cache(probe, names_filter=lambda n: n in needed, return_type=None)
    missing = [n for n in needed if n not in cache]
    if missing:
        raise RuntimeError(
            f"M3 requires hooks not found in cache: {missing[:3]}{'...' if len(missing)>3 else ''}. "
            "Check TransformerLens version."
        )


def _run_layer_transport(
    provider: ResidualStreamProvider,
    val_tokens: np.ndarray,
    layer: int,
    heads: list[int],
    directions: np.ndarray,       # (n_dirs, d_model) float32
    min_dest_pos: int,
    batch_size: int,
    n_docs: int,
) -> tuple[dict[int, PearsonAccum], dict[int, PearsonAccum]]:
    """Compute transport correlation sufficient stats for all heads in one layer.

    Returns (accums_excl_self, accums_incl_self):
      excl_self: diagonal of attention pattern zeroed and renormalized — tests cross-position routing.
      incl_self: raw attention pattern — can conflate same-token and routing signal.

    Processes one layer at a time (one forward pass per batch) to keep
    attention pattern GPU memory bounded (~1 GB for batch=32, seq=1024).
    """
    model = provider.model
    d_model = int(model.cfg.d_model)
    n_dirs = directions.shape[0]
    hook_resid = f"blocks.{layer}.hook_resid_pre"
    hook_pat   = f"blocks.{layer}.attn.hook_pattern"
    hook_z     = f"blocks.{layer}.attn.hook_z"
    hooks_set  = {hook_resid, hook_pat, hook_z}

    # Pre-fuse: W_O_h @ directions.T → (d_head, n_dirs) — avoids large d_model intermediate per batch
    w_o_proj: dict[int, torch.Tensor] = {}
    for h in heads:
        w_o_h   = model.W_O[layer][h].detach().to(provider.device)    # (d_head, d_model)
        dirs_gpu = torch.as_tensor(directions.T, dtype=w_o_h.dtype, device=provider.device)
        w_o_proj[h] = (w_o_h @ dirs_gpu).detach()            # (d_head, n_dirs)

    dirs_gpu_f32 = torch.as_tensor(directions.T, dtype=torch.float32, device=provider.device)

    accums_excl: dict[int, PearsonAccum] = {h: PearsonAccum(n_docs=n_docs, n_dirs=n_dirs) for h in heads}
    accums_incl: dict[int, PearsonAccum] = {h: PearsonAccum(n_docs=n_docs, n_dirs=n_dirs) for h in heads}

    doc_offset = 0
    for start in range(0, len(val_tokens), batch_size):
        batch = val_tokens[start: start + batch_size]
        b = batch.shape[0]
        batch_tok = torch.as_tensor(batch, dtype=torch.long, device=provider.device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(
                batch_tok, names_filter=lambda n: n in hooks_set, return_type=None
            )

        # Source states: r_pre projected onto each direction → (b, seq, n_dirs)
        r_pre = cache[hook_resid].float()                         # (b, seq, d_model)
        seq_len = r_pre.shape[1]
        src = torch.mm(r_pre.reshape(-1, d_model), dirs_gpu_f32).reshape(b, seq_len, n_dirs)
        pattern_all = cache[hook_pat].float()                     # (b, n_heads, seq, seq)
        z_all = cache[hook_z].float()                             # (b, seq, n_heads, d_head)

        # Diagonal index for self-token masking (shared across heads and batch)
        diag = torch.arange(seq_len, device=provider.device)

        for h in heads:
            pat_h = pattern_all[:, h, :, :]                       # (b, seq_dest, seq_src)

            # --- include-self: raw pattern ---
            transport_incl = torch.bmm(pat_h, src)                # (b, seq, n_dirs)

            # --- exclude-self: zero diagonal, renormalize over prior positions ---
            pat_excl = pat_h.clone()
            pat_excl[:, diag, diag] = 0.0
            # Renormalize; guard against all-zero rows (should not occur at t ≥ min_dest_pos
            # since there are always s < t positions, but guard anyway)
            pat_excl_sum = pat_excl.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            transport_excl = torch.bmm(pat_excl / pat_excl_sum, src)  # (b, seq, n_dirs)

            # Head write: z_h @ (W_O_h @ directions.T) → (b, seq, n_dirs)
            z_h = z_all[:, :, h, :]                                # (b, seq, d_head)
            o_proj = torch.mm(
                z_h.reshape(-1, z_h.shape[-1]),
                w_o_proj[h].to(torch.float32),
            ).reshape(b, seq_len, n_dirs)                          # (b, seq, n_dirs)

            # Mask to t ≥ min_dest_pos and transfer to CPU
            o_valid     = o_proj         [:, min_dest_pos:, :].detach().cpu().numpy()
            t_excl_cpu  = transport_excl [:, min_dest_pos:, :].detach().cpu().numpy()
            t_incl_cpu  = transport_incl [:, min_dest_pos:, :].detach().cpu().numpy()

            accums_excl[h].update(doc_offset, t_excl_cpu, o_valid)
            accums_incl[h].update(doc_offset, t_incl_cpu, o_valid)

        doc_offset += b
        del cache, r_pre, src, pattern_all, z_all

    return accums_excl, accums_incl


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _compute_summary(
    rows: pd.DataFrame,
    head_specs: list[HeadSpec],
    metadata: dict[str, object],
    m0_status: str = "missing",
) -> dict[str, object]:
    """Per-head-group summary with correct reduction hierarchy and M0-gated claim.

    Step 1: reduce to one value per (layer, head) = median over persistent directions.
    Step 2: compare aligned head-level medians to control head-level medians.
    This avoids overweighting heads with more direction rows or correlated directions.

    transport_statistic_positive: aligned > ALL matched controls (not any-positive gate).
    claim_level_7b_supported: additionally requires M0 attention-specific geometry.
    """
    EXCL_COL = "pearson_r_excl_self"   # primary metric
    INCL_COL = "pearson_r_incl_self"   # sensitivity
    ALIGNED  = {"raw_top", "resid_top"}
    CONTROLS = ["low_align", "random_ctrl", "high_var"]

    summary: dict[str, object] = {
        **metadata,
        "primary_metric": "pearson_r_excl_self",
        "sensitivity_metric": "pearson_r_incl_self",
        "bootstrap_method": "document (per-head/per-direction CIs)",
        "comparison_note": (
            "Aligned-vs-control differences are computed on head-level medians "
            "(median over persistent directions per head) to avoid inflating n "
            "from correlated directions."
        ),
        "by_head_group": {},
        "aligned_vs_control": {},
    }

    persistent = rows[rows["direction_group"] == "persistent"]

    # Step 1: head-level medians (one value per head, per split)
    head_medians = (
        persistent
        .groupby(["split", "layer", "head", "head_group"])[[EXCL_COL, INCL_COL]]
        .median()
        .reset_index()
    )

    # Per-group stats (on head-level medians)
    for group in sorted(rows["head_group"].dropna().unique()):
        grp = head_medians[head_medians["head_group"] == group]
        summary["by_head_group"][str(group)] = {
            "head_count": int(len(grp)),
            "median_r_excl_self": float(grp[EXCL_COL].median()) if len(grp) else float("nan"),
            "median_r_incl_self": float(grp[INCL_COL].median()) if len(grp) else float("nan"),
            "q90_r_excl_self":    float(grp[EXCL_COL].quantile(0.9)) if len(grp) else float("nan"),
        }

    # Step 2: aligned vs each control on head-level medians
    aligned_r = head_medians[head_medians["head_group"].isin(ALIGNED)][EXCL_COL].dropna()
    all_controls_positive = True
    for ctrl_grp in CONTROLS:
        ctrl_r = head_medians[head_medians["head_group"] == ctrl_grp][EXCL_COL].dropna()
        if len(aligned_r) and len(ctrl_r):
            diff = float(aligned_r.median() - ctrl_r.median())
            positive = bool(diff > 0)
        else:
            diff = float("nan")
            positive = False
        summary["aligned_vs_control"][ctrl_grp] = {
            "aligned_median_r": float(aligned_r.median()) if len(aligned_r) else float("nan"),
            "control_median_r": float(ctrl_r.median()) if len(ctrl_r) else float("nan"),
            "difference": diff,
            f"positive_vs_{ctrl_grp}": positive,
        }
        if not positive:
            all_controls_positive = False

    # Transport statistic: aligned > ALL matched controls
    transport_statistic_positive = bool(all_controls_positive and len(aligned_r) > 0)
    summary["positive_vs_all_controls"] = transport_statistic_positive
    summary["transport_statistic_positive"] = transport_statistic_positive

    # M0-gated claim: level-7b routing claim requires positive M3 AND M0 attention-specific
    summary["m0_status"] = m0_status
    claim_supported = transport_statistic_positive and m0_status == "attention_specific"
    summary["claim_level_7b_supported"] = claim_supported

    if claim_supported:
        interpretation = (
            "Positive M3 transport statistic with M0 attention-specific geometry: "
            "attention routing claim supported."
        )
    elif transport_statistic_positive and m0_status == "missing":
        interpretation = (
            "M3 transport statistic is positive, but M0 is missing; "
            "do not claim attention-specific routing. Provide --m0-summary to evaluate."
        )
    elif transport_statistic_positive:
        interpretation = (
            f"Transport correlation may reflect generic residual geometry "
            f"(M0 status: {m0_status})."
        )
    else:
        interpretation = "Transport statistic not positive."
    summary["interpretation"] = interpretation

    # Backward-compatible status field (reflects transport statistic, not routing claim)
    summary["status"] = (
        "positive_transport" if transport_statistic_positive
        else "not_positive_or_not_evaluable"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="M3: attention transport test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-top-heads", type=int, default=5,
                        help="Number of top heads per group to select from 06 results.")
    parser.add_argument("--heads", nargs="+", default=None,
                        help="Manual head override, e.g. L12H7:raw_top L11H0:resid_top. "
                             "Bypasses auto-selection when provided.")
    parser.add_argument("--min-dest-pos", type=int, default=64,
                        help="Exclude destination positions < this (early context build-up).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Docs per forward pass. Default 4: attention pattern is ~1 GB at batch=32. "
                             "Increase on GPUs with headroom; watch for OOM.")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test"],
                        help="Document splits to evaluate. Default: val only.")
    parser.add_argument("--m0-summary", default=None,
                        help="Path to M0 summary JSON. Required to gate claim_level_7b_supported; "
                             "without it, m0_status is recorded as 'missing' and the routing "
                             "claim is not asserted even if the transport statistic is positive.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Recompute attention_transport_summary.json from existing parquet without "
                             "rerunning model forwards. Uses head_specs from the existing summary.")
    args = parser.parse_args()

    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    output_path  = os.path.join(store.subspace_dir, "attention_transport.parquet")
    summary_path = os.path.join(store.subspace_dir, "attention_transport_summary.json")
    metadata = artifact_metadata(config, "residual_geometry_M3_attention_transport")

    if args.summary_only:
        if not Path(output_path).exists():
            raise FileNotFoundError(f"Missing existing M3 parquet: {output_path}")
        if not Path(summary_path).exists():
            raise FileNotFoundError(f"Missing existing M3 summary for head_specs: {summary_path}")
        result = pd.read_parquet(output_path)
        head_specs = _head_specs_from_summary(summary_path)
        m0_status, m0_path = _resolve_m0_status(store, args.m0_summary)
        summary = _compute_summary(result, head_specs, metadata, m0_status=m0_status)
        summary["m0_summary_path"] = m0_path
        summary["summary_only_recompute"] = True
        summary["source_residual_hook"] = "hook_resid_pre (pre-attention, pre-LayerNorm)"
        summary["head_specs"] = [
            {"layer": s.layer, "head": s.head, "group": s.head_group, "alignment_score": s.alignment_score}
            for s in head_specs
        ]
        save_json(summary, summary_path)
        return 0

    if Path(output_path).exists() and Path(summary_path).exists() and not args.overwrite:
        return 0

    # Head selection
    alignment_path = os.path.join(store.subspace_dir, "residual_direction_head_alignment.parquet")
    pca_path       = os.path.join(store.subspace_dir, "head_output_pca.parquet")
    if args.heads:
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

    # Upfront prerequisite checks — fail before loading the model
    if not any(Path(p).exists() for p in [
        store.time_lagged_residual_directions_path,
        store.residual_pca_directions_path,
        store.random_residual_directions_path,
    ]):
        raise FileNotFoundError(
            "No probe direction files found. "
            "Run 02_compute_residual_probes.py first."
        )
    if not Path(store.residual_probe_timescales_path).exists():
        raise FileNotFoundError(
            f"Missing probe timescales at {store.residual_probe_timescales_path}. "
            "Run 03_compute_residual_autocorr.py first."
        )

    provider = ResidualStreamProvider(config)
    provider.load()
    n_model_heads = int(provider.model.cfg.n_heads)

    # Validate head indices
    invalid = [(s.layer, s.head) for s in head_specs if s.head >= n_model_heads]
    if invalid:
        raise ValueError(f"Head index out of range for n_heads={n_model_heads}: {invalid}")

    unique_layers = sorted({s.layer for s in head_specs})
    _validate_transport_hooks(provider, unique_layers)

    # Probe directions
    probes = _load_all_probes(store)
    timescales = _validation_timescales(store)
    direction_rows = _direction_sets(store, probes, timescales)
    directions = probes.directions[direction_rows["direction_index"].to_numpy()].astype(np.float32)  # (n_dirs, d_model)
    n_dirs = len(directions)

    batch_size = args.batch_size or residual_cfg.projections.batch_size

    all_rows: list[pd.DataFrame] = []
    for split in args.splits:
        tokens = token_matrix(split_contexts(store, split))
        n_docs = len(tokens)

        # Group head specs by layer and process one layer at a time
        layer_to_heads: dict[int, list[int]] = {}
        for s in head_specs:
            layer_to_heads.setdefault(s.layer, [])
            if s.head not in layer_to_heads[s.layer]:
                layer_to_heads[s.layer].append(s.head)

        # head_spec_map for metadata lookup
        spec_map: dict[tuple[int, int], HeadSpec] = {(s.layer, s.head): s for s in head_specs}

        for layer, heads in layer_to_heads.items():
            accums_excl, accums_incl = _run_layer_transport(
                provider=provider,
                val_tokens=tokens,
                layer=layer,
                heads=heads,
                directions=directions,
                min_dest_pos=args.min_dest_pos,
                batch_size=batch_size,
                n_docs=n_docs,
            )
            for h in heads:
                spec = spec_map.get((layer, h), HeadSpec(layer, h, "manual"))
                seed = layer * 100 + h
                r_excl = accums_excl[h].pearson_pooled()
                r_incl = accums_incl[h].pearson_pooled()
                ci_lo_excl, ci_hi_excl = accums_excl[h].bootstrap_ci(seed=seed, reps=args.bootstrap_reps)
                ci_lo_incl, ci_hi_incl = accums_incl[h].bootstrap_ci(seed=seed + 1, reps=args.bootstrap_reps)
                for dir_i, drow in enumerate(direction_rows.itertuples(index=False)):
                    all_rows.append({
                        "split": split,
                        "layer": int(layer),
                        "head": int(h),
                        "head_group": spec.head_group,
                        "head_alignment_score": float(spec.alignment_score),
                        "direction_group": str(drow.direction_group),
                        "direction_rank": int(drow.direction_rank),
                        "probe_id": str(drow.probe_id),
                        "probe_family": str(drow.probe_family),
                        "tau_within": float(drow.tau_within),
                        "pearson_r_excl_self": float(r_excl[dir_i]),
                        "pearson_r_incl_self": float(r_incl[dir_i]),
                        "bootstrap_ci_low_excl":  float(ci_lo_excl[dir_i]),
                        "bootstrap_ci_high_excl": float(ci_hi_excl[dir_i]),
                        "bootstrap_ci_low_incl":  float(ci_lo_incl[dir_i]),
                        "bootstrap_ci_high_incl": float(ci_hi_incl[dir_i]),
                        "n_docs": int(n_docs),
                        "min_dest_pos": int(args.min_dest_pos),
                    })

    result = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    for key, value in metadata.items():
        result[key] = value
    save_parquet(result, output_path)

    # Load M0 status for claim gating. Defaults to the M0 artifact in this run if present.
    m0_status, m0_path = _resolve_m0_status(store, args.m0_summary)

    summary = _compute_summary(result, head_specs, metadata, m0_status=m0_status)
    summary["m0_summary_path"] = m0_path
    summary["source_residual_hook"] = "hook_resid_pre (pre-attention, pre-LayerNorm)"
    summary["head_specs"] = [
        {"layer": s.layer, "head": s.head, "group": s.head_group, "alignment_score": s.alignment_score}
        for s in head_specs
    ]
    save_json(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
