#!/usr/bin/env python
"""M0: Block-output subspace comparison.

For each layer ℓ, fits PCA subspaces on train-document activations:
  U_attn,ℓ  = top PCs of total attention block output (hook_attn_out)
  U_mlp,ℓ   = top PCs of MLP output (hook_mlp_out)
  U_resid,ℓ = top PCs of residual stream (hook_resid_post)

For each probe direction v, computes:
  B_X,ℓ(v) = ||U_X,ℓᵀ v||₂   (subspace projection norm, ∈ [0, 1] for unit v)
  E_X,ℓ(v) = B_X,ℓ(v) − B_resid,ℓ(v)   (excess over residual geometry baseline)

Primary output: ΔE_attn,ℓ = median_persistent(E_attn) − median_random(E_attn),
and the same for MLP, with bootstrap CIs over directions.

Interpretation (per spec §14.3):
  attention > MLP beyond residual PCA  →  attention-specific geometry
  MLP > attention                      →  MLP-written state
  both high                            →  both mechanisms involved
  neither survives residual-PCA control→  mechanism not localized by tested subspaces
"""
from __future__ import annotations

import argparse
import json
import os
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
from residual_geometry.residuals.probes import ResidualProbeSet, load_many_probe_sets, load_probe_set
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.subspace.residual_geometry import deduplicate_ranked_probes
from residual_geometry.utils.io import save_json, save_parquet


HOOK_TYPES = ("attn", "mlp", "resid")


def _hook_names_for_layer(layer: int) -> dict[str, str]:
    return {
        "attn": f"blocks.{layer}.hook_attn_out",
        "mlp": f"blocks.{layer}.hook_mlp_out",
        "resid": f"blocks.{layer}.hook_resid_post",
    }


# ---------------------------------------------------------------------------
# Shared helpers (mirrors 06_attention_alignment.py)
# ---------------------------------------------------------------------------

def _selected_global_positions(total_positions: int, max_positions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(max_positions, total_positions)
    return np.sort(rng.choice(total_positions, size=count, replace=False))


def _top_k_eigh(cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvalues (descending) and eigenvectors via scipy partial eigensolver."""
    n = cov.shape[0]
    if k >= n:
        vals, vecs = np.linalg.eigh(cov)
        return vals[::-1].copy(), vecs[:, ::-1].copy()
    try:
        from scipy.linalg import eigh as _scipy_eigh
        vals, vecs = _scipy_eigh(cov, subset_by_index=[n - k, n - 1])
        return vals[::-1].copy(), vecs[:, ::-1].copy()
    except ImportError:
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        return vals[order[:k]], vecs[:, order[:k]]


def _load_all_probes(store) -> ResidualProbeSet:
    paths = [
        p for p in [
            store.time_lagged_residual_directions_path,
            store.residual_pca_directions_path,
            store.random_residual_directions_path,
        ] if Path(p).exists()
    ]
    if not paths:
        raise FileNotFoundError("No probe direction files found; run 02 first.")
    return load_many_probe_sets(paths)


def _validation_timescales(store) -> pd.DataFrame:
    if not Path(store.residual_probe_timescales_path).exists():
        raise FileNotFoundError("Missing timescales; run 03 first.")
    ts = pd.read_parquet(store.residual_probe_timescales_path)
    ts = ts[(ts["split"] == "val") & (ts["control"] == "real")].copy()
    if "tau_valid_within" in ts:
        ts = ts[ts["tau_valid_within"].astype(bool)].copy()
    return ts


def _direction_sets(store, probes: ResidualProbeSet, timescales: pd.DataFrame) -> pd.DataFrame:
    """Build persistent / random / low-lifetime direction groups (same as 06)."""
    id_to_idx = {str(pid): idx for idx, pid in enumerate(probes.probe_ids.astype(str))}
    k = 32
    dim_path = os.path.join(store.subspace_dir, "dimensionality_summary.json")
    if os.path.exists(dim_path):
        with open(dim_path, "r", encoding="utf-8") as fh:
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
    low = (
        timescales[timescales["probe_family"] == "time_lagged"]
        .sort_values("tau_within", ascending=True)
        .head(len(persistent))
    )
    rows: list[dict[str, object]] = []
    for group, table in [("persistent", persistent), ("random", random), ("low_lifetime_time_lagged", low)]:
        for rank, row in enumerate(table.itertuples(index=False)):
            probe_id = str(getattr(row, "probe_id"))
            idx = id_to_idx.get(probe_id)
            if idx is None:
                continue
            rows.append({
                "direction_group": group,
                "direction_rank": rank,
                "probe_id": probe_id,
                "probe_family": str(getattr(row, "probe_family")),
                "tau_within": float(getattr(row, "tau_within", np.nan)),
                "direction_index": int(idx),
            })
    return pd.DataFrame(rows)


def _bootstrap_median_diff(
    a: np.ndarray, b: np.ndarray, seed: int, reps: int = 1000
) -> dict[str, float]:
    """Bootstrap CI over whatever unit is passed in (directions or documents).
    Caller is responsible for passing the right unit and labeling the result.
    """
    rng = np.random.default_rng(seed)
    diffs = np.array([
        np.median(rng.choice(a, len(a), replace=True)) - np.median(rng.choice(b, len(b), replace=True))
        for _ in range(reps)
    ])
    return {
        "median_difference": float(np.median(a) - np.median(b)),
        "bootstrap_ci_low": float(np.quantile(diffs, 0.025)),
        "bootstrap_ci_high": float(np.quantile(diffs, 0.975)),
        "bootstrap_method": "direction",  # bootstraps over probe directions, not documents
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


# ---------------------------------------------------------------------------
# M0-specific: subspace PCA fitting
# ---------------------------------------------------------------------------

def _validate_hooks(provider: ResidualStreamProvider, layers: list[int]) -> None:
    """Fail fast if any requested hook is missing for this model/TL version.

    Runs a single tiny forward pass and checks that all 3 × n_layers hook names
    are populated. Raises RuntimeError listing every missing hook before any
    expensive computation starts.
    """
    model = provider.model
    probe = torch.zeros((1, 4), dtype=torch.long, device=provider.device)
    all_hooks: dict[str, tuple[str, int]] = {}
    for layer in layers:
        for ht, name in _hook_names_for_layer(layer).items():
            all_hooks[name] = (ht, layer)
    names_set = set(all_hooks)
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            probe, names_filter=lambda name: name in names_set, return_type=None
        )
    missing = [name for name in all_hooks if name not in cache]
    if missing:
        raise RuntimeError(
            f"M0 requires {len(all_hooks)} hooks but {len(missing)} are missing from the cache. "
            f"First missing: {missing[0]}. Check TransformerLens version and model hook names."
        )


def _sample_block_outputs(
    provider: ResidualStreamProvider,
    tokens: torch.Tensor,
    layers: list[int],
    local_indices: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    """One forward pass → GPU-side sample selected positions → CPU transfer.

    Returns {(hook_type, layer): (n_selected, d_model) float32}.
    All three hooks (attn_out, mlp_out, resid_post) for all requested layers
    are cached in a single run_with_cache call. Call _validate_hooks before
    the batch loop so missing hooks raise immediately rather than silently
    producing NaNs.
    """
    model = provider.model
    d_model = int(model.cfg.d_model)
    name_to_key: dict[str, tuple[str, int]] = {}
    for layer in layers:
        for ht, name in _hook_names_for_layer(layer).items():
            name_to_key[name] = (ht, layer)
    names_set = set(name_to_key)
    idx = torch.as_tensor(local_indices, dtype=torch.long, device=provider.device)
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name in names_set, return_type=None
        )
    return {
        key: cache[name].reshape(-1, d_model)[idx].float().cpu().numpy()
        for name, key in name_to_key.items()
        if name in cache  # hooks validated upfront; missing here would be a runtime anomaly
    }


def _fit_subspace_pca(
    provider: ResidualStreamProvider,
    train_tokens: np.ndarray,
    layers: list[int],
    selected_positions: np.ndarray,
    batch_size: int,
    components: int,
) -> tuple[dict[tuple[str, int], np.ndarray], pd.DataFrame]:
    """Fit top-k PCA on attn_out / mlp_out / resid_post for each layer.

    Returns:
        pca_dirs: {(hook_type, layer): (d_model, k) float32 orthonormal basis}
        diagnostics: DataFrame of per-component eigenvalues
    """
    _validate_hooks(provider, layers)   # fail fast before expensive batch loop
    d_model = int(provider.model.cfg.d_model)
    keys = [(ht, layer) for ht in HOOK_TYPES for layer in layers]
    sums  = {key: np.zeros(d_model, dtype=np.float64) for key in keys}
    cross = {key: np.zeros((d_model, d_model), dtype=np.float64) for key in keys}
    counts = {key: 0 for key in keys}

    global_offset = 0
    sel_offset = 0
    for start in range(0, len(train_tokens), batch_size):
        batch = train_tokens[start : start + batch_size]
        batch_positions = batch.shape[0] * batch.shape[1]
        batch_start = global_offset
        batch_end   = global_offset + batch_positions
        lo = sel_offset
        while lo < len(selected_positions) and selected_positions[lo] < batch_start:
            lo += 1
        hi = lo
        while hi < len(selected_positions) and selected_positions[hi] < batch_end:
            hi += 1
        if hi > lo:
            batch_tokens = torch.as_tensor(batch, dtype=torch.long, device=provider.device)
            local = selected_positions[lo:hi] - batch_start
            samples = _sample_block_outputs(provider, batch_tokens, layers, local)
            for key, sample in samples.items():
                s = sample.astype(np.float64, copy=False)  # (n_selected, d_model)
                sums[key]  += s.sum(axis=0)
                cross[key] += s.T @ s              # BLAS DGEMM — not einsum
                counts[key] += len(s)
        global_offset = batch_end
        sel_offset = hi

    pca_dirs: dict[tuple[str, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for ht in HOOK_TYPES:
        for layer in layers:
            key = (ht, layer)
            count = counts[key]
            if count <= components:
                continue
            mean = sums[key] / count
            cov  = cross[key] / count - np.outer(mean, mean)
            cov  = (cov + cov.T) * 0.5
            total = float(np.trace(cov))
            vals, vecs = _top_k_eigh(cov, components)
            vals = np.maximum(vals[:components], 0.0)
            pca_dirs[key] = vecs[:, :components].astype(np.float32)  # (d_model, k)
            for comp_i, eig in enumerate(vals):
                rows.append({
                    "hook_type": ht,
                    "layer": int(layer),
                    "component": int(comp_i),
                    "eigenvalue": float(eig),
                    "explained_variance_ratio": float(eig / max(total, 1e-12)),
                    "sample_count": int(count),
                })
    return pca_dirs, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def _build_control_bases(
    residual_pca_directions: np.ndarray,
    d_model: int,
    k_values: list[int],
) -> dict[int, np.ndarray]:
    """Orthonormal bases for each global residual-PCA control level."""
    bases: dict[int, np.ndarray] = {0: np.zeros((d_model, 0), dtype=np.float64)}
    for k in sorted(k_values):
        if k <= 0:
            continue
        n = min(k, len(residual_pca_directions), d_model)
        q, _ = np.linalg.qr(residual_pca_directions[:n].T.astype(np.float64))
        bases[k] = q[:, :n]
    return bases


def _overlap_table(
    pca_dirs: dict[tuple[str, int], np.ndarray],
    layers: list[int],
    direction_rows: pd.DataFrame,
    probes: ResidualProbeSet,
    residual_pca_directions: np.ndarray,
    resid_pca_control_k: list[int],
) -> pd.DataFrame:
    """Compute B_attn, B_mlp, B_resid = ||U^T v||₂ and excess E_X = B_X − B_resid.

    resid_pca_control_k=0 is always included (raw, no global PCA projected out).
    Higher values project out that many global residual PCA directions from v first.
    """
    d_model = probes.d_model
    ctrl_k_vals = [0, *sorted(int(k) for k in resid_pca_control_k if int(k) > 0)]
    control_bases = _build_control_bases(residual_pca_directions, d_model, ctrl_k_vals)

    rows: list[dict[str, object]] = []
    for drow in direction_rows.itertuples(index=False):
        raw = probes.directions[int(drow.direction_index)].astype(np.float64)
        raw_unit = raw / max(float(np.linalg.norm(raw)), 1e-12)

        for ctrl_k in ctrl_k_vals:
            basis_ctrl = control_bases[ctrl_k]
            if basis_ctrl.shape[1]:
                coords = raw_unit @ basis_ctrl
                direction = raw_unit - coords @ basis_ctrl.T
                residualized_norm = float(np.linalg.norm(direction))
                valid = residualized_norm >= 1e-3
                fraction_removed = float(np.sum(coords ** 2))
                direction = direction / residualized_norm if valid else direction
            else:
                direction = raw_unit
                residualized_norm = 1.0
                valid = True
                fraction_removed = 0.0

            for layer in layers:
                row: dict[str, object] = {
                    "direction_group": str(drow.direction_group),
                    "direction_rank": int(drow.direction_rank),
                    "probe_id": str(drow.probe_id),
                    "probe_family": str(drow.probe_family),
                    "tau_within": float(drow.tau_within),
                    "layer": int(layer),
                    "resid_pca_control_k": int(ctrl_k),
                    "fraction_removed_by_resid_pca": fraction_removed,
                    "residualized_direction_norm": residualized_norm,
                    "alignment_valid": valid,
                }
                for ht in HOOK_TYPES:
                    key = (ht, layer)
                    if key not in pca_dirs or not valid:
                        row[f"B_{ht}"] = float("nan")
                    else:
                        U = pca_dirs[key].astype(np.float64)      # (d_model, k)
                        row[f"B_{ht}"] = float(np.linalg.norm(U.T @ direction))
                # Excess over residual geometry baseline
                for ht in ("attn", "mlp"):
                    b_ht    = row.get(f"B_{ht}", float("nan"))
                    b_resid = row.get("B_resid",  float("nan"))
                    row[f"E_{ht}"] = (
                        float(b_ht) - float(b_resid)
                        if not (np.isnan(b_ht) or np.isnan(b_resid))
                        else float("nan")
                    )
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _delta_e_per_layer(
    overlap: pd.DataFrame,
    e_col: str,
    layer: int,
    ctrl_k: int,
    seed: int,
) -> dict[str, object]:
    """ΔE for a single layer, bootstrapping over directions (one row per direction)."""
    sub = overlap[
        (overlap["layer"] == layer)
        & (overlap["resid_pca_control_k"] == ctrl_k)
        & overlap["alignment_valid"].astype(bool)
    ]
    p = sub[sub["direction_group"] == "persistent"][e_col].dropna().to_numpy(np.float64)
    r = sub[sub["direction_group"] == "random"][e_col].dropna().to_numpy(np.float64)
    if not len(p) or not len(r):
        return {"median_difference": float("nan"), "bootstrap_ci_low": float("nan"),
                "bootstrap_ci_high": float("nan"), "bootstrap_method": "direction",
                "n_a": 0, "n_b": 0, "positive": False}
    result = _bootstrap_median_diff(p, r, seed=seed)
    result["positive"] = bool(
        result["median_difference"] > 0.01
        and not np.isnan(result["bootstrap_ci_low"])
        and result["bootstrap_ci_low"] > 0
    )
    return result


def _delta_e_across_layers(
    overlap: pd.DataFrame,
    e_col: str,
    ctrl_k: int,
    seed: int,
) -> dict[str, object]:
    """ΔE pooled across layers, reducing to one value per probe_id first.

    Each direction appears once per layer, so the rows are not independent.
    Reduce to max-over-layers per probe_id before bootstrapping directions.
    """
    sub = overlap[
        (overlap["resid_pca_control_k"] == ctrl_k)
        & overlap["alignment_valid"].astype(bool)
    ]
    # One value per (direction_group, probe_id): max E over layers
    per_probe = (
        sub.groupby(["direction_group", "probe_id"], as_index=False)[e_col]
        .max()
    )
    p = per_probe[per_probe["direction_group"] == "persistent"][e_col].dropna().to_numpy(np.float64)
    r = per_probe[per_probe["direction_group"] == "random"][e_col].dropna().to_numpy(np.float64)
    if not len(p) or not len(r):
        return {"median_difference": float("nan"), "bootstrap_ci_low": float("nan"),
                "bootstrap_ci_high": float("nan"), "bootstrap_method": "direction_max_over_layers",
                "n_a": 0, "n_b": 0, "positive": False}
    result = _bootstrap_median_diff(p, r, seed=seed)
    result["bootstrap_method"] = "direction_max_over_layers"
    result["positive"] = bool(
        result["median_difference"] > 0.01
        and not np.isnan(result["bootstrap_ci_low"])
        and result["bootstrap_ci_low"] > 0
    )
    return result


def _compute_summary(
    overlap: pd.DataFrame,
    layers: list[int],
    components: int,
    resid_pca_control_k: list[int],
    metadata: dict[str, object],
) -> dict[str, object]:
    ctrl_k_vals = [0, *sorted(int(k) for k in resid_pca_control_k if int(k) > 0)]
    summary: dict[str, object] = {
        **metadata,
        "layers": [int(l) for l in layers],
        "components_per_subspace": int(components),
        "resid_pca_control_k_values": ctrl_k_vals,
        "bootstrap_note": (
            "CIs are direction-bootstrap (over probe directions), not document-bootstrap. "
            "across_layers entries reduce to max-over-layers per probe_id before bootstrapping "
            "to avoid treating the same direction at multiple layers as independent samples."
        ),
        "by_layer": {},
        "across_layers": {},
    }

    # Per-layer: raw group stats + ΔE at each ctrl_k level
    for layer in layers:
        layer_sub = overlap[
            (overlap["layer"] == layer)
            & (overlap["resid_pca_control_k"] == 0)
            & overlap["alignment_valid"].astype(bool)
        ]
        layer_entry: dict[str, object] = {}
        for ht in HOOK_TYPES:
            b_col = f"B_{ht}"
            layer_entry[f"B_{ht}"] = {
                group: {
                    "count": int(len(vals := layer_sub[layer_sub["direction_group"] == group][b_col].dropna())),
                    "median": float(vals.median()) if len(vals) else float("nan"),
                    "q90": float(vals.quantile(0.9)) if len(vals) else float("nan"),
                }
                for group in ("persistent", "random", "low_lifetime_time_lagged")
            }
        for ht in ("attn", "mlp"):
            layer_entry[f"delta_E_{ht}_by_ctrl_k"] = {
                str(k): _delta_e_per_layer(overlap, f"E_{ht}", layer=layer, ctrl_k=k, seed=42 + layer + k)
                for k in ctrl_k_vals
            }
        summary["by_layer"][str(layer)] = layer_entry

    # Across layers: reduce to max-over-layers per probe_id before bootstrapping
    for ctrl_k in ctrl_k_vals:
        ctrl_entry: dict[str, object] = {}
        for ht in ("attn", "mlp"):
            ctrl_entry[f"delta_E_{ht}"] = _delta_e_across_layers(
                overlap, f"E_{ht}", ctrl_k=ctrl_k, seed=17 + ctrl_k
            )
        summary["across_layers"][str(ctrl_k)] = ctrl_entry

    # Top-level status: requires survival at raw AND at least one controlled level.
    # A raw positive that collapses under all residual-PCA controls is "raw_only".
    def _is_positive(ctrl_k_str: str, ht: str) -> bool:
        return bool(
            summary["across_layers"].get(ctrl_k_str, {}).get(f"delta_E_{ht}", {}).get("positive", False)
        )

    raw_attn = _is_positive("0", "attn")
    raw_mlp  = _is_positive("0", "mlp")
    ctrl_attn = any(_is_positive(str(k), "attn") for k in ctrl_k_vals if k > 0)
    ctrl_mlp  = any(_is_positive(str(k), "mlp")  for k in ctrl_k_vals if k > 0)

    attn_survives = raw_attn and ctrl_attn
    mlp_survives  = raw_mlp  and ctrl_mlp

    if attn_survives and not mlp_survives:
        status = "attention_specific"
    elif mlp_survives and not attn_survives:
        status = "mlp_specific"
    elif attn_survives and mlp_survives:
        status = "both_attn_and_mlp"
    elif raw_attn or raw_mlp:
        # Raw positive but did not survive any residual-PCA control level
        status = "raw_only_positive_not_attention_mlp_specific"
    else:
        status = "not_positive_or_not_evaluable"

    summary["status"] = status
    summary["uninformative"] = not (attn_survives or mlp_survives)
    summary["raw_positive_attn"] = bool(raw_attn)
    summary["raw_positive_mlp"]  = bool(raw_mlp)
    summary["controlled_positive_attn"] = bool(ctrl_attn)
    summary["controlled_positive_mlp"]  = bool(ctrl_mlp)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="M0: block-output subspace comparison.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[7, 8, 9, 10, 11, 12],
        help="Layers to analyse (default: pilot scope [7..12]).",
    )
    parser.add_argument("--components", type=int, default=32, help="PCA components per subspace.")
    parser.add_argument("--max-positions", type=int, default=50_000)
    parser.add_argument(
        "--resid-pca-control-k", nargs="+", type=int, default=[16, 32, 64, 128],
        help="Global residual PCA projection levels for control analysis.",
    )
    parser.add_argument(
        "--pca-batch-size", type=int, default=None,
        help=(
            "Docs per forward pass for subspace PCA fitting. "
            "Defaults to max(1, 128 // (n_layers × 3)) — conservative given that "
            "3 × n_layers block-output hooks are cached simultaneously. "
            "Increase if GPU is underutilised; decrease if OOM."
        ),
    )
    args = parser.parse_args()

    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    output_path    = os.path.join(store.subspace_dir, "block_output_subspace_overlap.parquet")
    summary_path   = os.path.join(store.subspace_dir, "block_output_subspace_summary.json")
    directions_path = os.path.join(store.subspace_dir, "block_output_pca_directions.npz")
    if Path(output_path).exists() and Path(summary_path).exists() and not args.overwrite:
        return 0

    metadata = artifact_metadata(config, "residual_geometry_M0_block_output_subspace")
    provider = ResidualStreamProvider(config)
    provider.load()
    n_layers_total = int(getattr(provider.model.cfg, "n_layers"))
    layers = sorted(set(l for l in args.layers if 0 <= l < n_layers_total))
    if not layers:
        raise ValueError(f"No valid layers in {args.layers} for n_layers={n_layers_total}")

    train_tokens = token_matrix(split_contexts(store, "train"))
    total_positions = int(train_tokens.shape[0] * train_tokens.shape[1])
    selected = _selected_global_positions(
        total_positions, args.max_positions, seed=residual_cfg.time_lagged.seed + 42
    )
    # 3 hooks × n_layers cached simultaneously; default is conservative for 24 GB GPU
    pca_batch_size = args.pca_batch_size if args.pca_batch_size is not None else max(1, 128 // max(len(layers) * len(HOOK_TYPES), 1))

    pca_dirs, pca_diagnostics = _fit_subspace_pca(
        provider=provider,
        train_tokens=train_tokens,
        layers=layers,
        selected_positions=selected,
        batch_size=pca_batch_size,
        components=args.components,
    )
    save_parquet(pca_diagnostics, os.path.join(store.subspace_dir, "block_output_pca_diagnostics.parquet"))

    # Persist PCA bases for downstream use (M3, reporting)
    npz_payload: dict[str, np.ndarray] = {
        "layers": np.asarray(layers, dtype=np.int64),
        "components": np.asarray(args.components, dtype=np.int64),
    }
    for (ht, layer), basis in pca_dirs.items():
        npz_payload[f"{ht}_layer{layer:02d}"] = basis.astype(np.float32)
    np.savez_compressed(directions_path, **npz_payload)

    probes = _load_all_probes(store)
    residual_pca = load_probe_set(store.residual_pca_directions_path)
    timescales = _validation_timescales(store)
    direction_rows = _direction_sets(store, probes, timescales)

    overlap = _overlap_table(
        pca_dirs=pca_dirs,
        layers=layers,
        direction_rows=direction_rows,
        probes=probes,
        residual_pca_directions=residual_pca.directions,
        resid_pca_control_k=args.resid_pca_control_k,
    )
    for key, value in metadata.items():
        overlap[key] = value
    save_parquet(overlap, output_path)

    summary = _compute_summary(
        overlap=overlap,
        layers=layers,
        components=args.components,
        resid_pca_control_k=args.resid_pca_control_k,
        metadata=metadata,
    )
    save_json(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
