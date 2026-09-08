#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import artifact_metadata, load_config_and_store, require_residual_geometry_config, split_contexts, token_matrix
from residual_geometry.residuals.probes import ResidualProbeSet, load_many_probe_sets, load_probe_set
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.subspace.residual_geometry import deduplicate_ranked_probes
from residual_geometry.utils.io import save_json, save_parquet


def _parse_layers(spec: str, target_layer: int, n_layers: int) -> list[int]:
    if spec == "target":
        return [target_layer]
    if spec == "upstream":
        return list(range(target_layer + 1))
    if spec == "all":
        return list(range(n_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            layers.extend(range(lo, hi + 1))
        else:
            layers.append(int(part))
    out = sorted(set(layers))
    if not out or any(layer < 0 or layer >= n_layers for layer in out):
        raise ValueError(f"Invalid layer spec {spec!r} for n_layers={n_layers}")
    return out


def _load_all_probes(store) -> ResidualProbeSet:
    paths = [
        path
        for path in [
            store.time_lagged_residual_directions_path,
            store.residual_pca_directions_path,
            store.random_residual_directions_path,
        ]
        if Path(path).exists()
    ]
    if not paths:
        raise FileNotFoundError("No residual probe direction files found; run 02_compute_residual_probes.py first.")
    return load_many_probe_sets(paths)


def _validation_timescales(store) -> pd.DataFrame:
    if not Path(store.residual_probe_timescales_path).exists():
        raise FileNotFoundError("Missing residual-probe timescales; run 03_compute_residual_autocorr.py first.")
    timescales = pd.read_parquet(store.residual_probe_timescales_path)
    timescales = timescales[(timescales["split"] == "val") & (timescales["control"] == "real")].copy()
    if "tau_valid_within" in timescales:
        timescales = timescales[timescales["tau_valid_within"].astype(bool)].copy()
    return timescales


def _direction_sets(store, probes: ResidualProbeSet, timescales: pd.DataFrame) -> pd.DataFrame:
    id_to_idx = {str(pid): idx for idx, pid in enumerate(probes.probe_ids.astype(str))}
    dim_path = os.path.join(store.subspace_dir, "dimensionality_summary.json")
    k = 32
    if os.path.exists(dim_path):
        import json

        with open(dim_path, "r", encoding="utf-8") as handle:
            k = int(json.load(handle).get("k_80pct_lifetime_excess", k) or k)
    k = max(1, min(k, 128))
    tl_filtered = timescales.copy()
    if "gk_positive_validation_persistence" in tl_filtered:
        tl_filtered = tl_filtered[
            (tl_filtered["probe_family"] != "time_lagged") | tl_filtered["gk_positive_validation_persistence"].astype(bool)
        ].copy()
    persistent = deduplicate_ranked_probes(probes, tl_filtered, threshold=0.95).head(k)
    random = timescales[timescales["probe_family"] == "random"].sort_values("feature_index").head(len(persistent))
    low = (
        timescales[timescales["probe_family"] == "time_lagged"]
        .sort_values("tau_within", ascending=True)
        .head(len(persistent))
    )
    rows: list[dict[str, object]] = []
    for group_name, table in [("persistent", persistent), ("random", random), ("low_lifetime_time_lagged", low)]:
        for rank, row in enumerate(table.itertuples(index=False)):
            probe_id = str(getattr(row, "probe_id"))
            idx = id_to_idx.get(probe_id)
            if idx is None:
                continue
            rows.append(
                {
                    "direction_group": group_name,
                    "direction_rank": rank,
                    "probe_id": probe_id,
                    "probe_family": str(getattr(row, "probe_family")),
                    "tau_within": float(getattr(row, "tau_within", np.nan)),
                    "direction_index": int(idx),
                }
            )
    return pd.DataFrame(rows)


def _selected_global_positions(total_positions: int, max_positions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(max_positions, total_positions)
    return np.sort(rng.choice(total_positions, size=count, replace=False))


def _top_k_eigh(cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k eigenvalues (descending) and eigenvectors of a symmetric matrix.

    Uses scipy's partial eigensolver (O(n²k)) when available, ~n/k times faster
    than numpy's full eigh (O(n³)). Falls back to numpy otherwise.
    """
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


def _probe_hook_result_available(provider: ResidualStreamProvider, layers: list[int]) -> bool:
    """Run one tiny forward pass to check whether hook_result is populated for this model.

    Gemma-2-2b uses GroupedQueryAttention which may not support hook_result even after
    set_use_attn_result(True). Probing once avoids a wasted full-batch forward pass on
    every batch when the fallback path is needed.
    """
    model = provider.model
    probe = torch.zeros((1, 4), dtype=torch.long, device=provider.device)
    hook_name = f"blocks.{layers[0]}.attn.hook_result"
    try:
        if hasattr(model, "set_use_attn_result"):
            model.set_use_attn_result(True)
        with torch.inference_mode():
            _, cache = model.run_with_cache(probe, names_filter=hook_name, return_type=None)
        return hook_name in cache
    except Exception:
        return False


def _attention_results_for_layers(
    provider: ResidualStreamProvider,
    tokens: torch.Tensor,
    layers: list[int],
    local_indices: np.ndarray,
    use_hook_result: bool,
) -> list[np.ndarray]:
    """Run ONE forward pass, sample selected positions ON GPU, then transfer to CPU.

    Indexing before transfer reduces GPU→CPU data movement from O(batch×seq×n_heads×d_model)
    to O(n_selected×n_heads×d_model) per layer — ~82× smaller for typical selection rates.
    For the hook_z path this also shrinks the W_O matmul by the same factor.

    Returns a list of (n_selected, n_heads, d_model) float32 arrays, one per layer.
    """
    model = provider.model
    n_heads = int(model.cfg.n_heads)
    d_model = int(model.cfg.d_model)
    idx = torch.as_tensor(local_indices, dtype=torch.long, device=provider.device)
    with torch.inference_mode():
        if use_hook_result:
            names = [f"blocks.{layer}.attn.hook_result" for layer in layers]
            names_set = set(names)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name in names_set, return_type=None
            )
            return [
                cache[name].reshape(-1, n_heads, d_model)[idx].float().cpu().numpy()
                for name in names
            ]
        else:
            names = [f"blocks.{layer}.attn.hook_z" for layer in layers]
            names_set = set(names)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name in names_set, return_type=None
            )
            results = []
            for layer, name in zip(layers, names):
                z = cache[name]
                # Sample z BEFORE the W_O matmul: reduces matmul by n_selected/batch*seq
                z_sel = z.reshape(-1, z.shape[2], z.shape[3])[idx]  # (n_selected, n_heads, d_head)
                w_o = model.W_O[layer].to(z.device)
                r = torch.einsum("nhd,hdm->nhm", z_sel, w_o)         # (n_selected, n_heads, d_model)
                results.append(r.float().cpu().numpy())
            return results


def _fit_head_output_pca(
    provider: ResidualStreamProvider,
    train_tokens: np.ndarray,
    layers: list[int],
    selected_positions: np.ndarray,
    batch_size: int,
    components: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    cfg = provider.model.cfg
    n_heads = int(getattr(cfg, "n_heads"))
    d_model = int(getattr(cfg, "d_model"))

    # Probe once to decide whether hook_result or hook_z+W_O path is needed.
    # Avoids a wasted full-batch forward pass on every iteration when the fallback is required.
    use_hook_result = _probe_hook_result_available(provider, layers)

    sums = {layer: np.zeros((n_heads, d_model), dtype=np.float64) for layer in layers}
    cross = {layer: np.zeros((n_heads, d_model, d_model), dtype=np.float64) for layer in layers}
    counts = {layer: 0 for layer in layers}

    global_offset = 0
    sel_offset = 0
    for start in range(0, len(train_tokens), batch_size):
        batch = train_tokens[start : start + batch_size]
        batch_positions = batch.shape[0] * batch.shape[1]
        batch_start = global_offset
        batch_end = global_offset + batch_positions
        lo = sel_offset
        while lo < len(selected_positions) and selected_positions[lo] < batch_start:
            lo += 1
        hi = lo
        while hi < len(selected_positions) and selected_positions[hi] < batch_end:
            hi += 1
        if hi > lo:
            batch_tokens = torch.as_tensor(batch, dtype=torch.long, device=provider.device)
            local = selected_positions[lo:hi] - batch_start
            # Sampling happens on GPU inside _attention_results_for_layers.
            # batch_results[i] is already (n_selected, n_heads, d_model) — no CPU indexing needed.
            batch_results = _attention_results_for_layers(provider, batch_tokens, layers, local, use_hook_result)
            for layer_i, layer in enumerate(layers):
                sample = batch_results[layer_i].astype(np.float64, copy=False)
                sums[layer] += sample.sum(axis=0)
                # Explicit batched matmul always dispatches to BLAS DGEMM.
                # np.einsum("nhd,nhe->hde", ...) with optimize=True does NOT guarantee
                # BLAS dispatch for 3-index contractions and can fall back to element-wise
                # (~1 GFLOPS vs ~200 GFLOPS), turning this step into the dominant cost.
                S = sample.transpose(1, 2, 0)  # (n_heads, d_model, n_selected)
                cross[layer] += S @ S.transpose(0, 2, 1)  # (n_heads, d_model, d_model)
                counts[layer] += int(sample.shape[0])
        global_offset = batch_end
        sel_offset = hi

    all_directions: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for layer in layers:
        count = counts[layer]
        if count <= components:
            raise ValueError(f"Only sampled {count} positions for layer {layer}; need more than components={components}")
        mean = sums[layer] / count
        layer_dirs = np.zeros((n_heads, components, d_model), dtype=np.float32)
        for head in range(n_heads):
            cov = cross[layer][head] / count - np.outer(mean[head], mean[head])
            cov = (cov + cov.T) * 0.5
            total = float(np.trace(cov))
            vals, vecs = _top_k_eigh(cov, components)
            vals = np.maximum(vals[:components], 0.0)
            comps = vecs[:, :components].T.astype(np.float32)
            comps /= np.maximum(np.linalg.norm(comps, axis=1, keepdims=True), 1e-12)
            layer_dirs[head] = comps
            for comp_i, eig in enumerate(vals):
                rows.append(
                    {
                        "layer": int(layer),
                        "head": int(head),
                        "component": int(comp_i),
                        "eigenvalue": float(eig),
                        "explained_variance_ratio": float(eig / max(total, 1e-12)),
                        "sample_count": int(count),
                    }
                )
        all_directions.append(layer_dirs)
    return np.stack(all_directions), pd.DataFrame(rows)


def _alignment_table(
    head_dirs: np.ndarray,
    layers: list[int],
    direction_rows: pd.DataFrame,
    probes: ResidualProbeSet,
    residual_pca_directions: np.ndarray,
    residual_pca_control_k: list[int],
    min_residualized_norm: float = 1e-3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    d_model = probes.d_model
    if residual_pca_directions.ndim != 2:
        raise ValueError("residual_pca_directions must have shape (n_components, d_model)")
    if residual_pca_directions.shape[1] != d_model:
        raise ValueError(
            "residual_pca_directions must have shape (n_components, d_model); "
            f"got {residual_pca_directions.shape} for d_model={d_model}"
        )
    pca_controls = [0, *[int(k) for k in residual_pca_control_k if int(k) > 0]]
    pca_bases: dict[int, np.ndarray] = {}
    for k in pca_controls:
        if k <= 0:
            pca_bases[k] = np.zeros((d_model, 0), dtype=np.float64)
            continue
        n = min(k, len(residual_pca_directions), d_model)
        q, _ = np.linalg.qr(residual_pca_directions[:n].T.astype(np.float64, copy=False))
        pca_bases[k] = q[:, :n]
    for drow in direction_rows.itertuples(index=False):
        raw = probes.directions[int(drow.direction_index)].astype(np.float64)
        raw_norm = max(float(np.linalg.norm(raw)), 1e-12)
        raw_unit = raw / raw_norm
        for resid_pca_k in pca_controls:
            basis = pca_bases[resid_pca_k]
            if basis.shape[1]:
                coords = raw_unit @ basis
                removed = coords @ basis.T
                direction = raw_unit - removed
                fraction_removed = float(np.sum(coords * coords))
                residualized_norm = float(np.linalg.norm(direction))
                valid = bool(residualized_norm >= min_residualized_norm)
                if valid:
                    direction = direction / residualized_norm
            else:
                fraction_removed = 0.0
                residualized_norm = 1.0
                valid = True
                direction = raw_unit
            for layer_i, layer in enumerate(layers):
                for head in range(head_dirs.shape[1]):
                    signed = head_dirs[layer_i, head].astype(np.float64) @ direction if valid else np.full(head_dirs.shape[2], np.nan)
                    aligns = np.abs(signed)
                    rows.append(
                        {
                            "direction_group": str(drow.direction_group),
                            "direction_rank": int(drow.direction_rank),
                            "probe_id": str(drow.probe_id),
                            "probe_family": str(drow.probe_family),
                            "tau_within": float(drow.tau_within),
                            "resid_pca_k": int(resid_pca_k),
                            "fraction_removed_by_resid_pca": fraction_removed,
                            "residualized_direction_norm": residualized_norm,
                            "residualized_alignment_valid": valid,
                            "layer": int(layer),
                            "head": int(head),
                            "max_abs_alignment": float(np.nanmax(aligns)) if valid else float("nan"),
                            "head_subspace_projection_norm": float(np.linalg.norm(signed)) if valid else float("nan"),
                            "argmax_component": int(np.nanargmax(aligns)) if valid else -1,
                        }
                    )
    return pd.DataFrame(rows)


def _bootstrap_median_diff(persistent: np.ndarray, baseline: np.ndarray, seed: int, reps: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = np.zeros(reps, dtype=np.float64)
    for rep in range(reps):
        p = rng.choice(persistent, size=len(persistent), replace=True)
        b = rng.choice(baseline, size=len(baseline), replace=True)
        diffs[rep] = np.median(p) - np.median(b)
    return {
        "median_difference": float(np.median(persistent) - np.median(baseline)),
        "bootstrap_ci_low": float(np.quantile(diffs, 0.025)),
        "bootstrap_ci_high": float(np.quantile(diffs, 0.975)),
    }


def _per_direction_max(alignment: pd.DataFrame, metric: str, resid_pca_k: int) -> pd.DataFrame:
    table = alignment[
        (alignment["resid_pca_k"] == resid_pca_k)
        & alignment["residualized_alignment_valid"].astype(bool)
    ].copy()
    if table.empty:
        return pd.DataFrame()
    return table.groupby(["direction_group", "probe_id"], as_index=False)[metric].max()


def _group_stats(per_direction: pd.DataFrame, metric: str) -> dict[str, object]:
    group_stats: dict[str, object] = {}
    if per_direction.empty:
        return group_stats
    for group, table in per_direction.groupby("direction_group"):
        group_stats[str(group)] = {
            "direction_count": int(len(table)),
            f"median_{metric}_over_heads": float(table[metric].median()),
            f"q90_{metric}_over_heads": float(table[metric].quantile(0.9)),
        }
    return group_stats


def _comparison(per_direction: pd.DataFrame, metric: str, resid_pca_k: int) -> tuple[str, dict[str, float]]:
    status = "not_positive_or_not_evaluable"
    comparison: dict[str, float] = {}
    if per_direction.empty or not {"persistent", "random"}.issubset(set(per_direction["direction_group"].astype(str))):
        return status, comparison
    p = per_direction[per_direction["direction_group"] == "persistent"][metric].to_numpy(dtype=np.float64)
    r = per_direction[per_direction["direction_group"] == "random"][metric].to_numpy(dtype=np.float64)
    if len(p) and len(r):
        comparison = _bootstrap_median_diff(p, r, seed=17)
        if comparison["median_difference"] > 0 and comparison["bootstrap_ci_low"] > 0:
            status = "positive_raw_attention_alignment" if resid_pca_k == 0 else "positive_residual_pca_controlled_attention_alignment"
    return status, comparison


def _residual_pca_control_summary(alignment: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in sorted(int(x) for x in alignment["resid_pca_k"].dropna().unique()):
        item: dict[str, object] = {}
        for metric in ["max_abs_alignment", "head_subspace_projection_norm"]:
            per_direction = _per_direction_max(alignment, metric=metric, resid_pca_k=k)
            status, comparison = _comparison(per_direction, metric=metric, resid_pca_k=k)
            item[metric] = {
                "status": status,
                "group_stats": _group_stats(per_direction, metric=metric),
                "persistent_vs_random": comparison,
            }
        fr = (
            alignment[alignment["resid_pca_k"] == k]
            .groupby(["direction_group", "probe_id"], as_index=False)
            .agg(
                fraction_removed_by_resid_pca=("fraction_removed_by_resid_pca", "first"),
                residualized_direction_norm=("residualized_direction_norm", "first"),
                residualized_alignment_valid=("residualized_alignment_valid", "first"),
            )
        )
        item["residual_pca_containment"] = {
            str(group): {
                "direction_count": int(len(table)),
                "valid_after_residualization_count": int(table["residualized_alignment_valid"].sum()),
                "median_fraction_removed": float(table["fraction_removed_by_resid_pca"].median()),
                "q90_fraction_removed": float(table["fraction_removed_by_resid_pca"].quantile(0.9)),
                "median_residualized_norm": float(table["residualized_direction_norm"].median()),
            }
            for group, table in fr.groupby("direction_group")
        }
        out[str(k)] = item
    return out


def _summary(alignment: pd.DataFrame, metadata: dict[str, object], layers: list[int], components: int, sample_count: int) -> dict[str, object]:
    per_direction = (
        _per_direction_max(alignment, metric="max_abs_alignment", resid_pca_k=0)
        if not alignment.empty
        else pd.DataFrame()
    )
    group_stats = _group_stats(per_direction, metric="max_abs_alignment")
    # Preserve legacy key names in the summary for raw max-PC alignment.
    legacy_stats: dict[str, object] = {}
    for group, item in group_stats.items():
        if isinstance(item, dict):
            legacy_stats[group] = {
                "direction_count": item.get("direction_count", 0),
                "median_max_over_heads": item.get("median_max_abs_alignment_over_heads"),
                "q90_max_over_heads": item.get("q90_max_abs_alignment_over_heads"),
            }
    status, comparison = _comparison(per_direction, metric="max_abs_alignment", resid_pca_k=0)
    return {
        **metadata,
        "status": status,
        "layers": [int(layer) for layer in layers],
        "components_per_head": int(components),
        "sampled_train_positions": int(sample_count),
        "group_stats": legacy_stats,
        "persistent_vs_random_max_head_alignment": comparison,
        "attention_alignment_robustness": _residual_pca_control_summary(alignment),
        "head_output_pca_path": "subspace/head_output_pca.parquet",
        "head_output_pca_directions_path": "subspace/head_output_pca_directions.npz",
        "residual_direction_head_alignment_path": "subspace/residual_direction_head_alignment.parquet",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit attention-output PCA and align persistent residual directions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--layers", default="target", help="target, upstream, all, comma list, or ranges like 0-12")
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--max-positions", type=int, default=50_000)
    parser.add_argument("--resid-pca-control-k", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--min-residualized-norm", type=float, default=1e-3)
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=None,
        help="Docs per forward pass for head-output PCA. Defaults to max(1, 256 // n_layers). "
             "Reduce if OOM with --layers upstream; the default handles 24 GB GPUs safely.",
    )
    args = parser.parse_args()

    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    summary_path = os.path.join(store.subspace_dir, "attention_alignment_summary.json")
    alignment_path = os.path.join(store.subspace_dir, "residual_direction_head_alignment.parquet")
    pca_path = os.path.join(store.subspace_dir, "head_output_pca.parquet")
    directions_path = os.path.join(store.subspace_dir, "head_output_pca_directions.npz")
    if Path(summary_path).exists() and Path(alignment_path).exists() and not args.overwrite:
        return 0

    provider = ResidualStreamProvider(config)
    provider.load()
    n_layers = int(getattr(provider.model.cfg, "n_layers"))
    layers = _parse_layers(args.layers, target_layer=int(config.model.layer_index), n_layers=n_layers)
    train_tokens = token_matrix(split_contexts(store, "train"))
    total_positions = int(train_tokens.shape[0] * train_tokens.shape[1])
    selected = _selected_global_positions(total_positions, args.max_positions, seed=residual_cfg.time_lagged.seed + 41)
    pca_batch_size = args.pca_batch_size if args.pca_batch_size is not None else max(1, 256 // max(len(layers), 1))
    head_dirs, pca_table = _fit_head_output_pca(
        provider=provider,
        train_tokens=train_tokens,
        layers=layers,
        selected_positions=selected,
        batch_size=pca_batch_size,
        components=args.components,
    )
    save_parquet(pca_table, pca_path)
    np.savez_compressed(
        directions_path,
        directions=head_dirs.astype(np.float32),
        layers=np.asarray(layers, dtype=np.int64),
        components=np.asarray(args.components, dtype=np.int64),
        sampled_train_positions=selected.astype(np.int64),
    )

    probes = _load_all_probes(store)
    residual_pca = load_probe_set(store.residual_pca_directions_path)
    timescales = _validation_timescales(store)
    direction_rows = _direction_sets(store, probes, timescales)
    alignment = _alignment_table(
        head_dirs,
        layers,
        direction_rows,
        probes,
        residual_pca_directions=residual_pca.directions,
        residual_pca_control_k=args.resid_pca_control_k,
        min_residualized_norm=args.min_residualized_norm,
    )
    metadata = artifact_metadata(config, "residual_geometry_06_attention_alignment")
    for key, value in metadata.items():
        alignment[key] = value
    save_parquet(alignment, alignment_path)
    summary = _summary(alignment, metadata, layers=layers, components=args.components, sample_count=len(selected))
    save_json(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
