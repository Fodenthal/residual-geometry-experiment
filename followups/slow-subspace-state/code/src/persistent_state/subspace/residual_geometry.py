from __future__ import annotations

import numpy as np
import pandas as pd

from src.persistent_state.residuals.probes import ResidualProbeSet


PRIORITY = {"time_lagged": 0, "pca": 1, "attention_head_pca": 2, "random": 3}


def deduplicate_ranked_probes(
    probes: ResidualProbeSet,
    timescales: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    table = timescales.copy()
    if "probe_id" not in table:
        table["probe_id"] = probes.probe_ids
    table["priority"] = table["probe_family"].map(lambda fam: PRIORITY.get(str(fam), 99))
    table = table.sort_values(
        ["tau_within", "priority", "probe_id"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    kept: list[int] = []
    id_to_idx = {str(pid): idx for idx, pid in enumerate(probes.probe_ids.astype(str))}
    n_max = len(table)
    d_model = probes.directions.shape[1]
    kept_matrix = np.empty((n_max, d_model), dtype=np.float64)  # pre-allocated, no reallocation per step
    n_kept = 0
    probe_ids_col = table["probe_id"].to_numpy(dtype=str)
    row_indices = table.index.to_numpy()
    for i in range(len(table)):
        probe_idx = id_to_idx.get(probe_ids_col[i])
        if probe_idx is None:
            continue
        direction = probes.directions[probe_idx].astype(np.float64)
        if n_kept > 0 and bool(np.any(np.abs(kept_matrix[:n_kept] @ direction) > threshold)):
            continue
        kept.append(row_indices[i])
        kept_matrix[n_kept] = direction
        n_kept += 1
    out = table.loc[kept].drop(columns=["priority"]).reset_index(drop=True)
    out["dedup_rank"] = np.arange(len(out))
    return out


def orthonormal_basis(directions: np.ndarray, k: int) -> np.ndarray:
    if directions.ndim != 2:
        raise ValueError("directions must have shape (n, d_model)")
    selected = directions[:k].T.astype(np.float64)
    q, _ = np.linalg.qr(selected)
    return q[:, : min(k, q.shape[1])].astype(np.float32)


def lifetime_excess_summary(timescales: pd.DataFrame, k_values: list[int]) -> dict[str, object]:
    valid = timescales[timescales["tau_valid_within"].astype(bool)].copy() if "tau_valid_within" in timescales else timescales.copy()
    if valid.empty or "tau_within" not in valid:
        return {"valid_probe_count": 0}
    random = valid[valid["probe_family"] == "random"]
    baseline = float(random["tau_within"].median()) if not random.empty else float(valid["tau_within"].median())
    ranked = valid.sort_values("tau_within", ascending=False).reset_index(drop=True)
    excess = np.maximum(ranked["tau_within"].to_numpy(dtype=np.float64) - baseline, 0.0)
    cumulative = np.cumsum(excess)
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    if total > 0:
        k80 = int(np.searchsorted(cumulative, 0.8 * total) + 1)
    else:
        k80 = 0
    summary: dict[str, object] = {
        "valid_probe_count": int(len(valid)),
        "random_tau_within_median": baseline,
        "k_80pct_lifetime_excess": k80,
        "right_censored_probe_fraction": float(valid["right_censored_within"].mean()) if "right_censored_within" in valid else 0.0,
        "k_values": [int(k) for k in k_values],
    }
    if k80:
        summary["top_k_family_composition_at_k80"] = ranked.head(k80)["probe_family"].value_counts().to_dict()
    for family in ["random", "pca", "time_lagged"]:
        group = valid[valid["probe_family"] == family].sort_values("tau_within", ascending=False)
        if group.empty:
            continue
        family_excess = np.maximum(group["tau_within"].to_numpy(dtype=np.float64) - baseline, 0.0)
        family_cum = np.cumsum(family_excess)
        family_total = float(family_cum[-1]) if len(family_cum) else 0.0
        summary[f"k_80pct_{family}_only"] = int(np.searchsorted(family_cum, 0.8 * family_total) + 1) if family_total > 0 else 0
    return summary
