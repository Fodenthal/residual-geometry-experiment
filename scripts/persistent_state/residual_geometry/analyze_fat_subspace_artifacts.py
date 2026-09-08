#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_config_and_store


LOWER_TAIL_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
BOOTSTRAP_QUANTILES = [0.05, 0.10, 0.25, 0.50]


def _valid_timescales(timescales: pd.DataFrame) -> pd.DataFrame:
    required = {
        "split",
        "probe_id",
        "diagnostic_family",
        "tau_within",
        "tau_valid_within",
        "right_censored_within",
        "positive_profile_area",
    }
    missing = required - set(timescales)
    if missing:
        raise ValueError(f"direction_timescales.parquet is missing columns: {sorted(missing)}")
    return timescales[timescales["tau_valid_within"].astype(bool)].copy()


def _ambient_null_bands(timescales: pd.DataFrame) -> dict[str, float]:
    ambient = timescales[timescales["diagnostic_family"].astype(str) == "ambient_random"]
    return {
        str(split): float(group["tau_within"].quantile(0.95))
        for split, group in ambient.groupby("split")
    }


def _quantile_columns(row: dict[str, object], group: pd.DataFrame) -> None:
    for metric in ["tau_within", "positive_profile_area"]:
        for quantile in LOWER_TAIL_QUANTILES:
            row[f"q{int(quantile * 100):02d}_{metric}"] = float(group[metric].quantile(quantile))


def family_lower_tail_summary(timescales: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_timescales(timescales)
    null_bands = _ambient_null_bands(valid)
    rows: list[dict[str, object]] = []
    for (split, family), group in valid.groupby(["split", "diagnostic_family"]):
        null_band = null_bands.get(str(split))
        row: dict[str, object] = {
            "split": str(split),
            "diagnostic_family": str(family),
            "direction_count": int(len(group)),
            "ambient_random_q95_tau_within": null_band,
            "fraction_tau_above_ambient_q95": (
                float((group["tau_within"] > null_band).mean()) if null_band is not None else float("nan")
            ),
            "fraction_tau_at_or_below_ambient_q95": (
                float((group["tau_within"] <= null_band).mean()) if null_band is not None else float("nan")
            ),
            "right_censoring_rate": float(group["right_censored_within"].mean()),
        }
        _quantile_columns(row, group)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "diagnostic_family"]).reset_index(drop=True)


def direction_bootstrap_intervals(
    timescales: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    valid = _valid_timescales(timescales)
    null_bands = _ambient_null_bands(valid)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (split, family), group in valid.groupby(["split", "diagnostic_family"]):
        tau = group["tau_within"].to_numpy(dtype=np.float64)
        area = group["positive_profile_area"].to_numpy(dtype=np.float64)
        indices = rng.integers(0, len(group), size=(replicates, len(group)))
        sampled_tau = tau[indices]
        sampled_area = area[indices]
        for metric, samples in [("tau_within", sampled_tau), ("positive_profile_area", sampled_area)]:
            for quantile in BOOTSTRAP_QUANTILES:
                estimates = np.quantile(samples, quantile, axis=1)
                rows.append(
                    {
                        "split": str(split),
                        "diagnostic_family": str(family),
                        "metric": metric,
                        "quantile": float(quantile),
                        "estimate": float(group[metric].quantile(quantile)),
                        "bootstrap_ci_low": float(np.quantile(estimates, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(estimates, 0.975)),
                        "bootstrap_replicates": int(replicates),
                        "bootstrap_unit": "sampled_direction",
                    }
                )
        null_band = null_bands.get(str(split))
        if null_band is not None:
            estimates = (sampled_tau > null_band).mean(axis=1)
            rows.append(
                {
                    "split": str(split),
                    "diagnostic_family": str(family),
                    "metric": "fraction_tau_above_ambient_q95",
                    "quantile": float("nan"),
                    "estimate": float((tau > null_band).mean()),
                    "bootstrap_ci_low": float(np.quantile(estimates, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(estimates, 0.975)),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_unit": "sampled_direction",
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "diagnostic_family", "metric", "quantile"]).reset_index(drop=True)


def split_stability_summary(timescales: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_timescales(timescales)
    null_bands = _ambient_null_bands(valid)
    paired = valid[valid["split"].astype(str).isin(["val", "test"])].pivot_table(
        index=["diagnostic_family", "probe_id"],
        columns="split",
        values="tau_within",
        aggfunc="first",
    )
    if "val" not in paired or "test" not in paired:
        raise ValueError("direction_timescales.parquet must contain val and test rows")
    paired = paired.dropna(subset=["val", "test"]).reset_index()
    rows: list[dict[str, object]] = []
    for family, group in paired.groupby("diagnostic_family"):
        val = group["val"].astype(float)
        test = group["test"].astype(float)
        val_null = null_bands.get("val")
        test_null = null_bands.get("test")
        rows.append(
            {
                "diagnostic_family": str(family),
                "paired_direction_count": int(len(group)),
                "spearman_val_test_tau": float(val.corr(test, method="spearman")),
                "q50_tau_within_val": float(val.median()),
                "q50_tau_within_test": float(test.median()),
                "median_absolute_tau_change": float((test - val).abs().median()),
                "fraction_above_split_null_on_val": (
                    float((val > val_null).mean()) if val_null is not None else float("nan")
                ),
                "fraction_above_split_null_on_test": (
                    float((test > test_null).mean()) if test_null is not None else float("nan")
                ),
                "fraction_above_split_null_on_both": (
                    float(((val > val_null) & (test > test_null)).mean())
                    if val_null is not None and test_null is not None
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("diagnostic_family").reset_index(drop=True)


def _sample_index(probe_id: str) -> int:
    try:
        return int(str(probe_id).rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse generated sample index from probe_id={probe_id!r}") from exc


def core_loading_sensitivity(
    timescales: pd.DataFrame,
    coefficient_matrix: np.ndarray,
    *,
    k_fat: int,
    prefixes: list[int],
) -> pd.DataFrame:
    valid = _valid_timescales(timescales)
    null_bands = _ambient_null_bands(valid)
    family = f"random_in_q{k_fat}"
    endpoint = valid[valid["diagnostic_family"].astype(str) == family].copy()
    if endpoint.empty:
        raise ValueError(f"Missing endpoint family {family}")
    if coefficient_matrix.ndim != 2 or coefficient_matrix.shape[1] != k_fat:
        raise ValueError(f"coefficient_matrix must have shape (samples, {k_fat})")
    endpoint["sample_index"] = endpoint["probe_id"].astype(str).map(_sample_index)
    if endpoint["sample_index"].max() >= len(coefficient_matrix):
        raise ValueError("Generated endpoint probe index exceeds coefficient_matrix rows")
    squared = np.asarray(coefficient_matrix, dtype=np.float64) ** 2
    total_energy = squared.sum(axis=1)
    rows: list[dict[str, object]] = []
    for split, group in endpoint.groupby("split"):
        selected = group["sample_index"].to_numpy(dtype=np.int64)
        null_band = null_bands.get(str(split))
        for prefix in sorted(set(int(value) for value in prefixes if 0 < int(value) < k_fat)):
            energy = squared[selected, :prefix].sum(axis=1) / total_energy[selected]
            enriched = group.assign(core_energy_fraction=energy)
            cutoff = float(enriched["core_energy_fraction"].quantile(0.25))
            low = enriched[enriched["core_energy_fraction"] <= cutoff]
            high = enriched[enriched["core_energy_fraction"] >= enriched["core_energy_fraction"].quantile(0.75)]
            rows.append(
                {
                    "split": str(split),
                    "diagnostic_family": family,
                    "core_prefix_k": int(prefix),
                    "direction_count": int(len(enriched)),
                    "expected_mean_core_energy_fraction": float(prefix / k_fat),
                    "observed_mean_core_energy_fraction": float(enriched["core_energy_fraction"].mean()),
                    "core_energy_fraction_q25": cutoff,
                    "tau_vs_core_energy_spearman": float(
                        enriched["tau_within"].corr(enriched["core_energy_fraction"], method="spearman")
                    ),
                    "lowest_core_loading_quartile_count": int(len(low)),
                    "lowest_core_loading_quartile_q10_tau_within": float(low["tau_within"].quantile(0.10)),
                    "lowest_core_loading_quartile_q25_tau_within": float(low["tau_within"].quantile(0.25)),
                    "lowest_core_loading_quartile_q50_tau_within": float(low["tau_within"].quantile(0.50)),
                    "lowest_core_loading_quartile_q50_positive_profile_area": float(
                        low["positive_profile_area"].quantile(0.50)
                    ),
                    "lowest_core_loading_quartile_fraction_tau_above_ambient_q95": (
                        float((low["tau_within"] > null_band).mean()) if null_band is not None else float("nan")
                    ),
                    "highest_core_loading_quartile_q50_tau_within": float(high["tau_within"].quantile(0.50)),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "core_prefix_k"]).reset_index(drop=True)


def _headline_rows(lower_tail: pd.DataFrame, k_fat: int) -> pd.DataFrame:
    families = [
        f"random_in_q{k_fat}",
        "ambient_random",
        "random_in_pca31",
        "random_in_pca128",
        "random_in_pca256",
        f"mix_band_9_{k_fat}",
        f"mix_band_17_{k_fat}",
    ]
    return lower_tail[
        (lower_tail["split"] == "test") & lower_tail["diagnostic_family"].isin(families)
    ].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze saved fat-subspace artifacts without model forward passes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--direction-bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--direction-bootstrap-seed", type=int, default=37)
    parser.add_argument("--core-prefixes", nargs="+", type=int, default=[1, 3, 5, 8])
    args = parser.parse_args()

    _, store = load_config_and_store(args.config)
    fat_dir = Path(store.fat_subspace_dir)
    summary_path = fat_dir / "fat_subspace_summary.json"
    timescale_path = fat_dir / "direction_timescales.parquet"
    nested_path = fat_dir / "nested_random_in_span_directions.npz"
    for path in [summary_path, timescale_path, nested_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run 05b_fat_subspace_diagnostics.py first.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    k_fat = int(summary["k_fat"])
    timescales = pd.read_parquet(timescale_path)
    with np.load(nested_path, allow_pickle=False) as nested:
        coefficient_matrix = nested["coefficient_matrix"].astype(np.float32)

    out_dir = Path(args.out_dir or fat_dir / "artifact_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    lower_tail = family_lower_tail_summary(timescales)
    direction_bootstrap = direction_bootstrap_intervals(
        timescales,
        replicates=args.direction_bootstrap_replicates,
        seed=args.direction_bootstrap_seed,
    )
    split_stability = split_stability_summary(timescales)
    core_loading = core_loading_sensitivity(
        timescales,
        coefficient_matrix,
        k_fat=k_fat,
        prefixes=args.core_prefixes,
    )
    lower_tail.to_csv(out_dir / "family_lower_tail_summary.csv", index=False)
    direction_bootstrap.to_csv(out_dir / "direction_bootstrap_intervals.csv", index=False)
    split_stability.to_csv(out_dir / "split_stability_summary.csv", index=False)
    core_loading.to_csv(out_dir / "core_loading_sensitivity.csv", index=False)

    headline = _headline_rows(lower_tail, k_fat)
    endpoint_stability = split_stability[
        split_stability["diagnostic_family"] == f"random_in_q{k_fat}"
    ].to_dict(orient="records")
    derived_summary = {
        "analysis_scope": (
            "Artifact-only follow-up. No model forward pass, probe refit, direction resampling, "
            "or locked-artifact mutation is performed."
        ),
        "k_fat": k_fat,
        "direction_bootstrap_replicates": int(args.direction_bootstrap_replicates),
        "direction_bootstrap_seed": int(args.direction_bootstrap_seed),
        "core_prefixes": [int(value) for value in args.core_prefixes],
        "headline_test_lower_tail": headline.to_dict(orient="records"),
        "endpoint_val_test_stability": endpoint_stability,
        "limitations": [
            "Direction-bootstrap intervals characterize Monte Carlo uncertainty for the already sampled directions.",
            "This does not replace a prospectively locked fresh split.",
            "Core-loading sensitivity uses orthogonalized Q-prefix coordinates; lower-band controls separately test saved source-probe tail spans.",
        ],
        "outputs": {
            "family_lower_tail_summary": str(out_dir / "family_lower_tail_summary.csv"),
            "direction_bootstrap_intervals": str(out_dir / "direction_bootstrap_intervals.csv"),
            "split_stability_summary": str(out_dir / "split_stability_summary.csv"),
            "core_loading_sensitivity": str(out_dir / "core_loading_sensitivity.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(derived_summary, indent=2) + "\n", encoding="utf-8")

    print("Test lower-tail headline:")
    print(headline.to_string(index=False))
    print("\nVal/test stability for the full candidate-pool endpoint:")
    print(split_stability[split_stability["diagnostic_family"] == f"random_in_q{k_fat}"].to_string(index=False))
    print("\nLow-core-loading sensitivity:")
    print(core_loading.to_string(index=False))
    print(f"\nWrote artifact-only robustness analysis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
