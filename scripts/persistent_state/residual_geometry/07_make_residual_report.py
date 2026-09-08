#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from _common import load_config_and_store
from residual_geometry.reporting.residual_report import write_residual_report
from residual_geometry.utils.io import load_json


def make_b1_plots(store, timescales: pd.DataFrame) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"plot_status": "matplotlib_unavailable"}
    plot_dir = f"{store.reports_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    outputs: dict[str, str] = {}
    real = timescales[timescales["control"] == "real"] if "control" in timescales else timescales
    val = real[real["split"] == "val"] if "split" in real else real
    if not val.empty and "tau_within" in val:
        plt.figure(figsize=(8, 5))
        for family, group in val.groupby("probe_family"):
            plt.hist(group["tau_within"], bins=30, alpha=0.45, label=str(family))
        plt.xlabel("tau_within")
        plt.ylabel("probe count")
        plt.legend()
        path = f"{plot_dir}/tau_within_histograms.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["tau_within_histograms"] = path

        q = val.groupby("probe_family")["tau_within"].quantile([0.9, 0.95]).unstack()
        plt.figure(figsize=(7, 4))
        q.plot(kind="bar", ax=plt.gca())
        plt.ylabel("tau_within")
        path = f"{plot_dir}/upper_tail_comparison.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["upper_tail_comparison"] = path

    if not val.empty and {"tau_raw", "tau_within"}.issubset(val.columns):
        plt.figure(figsize=(5, 5))
        for family, group in val.groupby("probe_family"):
            plt.scatter(group["tau_raw"], group["tau_within"], s=12, alpha=0.6, label=str(family))
        plt.xlabel("tau_raw")
        plt.ylabel("tau_within")
        plt.legend()
        path = f"{plot_dir}/raw_vs_within_tau_scatter.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["raw_vs_within_tau_scatter"] = path

    perm = timescales[(timescales["control"] == "document_permutation") & (timescales["split"] == "val")] if {"control", "split"}.issubset(timescales.columns) else pd.DataFrame()
    if not perm.empty and "tau_perm_within" in perm:
        plt.figure(figsize=(8, 5))
        plt.hist(val["tau_within"], bins=30, alpha=0.45, label="real")
        plt.hist(perm["tau_perm_within"], bins=30, alpha=0.45, label="document permutation")
        plt.xlabel("tau_within")
        plt.ylabel("probe count")
        plt.legend()
        path = f"{plot_dir}/real_vs_document_permutation_tau.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["real_vs_document_permutation_tau"] = path

    # Top persistent decay curves, using saved validation within profiles.
    top = val.sort_values("tau_within", ascending=False).head(10) if not val.empty and "tau_within" in val else pd.DataFrame()
    if not top.empty:
        plt.figure(figsize=(8, 5))
        for family, group in top.groupby("probe_family"):
            profile_path = store.residual_autocorr_profiles_path("val", str(family), "within")
            if not os.path.exists(profile_path):
                continue
            with np.load(profile_path, allow_pickle=False) as data:
                ids = data["probe_ids"].astype(str)
                profiles = data["profiles"]
            id_to_idx = {pid: idx for idx, pid in enumerate(ids)}
            for _, row in group.head(4).iterrows():
                idx = id_to_idx.get(str(row["probe_id"]))
                if idx is not None:
                    plt.plot(profiles[idx], alpha=0.75, label=str(row["probe_id"]))
        plt.xlabel("lag")
        plt.ylabel("R_within(k)")
        plt.legend(fontsize=6)
        path = f"{plot_dir}/top_persistent_decay_curves.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["top_persistent_decay_curves"] = path
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Write residual geometry report.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    _, store = load_config_and_store(args.config)
    timescales = pd.read_parquet(store.residual_probe_timescales_path) if os.path.exists(store.residual_probe_timescales_path) else pd.DataFrame()
    if not timescales.empty and "control" in timescales:
        plot_source = timescales.copy()
        all_real_timescales = timescales[timescales["control"] == "real"].copy()
        timescales = all_real_timescales[all_real_timescales["split"] == "val"].copy() if "split" in all_real_timescales else all_real_timescales
    else:
        plot_source = timescales.copy()
        all_real_timescales = timescales.copy()
    extra = {}
    for path, key in [
        (store.residual_probe_summary_path, "residual_probe_summary"),
        (f"{store.subspace_dir}/dimensionality_summary.json", "dimensionality_summary"),
        (f"{store.subspace_dir}/projection_collapse_summary.json", "projection_collapse_summary"),
        (store.fat_subspace_summary_path, "fat_subspace_summary"),
        (f"{store.subspace_dir}/attention_alignment_summary.json", "attention_alignment_summary"),
        (f"{store.subspace_dir}/block_output_subspace_summary.json", "block_output_subspace_summary"),
        (f"{store.subspace_dir}/attention_transport_summary.json", "attention_transport_summary"),
        (f"{store.subspace_dir}/head_ablation_m1_lite_summary.json", "head_ablation_m1_lite_summary"),
        (store.time_lagged_fit_summary_path, "time_lagged_fit_summary"),
    ]:
        if os.path.exists(path):
            extra[key] = load_json(path)
    extra["plots"] = make_b1_plots(store, plot_source) if not plot_source.empty else {}
    write_residual_report(
        report_path=store.residual_geometry_report_path,
        summary_path=store.residual_geometry_summary_path,
        timescales=timescales,
        architecture_path=store.resolved_model_architecture_path,
        extra_summary=extra,
        decision_split="val",
        all_real_timescales=all_real_timescales,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
