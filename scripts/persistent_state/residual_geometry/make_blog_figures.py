#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch


COLORS = {
    "random": "#6B7280",
    "pca": "#4F46E5",
    "time_lagged": "#D97706",
    "residual_first": "#059669",
    "random_control": "#9CA3AF",
    "persistent": "#B45309",
    "low_lifetime": "#0EA5E9",
    "top_pca": "#4338CA",
    "pca_late": "#818CF8",
    "pca_span_random": "#64748B",
    "attn": "#0F766E",
    "mlp": "#BE123C",
}

LABELS = {
    "random": "Random",
    "pca": "PCA",
    "time_lagged": "Time-lagged",
    "residual_first": "Residual-first",
    "random_control": "Random control",
    "lag_heldout": "Held-out time-lagged",
    "persistent_top31": "Persistent top-31",
    "random31": "Random-31",
    "low_lifetime_time_lagged31": "Low-lifetime TL-31",
    "top_pca31": "Top PCA-31",
    "pca_129_159": "PCA 129-159",
    "random_in_top_pca128_span31": "Random in PCA-128 span",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load(base: Path, rel: str) -> pd.DataFrame:
    path = base / rel
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def fig_tau_histograms(base: Path, out: Path) -> None:
    df = load(base, "residual_probes/autocorr/probe_timescales.parquet")
    val = df[(df["split"] == "val") & (df["control"] == "real")].copy()
    order = ["random", "pca", "time_lagged"]
    bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=True)
    for ax, family in zip(axes, order):
        g = val[val["probe_family"] == family]
        vals = g["tau_within"].dropna().astype(float).clip(lower=1)
        ax.hist(vals, bins=bins, color=COLORS[family], alpha=0.82, edgecolor="white")
        q90 = vals.quantile(0.90)
        q95 = vals.quantile(0.95)
        ax.axvline(q90, color="#111827", lw=1.2, ls="--")
        ax.axvline(q95, color="#111827", lw=1.2, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"])
        ax.set_title(f"{LABELS[family]}\nQ90={q90:g}, Q95={q95:g}")
        ax.set_xlabel("within-document timescale tau")
        ax.set_xlim(1, 512)
    axes[0].set_ylabel("probe count")
    fig.subplots_adjust(top=0.76, bottom=0.15)
    save(fig, out, "F1_tau_histograms_by_probe_family")


def fig_permutation_collapse(base: Path, out: Path) -> None:
    df = load(base, "residual_probes/autocorr/probe_timescales.parquet")
    val_real = df[(df["split"] == "val") & (df["control"] == "real")].copy()
    val_perm = df[(df["split"] == "val") & (df["control"] == "document_permutation")].copy()

    families = ["random", "pca", "time_lagged"]
    x = np.arange(len(families))
    real_medians = []
    perm_medians = []
    for fam in families:
        r = val_real[val_real["probe_family"] == fam].copy()
        cutoff = r["tau_within"].quantile(0.9)
        top_ids = set(r[r["tau_within"] >= cutoff]["probe_id"])
        real_medians.append(r[r["probe_id"].isin(top_ids)]["tau_within"].median())
        p = val_perm[(val_perm["probe_family"] == fam) & (val_perm["probe_id"].isin(top_ids))]
        perm_medians.append(p["tau_perm_within"].median())

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    width = 0.34
    ax.bar(x - width / 2, real_medians, width, color="#111827", label="real order")
    ax.bar(x + width / 2, perm_medians, width, color="#94A3B8", label="document-permuted")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in families])
    ax.set_ylabel("top-decile median tau")
    ax.legend()
    ax.annotate(
        f"{real_medians[2]:.0f} → {perm_medians[2]:.0f}\n94.1% collapse",
        xy=(2 + width / 2, perm_medians[2] + 0.15),
        xytext=(2.35, max(real_medians) * 0.38),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#111827"},
        fontsize=10,
        ha="center",
    )
    fig.subplots_adjust(top=0.96)
    save(fig, out, "F2_permutation_collapse")


def fig_projection_collapse(base: Path, out: Path) -> None:
    df = load(base, "subspace/projection_collapse_summary.parquet")
    d = df[(df["eval_family"] == "lag_heldout") & (df["basis_family"] != "none")].copy()
    basis_order = ["residual_first", "pca", "random_control"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for basis in basis_order:
        for split, ls, alpha in [("val", "--", 0.55), ("test", "-", 1.0)]:
            g = d[(d["basis_family"] == basis) & (d["split"] == split)].sort_values("basis_k")
            if g.empty:
                continue
            ax.plot(
                g["basis_k"],
                g["collapse_fraction"],
                marker="o",
                lw=2.2 if split == "test" else 1.5,
                ls=ls,
                alpha=alpha,
                color=COLORS[basis],
                label=LABELS[basis] if split == "test" else "_nolegend_",
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("basis dimension k")
    ax.set_ylabel("collapse of timescale excess C(k)")
    ax.legend(ncol=1, fontsize=9, loc="lower left")
    save(fig, out, "F3_projection_collapse_by_basis")


def fig_pca_schematic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    outer = Ellipse((0, 0), width=7.2, height=4.5, angle=0, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=2)
    pca = Ellipse((0, 0), width=5.4, height=2.55, angle=18, facecolor="#EEF2FF", edgecolor=COLORS["pca"], lw=2, alpha=0.95)
    persist = Ellipse((0.25, 0.02), width=3.2, height=0.72, angle=-22, facecolor="#FEF3C7", edgecolor=COLORS["persistent"], lw=2.2, alpha=0.95)
    ax.add_patch(outer)
    ax.add_patch(pca)
    ax.add_patch(persist)
    for angle in [18, 108]:
        rad = np.deg2rad(angle)
        ax.plot([-2.55 * np.cos(rad), 2.55 * np.cos(rad)], [-2.55 * np.sin(rad), 2.55 * np.sin(rad)], color=COLORS["pca"], lw=1.3, alpha=0.55)
    for angle in [-22, 68]:
        rad = np.deg2rad(angle)
        ax.plot([-1.65 * np.cos(rad) + 0.25, 1.65 * np.cos(rad) + 0.25], [-1.65 * np.sin(rad) + 0.02, 1.65 * np.sin(rad) + 0.02], color=COLORS["persistent"], lw=1.6)
    ax.text(-3.35, 2.3, "full residual stream", color="#475569", fontsize=11)
    ax.text(-2.15, 1.3, "top PCA span\nhigh variance", color=COLORS["pca"], fontsize=11, weight="bold")
    # label placed to the right of the orange ellipse, arrow points into the ellipse centre
    ax.annotate(
        "persistent subspace\nrotated inside PCA span",
        xy=(1.55, 0.02),
        xytext=(2.1, -1.2),
        color=COLORS["persistent"],
        fontsize=11,
        fontweight="bold",
        ha="left",
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": COLORS["persistent"]},
    )
    ax.text(-3.3, -2.25, "Individual PCA axes are short-lived;\nthe PCA span still contains the persistent rotations.", fontsize=10, color="#111827")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.7, 2.7)
    fig.suptitle("The PCA result is a subspace result, not an axis-by-axis result", y=0.97)
    save(fig, out, "F4_pca_containment_schematic")


def fig_lifetime_excess(base: Path, out: Path) -> None:
    df = load(base, "subspace/persistent_direction_analysis/lifetime_excess_contribution_curve.csv")
    top = load(base, "subspace/persistent_direction_analysis/top31_lifetime_excess_table.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(df["rank"], df["cumulative_global_fraction"], color=COLORS["persistent"], lw=2.4)
    ax.axhline(0.8, color="#111827", ls="--", lw=1)
    ax.axvline(31, color="#111827", ls="--", lw=1)
    ax.scatter([31], [top["cumulative_global_fraction"].iloc[-1]], s=70, color=COLORS["persistent"], zorder=5)
    ax.set_xlim(1, 160)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("probe rank by lifetime excess")
    ax.set_ylabel("cumulative share of global lifetime excess")
    ax.set_title("Eighty percent of the persistence signal concentrates in 31 directions")
    ax.annotate(
        "k80 = 31\nshare = 0.803",
        xy=(31, 0.803),
        xytext=(58, 0.60),
        arrowprops={"arrowstyle": "->", "lw": 1.1},
        fontsize=10,
    )
    ax.text(88, 0.28, "Top-31 effective rank: 28.2\nmax pairwise |cos|: 0.237", fontsize=10, color="#374151")
    save(fig, out, "F5_lifetime_excess_cumulative")


def fig_pca_vs_persistent(base: Path, out: Path) -> None:
    df = load(base, "subspace/pca_vs_persistent_alignment/group_summary.csv")
    groups = ["random31", "persistent_top31", "top_pca31", "random_in_top_pca128_span31"]
    d = df.set_index("group").loc[groups]
    labels = [LABELS[g] for g in groups]
    colors = [COLORS["random"], COLORS["persistent"], COLORS["top_pca"], COLORS["pca_span_random"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=False)
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.35)
    x = np.arange(len(groups))
    axes[0].bar(x, d["head_subspace_direction_median"], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("median max head-subspace alignment")
    axes[0].set_title("Raw attention alignment", pad=8)
    # annotation below the top-PCA bar, pointing up into it
    axes[0].annotate(
        "top PCA > persistent",
        xy=(2, d.loc["top_pca31", "head_subspace_direction_median"] * 0.5),
        xytext=(2.55, 0.13),
        arrowprops={"arrowstyle": "->", "lw": 1.1},
        fontsize=9,
        ha="left",
    )
    axes[1].bar(x, d["M0_E_attn_max_layer_direction_median"], color=colors)
    axes[1].axhline(0, color="#111827", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("median M0 attention excess")
    axes[1].set_title("Attention-specific excess beyond residual PCA", pad=8)
    # annotation in the negative-y region where there is open space
    axes[1].annotate(
        "persistent: +0.116\ntop PCA: −0.399",
        xy=(1, d.loc["persistent_top31", "M0_E_attn_max_layer_direction_median"]),
        xytext=(0.2, -0.28),
        arrowprops={"arrowstyle": "->", "lw": 1.1},
        fontsize=9,
        ha="left",
    )
    fig.suptitle(
        "High variance and raw attention alignment are not sufficient for persistence",
        y=0.98,
        fontsize=12,
    )
    save(fig, out, "F6_pca_vs_persistent_stress_test")


def fig_semantic_cards(base: Path, out: Path) -> None:
    summary = load(base, "subspace/semantic_dossiers/direction_activation_summary.csv").set_index("rank")
    examples = [
        (1, "technical / instructional state", "device setup, networking, configuration, software support pages"),
        (3, "SEO / catalog-template state", "repetitive product pages, grinder/crusher listings, furniture catalog text"),
        (29, "recipe / food-menu state", "recipe and menu-like spans with persistent food/lifestyle formatting"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, (rank, pattern, paraphrase) in enumerate(examples):
        y = 0.77 - i * 0.25
        row = summary.loc[rank]
        ax.add_patch(plt.Rectangle((0.02, y - 0.02), 0.96, 0.17, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1))
        ax.text(0.045, y + 0.095, f"rank {rank}", fontsize=10, color="#6B7280", weight="bold")
        ax.text(0.135, y + 0.095, f"tau = {row['tau_within']:.0f}", fontsize=10, color=COLORS["persistent"], weight="bold")
        ax.text(0.27, y + 0.095, pattern, fontsize=12, color="#111827", weight="bold")
        ax.text(0.045, y + 0.045, fill(paraphrase, 92), fontsize=10.5, color="#374151")
        ax.text(0.045, y + 0.005, f"q95 high spans length>=8: {int(row['n_q95_spans_len_ge_8'])}", fontsize=9.5, color="#6B7280")
    save(fig, out, "F7_semantic_span_cards")


def fig_singular_spectrum(base: Path, out: Path) -> None:
    df = load(base, "subspace/persistent_direction_analysis/top31_subspace_singular_values.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(df["component"], df["singular_value"], marker="o", color=COLORS["persistent"], lw=2)
    ax.set_xlabel("component")
    ax.set_ylabel("singular value")
    ax.set_title("Top-31 directions remain high-rank rather than collapsing to one axis")
    ax.text(16, df["singular_value"].min() + 0.03, "participation-ratio effective rank = 28.2", fontsize=10, color="#374151")
    save(fig, out, "A1_top31_singular_spectrum")


def fig_cosine_heatmap(base: Path, out: Path) -> None:
    pairs = load(base, "subspace/persistent_direction_analysis/top31_pairwise_abs_cosine.csv")
    n = 31
    mat = np.eye(n)
    for _, row in pairs.iterrows():
        i = int(row["rank_i"]) - 1
        j = int(row["rank_j"]) - 1
        mat[i, j] = mat[j, i] = float(row["abs_cosine"])
    np.fill_diagonal(mat, np.nan)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(mat, cmap="magma_r", vmin=0, vmax=0.25)
    ax.set_xlabel("top-31 rank")
    ax.set_ylabel("top-31 rank")
    ax.set_title("Pairwise abs-cosine among persistent top-31")
    ax.set_xticks([0, 9, 19, 30])
    ax.set_xticklabels(["1", "10", "20", "31"])
    ax.set_yticks([0, 9, 19, 30])
    ax.set_yticklabels(["1", "10", "20", "31"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|cosine|")
    ax.text(0.02, -0.11, "max=0.237, median=0.035, Q90=0.092", transform=ax.transAxes, fontsize=9, color="#374151")
    save(fig, out, "A2_top31_pairwise_abs_cosine")


def fig_attention_control_k(base: Path, out: Path) -> None:
    with open(base / "subspace/block_output_subspace_summary.json") as f:
        s = json.load(f)
    ks = sorted(int(k) for k in s["across_layers"].keys())
    metrics = [
        ("delta_E_attn", "Attention excess", COLORS["attn"]),
        ("delta_E_mlp", "MLP excess", COLORS["mlp"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for metric, label, color in metrics:
        y, lo, hi = [], [], []
        for k in ks:
            d = s["across_layers"][str(k)][metric]
            y.append(d["median_difference"])
            lo.append(d["bootstrap_ci_low"])
            hi.append(d["bootstrap_ci_high"])
        y = np.array(y)
        lo = np.array(lo)
        hi = np.array(hi)
        ax.plot(ks, y, marker="o", color=color, lw=2.2, label=label)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.16)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xscale("symlog", linthresh=1, base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("residual PCA components removed before alignment")
    ax.set_ylabel("persistent minus random median excess")
    ax.legend()
    save(fig, out, "A3_attention_alignment_vs_residual_pca_control")


def fig_m0_by_layer(base: Path, out: Path) -> None:
    df = load(base, "subspace/block_output_subspace_overlap.parquet")
    d = df[(df["direction_group"].isin(["persistent", "random"])) & (df["resid_pca_control_k"] == 128)].copy()
    rows = []
    for layer in sorted(d["layer"].unique()):
        for metric in ["E_attn", "E_mlp"]:
            p = d[(d["layer"] == layer) & (d["direction_group"] == "persistent")][metric].median()
            r = d[(d["layer"] == layer) & (d["direction_group"] == "random")][metric].median()
            rows.append({"layer": layer, "metric": metric, "delta": p - r})
    plot = pd.DataFrame(rows)
    layers = sorted(plot["layer"].unique())
    x = np.arange(len(layers))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, metric, color, label in [(-width / 2, "E_attn", COLORS["attn"], "Attention excess"), (width / 2, "E_mlp", COLORS["mlp"], "MLP excess")]:
        g = plot[plot["metric"] == metric].set_index("layer").loc[layers]
        ax.bar(x + offset, g["delta"], width, color=color, label=label)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("layer")
    ax.set_ylabel("persistent minus random median excess (k=128 control)")
    ax.set_title("Late-layer attention has the strongest block-output excess")
    ax.legend()
    save(fig, out, "A4_m0_attention_vs_mlp_by_layer")


def fig_top_decay_curves(base: Path, out: Path) -> None:
    top = load(base, "subspace/persistent_direction_analysis/top31_lifetime_excess_table.csv").head(8)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for _, row in top.iterrows():
        fam = row["probe_family"]
        profile_path = base / f"residual_probes/autocorr/profiles_val_{fam}_within.npz"
        with np.load(profile_path, allow_pickle=False) as data:
            ids = data["probe_ids"].astype(str)
            profiles = data["profiles"]
        idxs = np.where(ids == str(row["probe_id"]))[0]
        if len(idxs) == 0:
            continue
        prof = profiles[idxs[0]]
        ax.plot(np.arange(len(prof)), prof, lw=1.6, alpha=0.75, label=f"rank {int(row['rank'])}")
    ax.axhline(1 / np.e, color="#111827", lw=1, ls="--", label="1/e")
    ax.set_xlim(0, 128)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("lag")
    ax.set_ylabel("within-document autocorrelation")
    ax.set_title("Top persistent directions decay at very different rates")
    ax.legend(ncol=2, fontsize=8)
    save(fig, out, "A5_top_persistent_decay_curves")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate residual-geometry blog figures.")
    parser.add_argument(
        "--run-dir",
        default="outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot",
        help="Residual-geometry run artifact directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="figures/residual-geometry",
        help="Output directory for figures.",
    )
    args = parser.parse_args()
    base = Path(args.run_dir)
    out = Path(args.out_dir)
    set_style()
    fig_tau_histograms(base, out)
    fig_permutation_collapse(base, out)
    fig_projection_collapse(base, out)
    fig_pca_schematic(out)
    fig_lifetime_excess(base, out)
    fig_pca_vs_persistent(base, out)
    fig_semantic_cards(base, out)
    fig_singular_spectrum(base, out)
    fig_cosine_heatmap(base, out)
    fig_attention_control_k(base, out)
    fig_m0_by_layer(base, out)
    fig_top_decay_curves(base, out)
    print(f"Wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
