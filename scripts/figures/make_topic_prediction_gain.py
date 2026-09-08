import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


parser = argparse.ArgumentParser(description="Build the supervised topic-gain figure.")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "figures",
)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

labels = ["Slow-31", "PCA-31", "Random-31 median\n(20 controls)"]
values = [0.033, 0.017, 0.010]
# Match the palette used by the original timescale-distribution figure:
# time-lagged/slow = orange, PCA = violet, random = slate gray.
colors = ["#D08A38", "#6458E1", "#7C7F8C"]
y = [2, 1, 0]

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 19,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
fig.subplots_adjust(left=0.22, right=0.93, top=0.75, bottom=0.24)

# Light stems make the comparison readable without implying uncertainty ranges.
for yi, value, color in zip(y, values, colors):
    ax.hlines(yi, 0, value, color=color, linewidth=6, alpha=0.34, zorder=1)
    marker = "D" if yi == 0 else "o"
    size = 115 if yi == 2 else 90
    ax.scatter(
        [value],
        [yi],
        s=size,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.text(
        value + 0.00105,
        yi,
        f"{value:.3f} nats",
        va="center",
        ha="left",
        fontsize=12.5,
        fontweight="bold" if yi == 2 else "normal",
        color="#17212B",
    )

ax.axvline(0, color="#4C5661", linewidth=1.3, linestyle=(0, (3, 3)), zorder=0)
ax.text(
    0.00035,
    2.43,
    "nuisance-only baseline",
    color="#59636E",
    fontsize=10.5,
    va="bottom",
)

# Direct effect-size annotations keep the figure interpretable without a legend.
ax.annotate(
    "+0.016 vs. PCA-31",
    xy=(0.033, 2),
    xytext=(0.0238, 2.43),
    ha="center",
    va="bottom",
    fontsize=10.5,
    color="#955B1C",
    arrowprops=dict(
        arrowstyle="-",
        color="#C99A62",
        linewidth=1.2,
        connectionstyle="angle3,angleA=0,angleB=90",
    ),
)

ax.set_yticks(y, labels)
for tick, color in zip(ax.get_yticklabels(), colors):
    tick.set_color(color)
    tick.set_fontweight("bold")

ax.set_xlim(-0.0005, 0.0395)
ax.set_ylim(-0.58, 2.63)
ax.xaxis.set_major_locator(MultipleLocator(0.01))
ax.xaxis.set_minor_locator(MultipleLocator(0.005))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: "0" if abs(x) < 1e-9 else f"{x:.2f}"))
ax.grid(axis="x", which="major", color="#D9DEE3", linewidth=0.9)
ax.grid(axis="x", which="minor", color="#EEF1F3", linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0, pad=13)
ax.tick_params(axis="x", colors="#45515C")

ax.set_xlabel(
    "Held-out cross-entropy reduction (nats; higher is better)",
    labelpad=14,
    color="#26313B",
)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#AEB6BE")

fig.suptitle(
    "Slow-31 carries more document-topic information",
    x=0.22,
    y=0.935,
    ha="left",
    fontsize=19,
    fontweight="bold",
    color="#15202A",
)
fig.text(
    0.22,
    0.865,
    "Incremental performance beyond a classifier given position, token-derived,\n"
    "and local-window nuisance features",
    ha="left",
    va="top",
    fontsize=12.2,
    color="#53606B",
    linespacing=1.35,
)
fig.text(
    0.22,
    0.065,
    "Random-31 is the reported median across 20 equal-rank random controls.\n"
    "All values are evaluated on held-out documents.",
    ha="left",
    va="bottom",
    fontsize=9.6,
    linespacing=1.25,
    color="#65717C",
)

stem = args.output_dir / "topic-prediction-gain"
fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
fig.savefig(stem.with_suffix(".svg"), facecolor="white")
fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
plt.close(fig)

for suffix in (".png", ".svg", ".pdf"):
    print(stem.with_suffix(suffix))
