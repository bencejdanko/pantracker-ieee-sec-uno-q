"""Charts for the deck. Every value is copied from tracker/bench/SCORECARD.md.

Run: tracker/export/.venv/bin/python deck/charts.py  (writes deck/img/*.svg)
"""
import os

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)
for f in [f for f in os.listdir(os.path.join(HERE, "fonts")) if f.endswith(".ttf")]:
    font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))

RED = "#E3120B"      # deployed config
BLUE = "#006BA2"     # GPU
GREY = "#B7C6CF"     # other configs (always direct-labelled)
INK = "#121212"
MUTED = "#595959"
GRID = "#D9D9D9"

matplotlib.rcParams.update({
    "font.family": "Source Sans 3", "font.size": 14, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
    "svg.fonttype": "path", "figure.facecolor": "white", "axes.facecolor": "white",
})


def style(ax, grid_axis="x"):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=13)


def header(fig, title, sub, source):
    fig.patches.append(plt.Rectangle((0.012, 0.955), 0.05, 0.022, transform=fig.transFigure,
                                     color=RED, zorder=10))
    fig.text(0.012, 0.89, title, fontsize=18, fontweight="bold", ha="left")
    fig.text(0.012, 0.83, sub, fontsize=14, color=MUTED, ha="left")
    fig.text(0.012, 0.02, source, fontsize=11.5, color=MUTED, ha="left")


def hbars(ax, labels, values, colors, fmt, xmax):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.015, yi, fmt(v), va="center", fontsize=13.5, color=INK)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, xmax)
    style(ax)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="svg")
    plt.close(fig)


# 1. LightFC: latency vs accuracy for each precision/placement (OTB, first 300 frames, app chain).
pts = [  # label, ms per frame p50, success AUC, colour, label offset (points)
    ("CPU fp32", 101.18, 0.5812, GREY, (10, -14)),
    ("CPU int8 (deployed)", 61.93, 0.6253, RED, (10, 4)),
    ("GPU fp16", 359.98, 0.5807, BLUE, (-10, 14)),
    ("GPU backbone, CPU rest", 270.86, 0.5572, BLUE, (10, -18)),
]
fig, ax = plt.subplots(figsize=(7.2, 4.8))
fig.subplots_adjust(left=0.11, right=0.97, top=0.74, bottom=0.17)
for lab, ms, auc, col, (dx, dy) in pts:
    ax.scatter(ms, auc, s=110, color=col, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"{lab}\n{ms:.0f} ms, AUC {auc:.3f}", (ms, auc), xytext=(dx, dy), textcoords="offset points",
                fontsize=12.5, ha="left" if dx > 0 else "right", va="center")
ax.set_xlim(0, 400)
ax.set_ylim(0.50, 0.65)
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0.50, 0.55, 0.60, 0.65])
ax.set_xlabel("ms per frame (lower is better)", fontsize=13)
style(ax, grid_axis="both")
header(fig, "int8: fastest, no AUC loss",
       "LightFC, OTB success AUC (higher is better)",
       "Source: SCORECARD.md, OTB 8 sequences, first 300 frames, app chain on the board")
save(fig, "lightfc_tradeoff.svg")


print("wrote", sorted(os.listdir(OUT)))
