"""Plot the README hero figure from the raw SSL results in results/.

Usage:  python scripts/plot_readme_hero.py
Output: docs/figures/ssl_hero.png and docs/figures/ssl_hero.svg

One row per pseudo-labelling arm; each dot is one seed of the YOLOv8n student
(test split, mAP@50:95); the short bar is the mean and the line spans min-max.
Only the default label split is used (files without "_split").
"""
import glob
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")

ARMS = [  # (arm key, row label, colour)
    ("supervised", "Supervised only (10% labels)", "#4D4D4D"),
    ("self", "+ pseudo-labels from YOLOv8n", "#8C6BB1"),
    ("yolo_large", "+ pseudo-labels from YOLOv8m", "#2B8CBE"),
    ("vfm", "+ pseudo-labels from RF-DETR (DINOv2)", "#D95F0E"),
]


def load(arm):
    vals = []
    for f in sorted(glob.glob(os.path.join(RES, f"DsPCBSD+_ssl_{arm}_frac0.1_ep60_seed*.json"))):
        d = json.load(open(f))
        if "mAP50_95" in d:
            vals.append(d["mAP50_95"])
    return vals


def main():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": "black",
        "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    })
    fig, ax = plt.subplots(figsize=(6.4, 2.1), dpi=200)
    fig.patch.set_facecolor("white")
    for i, (arm, label, col) in enumerate(ARMS):
        y = len(ARMS) - 1 - i
        v = load(arm)
        if not v:
            continue
        ax.scatter(v, [y] * len(v), s=22, color=col, alpha=0.85, zorder=3, linewidths=0)
        m = statistics.mean(v)
        if len(v) > 1:
            ax.plot([min(v), max(v)], [y, y], color=col, lw=1.0, zorder=2)
        ax.plot([m, m], [y - 0.22, y + 0.22], color=col, lw=2.0, zorder=4)
        ax.text(max(v) + 0.004, y, f"{m:.3f}" + ("" if len(v) > 1 else " (1 seed)"),
                va="center", ha="left", fontsize=8.5)
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([a[1] for a in ARMS][::-1])
    ax.set_xlabel("YOLOv8n student mAP@50:95 on DsPCBSD+ (test split)")
    ax.set_xlim(0.34, 0.455)
    ax.set_ylim(-0.6, len(ARMS) - 0.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "ssl_hero.png"), facecolor="white")
    fig.savefig(os.path.join(OUT, "ssl_hero.svg"), facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
