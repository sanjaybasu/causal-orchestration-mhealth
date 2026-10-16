#!/usr/bin/env python3
"""
R1 figures as PLOS-compliant TIFF (RGB, LZW, 300 dpi, width <= 7.5 in) plus PNG previews.

  python scripts/r1_figures.py --run r1_three_arm_n200 --out <dir>
  python scripts/r1_figures.py --feb --out <dir>      # preview on the February run
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "r1_analysis"
METRICS = ["safety", "efficiency", "equity", "composite"]
LABELS = {"safety": "Safety", "efficiency": "Efficiency", "equity": "Equity", "composite": "Composite"}
INK, MUTED, ACCENT = "#1f2933", "#7b8794", "#2f6fbd"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def save(fig, out: Path, name: str) -> None:
    png = out / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    Image.open(png).convert("RGB").save(out / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))


def fig1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec=INK, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.25)

    def arrow(x1, x2, y=34):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK))

    boxes = [(0.5, 16, "Patient context\nage, sex, race and\nethnicity, conditions,\nsocial needs"),
             (21, 14, "Draft plan\n(generator)"),
             (39.5, 20, "3 rounds of 3 critiques\nand 1 synthesis;\nround 3 writes\n3 candidate plans"),
             (64, 16, "Selection judge\nscores candidates;\nselection rule\npicks one plan"),
             (84, 15.5, "Outcome judges\nand blinded\nhuman review")]
    for x, w, t in boxes:
        box(x, 28, w, 12, t)
    for (x1, w1, _), (x2, _, _) in zip(boxes, boxes[1:]):
        arrow(x1 + w1 + 0.8, x2 - 0.8)

    cols = [(0.5, "Arm"), (30, "Critiques"), (72, "Selection rule")]
    rows = [("Role-specialized Nash", "Safety, Efficiency, and Equity agents", "Nash product (geometric mean)"),
            ("Generic self-critique", "One model, generic prompt, 3 critiques", "Nash product (geometric mean)"),
            ("Role-specialized additive", "Safety, Efficiency, and Equity agents", "Arithmetic mean")]
    for x, h in cols:
        ax.text(x, 21, h, fontsize=7, weight="bold", color=INK)
    for i, r in enumerate(rows):
        for (x, _), t in zip(cols, r):
            ax.text(x, 16 - 4.5 * i, t, fontsize=7, color=INK)
    ax.text(0.5, 1, "Each arm: 15 generation calls, 4 selection-judge calls, and 1 outcome-judge call per patient.",
            fontsize=6.8, color=MUTED)
    save(fig, out, "Fig1")


def fig2(diff: pd.DataFrame, out: Path, title_arms: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    for ax, m in zip(axes, METRICS):
        d = diff[m].to_numpy()
        lim = max(0.05, np.nanmax(np.abs(d)))
        bins = np.linspace(-lim, lim, 25)
        ax.hist(d, bins=bins, color=ACCENT, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvline(np.mean(d), color=INK, lw=0.8, ls="--")
        up, same, down = (d >= 0.005).sum(), (np.abs(d) < 0.005).sum(), (d <= -0.005).sum()
        ax.set_title(f"{LABELS[m]}\nhigher {up}, same {same},\nlower {down}", fontsize=7, color=INK)
        ax.set_xlabel(f"Difference ({title_arms})")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Patients")
    save(fig, out, "Fig2")


def fig3(sub: dict, out: Path, title_arms: str) -> None:
    order = [("race3", "Race and ethnicity", ["Black", "White", "Other"]),
             ("state", "State", ["VA", "WA"]),
             ("any_social_need", "Any social need", ["Yes", "No"]),
             ("social_count3", "Social-needs count", ["0", "1", ">=2"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.6), sharey=True)
    for ax, m in zip(axes, ["equity", "composite"]):
        y, ticks, labels = 0, [], []
        for var, name, levels in order:
            y -= 1
            ticks.append(y)
            labels.append(name)
            p = sub[var]["interaction"][m]["p"]
            ax.text(1.0, y, f"p = {p:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.5, color=MUTED)
            for lvl in levels:
                if lvl not in sub[var]["levels"]:
                    continue
                y -= 1
                r = sub[var]["levels"][lvl][m]
                ax.errorbar(r["mean_diff"], y, xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
                            fmt="o", ms=3, color=ACCENT, ecolor=INK, elinewidth=0.8, capsize=2)
                ticks.append(y)
                labels.append(f"{lvl.replace('>=', '≥')} (n = {r['n']})")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(LABELS[m], fontsize=8)
        ax.set_xlabel(f"Mean difference, 95% CI\n({title_arms})", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels, fontsize=7)
    for t, lab in zip(axes[0].get_yticklabels(), labels):
        if "(n =" not in lab:
            t.set_fontweight("bold")
    fig.tight_layout()
    save(fig, out, "Fig3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--feb", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    if a.feb:
        diff = pd.read_csv(A / "feb_paired_differences.csv")
        sub = json.loads((A / "feb_paired_analyses.json").read_text())["mean"]["subgroups"]
        arms = "two-arm run"
    else:
        diff = pd.read_csv(A / f"{a.run}_NB_paired_differences.csv")
        sub = json.loads((A / f"{a.run}_paired_analyses.json").read_text())["N-B"]["subgroups"]
        arms = "Nash − self-critique"
    fig2(diff, out, arms)
    fig3(sub, out, arms)


if __name__ == "__main__":
    main()
