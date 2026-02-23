#!/usr/bin/env python3
"""
Generate manuscript figures from current writeup artifacts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    _scripts_dir = Path(__file__).resolve().parent
    _project_root = _scripts_dir.parent
    parser = argparse.ArgumentParser(
        description="Generate Figure1 and Figure2 for submission package."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(_project_root),
        help="Path to causal_orchestration_mhealth project root.",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default=str(_project_root / "results" / "figures"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--package-dir",
        type=str,
        default=str(_project_root / "results" / "figures"),
        help="Copy destination for generated figures (defaults to same as --submission-dir).",
    )
    return parser.parse_args()


def parse_mean(cell: str) -> float:
    match = re.match(r"\s*([0-9]*\.?[0-9]+)\s*\(", str(cell))
    if not match:
        raise ValueError(f"Could not parse mean from cell: {cell}")
    return float(match.group(1))


def generate_figure1(output_path: Path) -> None:
    # Vertical PRISMA-style flowchart.
    # Boxes are stacked top-to-bottom in a single column with a side branch
    # for the judge scoring step, matching the Nash workflow description.
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1)
    ax.axis("off")

    BOX_W = 0.54
    BOX_H = 0.09
    CX = 0.50  # center-x for main column

    # Main-column box y-centers, top to bottom
    y_positions = [0.92, 0.77, 0.62, 0.47, 0.32]
    labels = [
        "Patient Context\nand Cohort Features",
        "Initial Draft Generation",
        "Safety Agent Critique",
        "Efficiency Agent Critique",
        "Equity Agent Critique",
    ]
    # Synthesis and judge boxes
    # Equity bottom = 0.32 - 0.045 = 0.275; synthesis top = 0.17 + 0.0425 = 0.2125
    # gap = 0.0625 — clear separation
    synth_y = 0.17
    judge_y = 0.02

    arrow_kw = dict(arrowstyle="-|>", lw=1.6, color="#1f2a44",
                    mutation_scale=14)

    def draw_box(cx: float, cy: float, w: float, h: float, text: str,
                 bold: bool = False) -> None:
        rect = plt.Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            facecolor="#f4f6f8", edgecolor="#1f2a44", linewidth=1.5
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=9.5, weight=weight, wrap=True)

    # Draw main-column boxes
    for cy, label in zip(y_positions, labels):
        draw_box(CX, cy, BOX_W, BOX_H, label)

    # Down-arrows in main column
    for i in range(len(y_positions) - 1):
        y_top = y_positions[i] - BOX_H / 2
        y_bot = y_positions[i + 1] + BOX_H / 2
        ax.annotate("", xy=(CX, y_bot), xytext=(CX, y_top),
                    arrowprops=arrow_kw)

    # Synthesis box (full width, bold)
    synth_w = 0.62
    synth_h = 0.085
    draw_box(CX, synth_y, synth_w, synth_h,
             "Nash Bargaining Synthesis\n(Final Plan)", bold=True)

    # Arrow from last critique to synthesis
    ax.annotate("", xy=(CX, synth_y + synth_h / 2),
                xytext=(CX, y_positions[-1] - BOX_H / 2),
                arrowprops=arrow_kw)

    # Judge box
    judge_h = 0.075
    judge_w = 0.70
    judge_cy = judge_y + judge_h / 2
    draw_box(CX, judge_cy, judge_w, judge_h,
             "Judge Scoring\nSafety | Efficiency | Equity | Composite  (0-1 scale)")

    # Arrow from synthesis to judge (tail = synthesis bottom, head = judge top)
    judge_cy = judge_y + judge_h / 2
    ax.annotate("", xy=(CX, judge_cy + judge_h / 2),
                xytext=(CX, synth_y - synth_h / 2),
                arrowprops=arrow_kw)

    # Baseline bracket: dashed box around critique steps to label comparator
    brace_x0 = CX - BOX_W / 2 - 0.04
    brace_y0 = y_positions[-1] - BOX_H / 2 - 0.01
    brace_y1 = y_positions[2] + BOX_H / 2 + 0.01
    brace_h = brace_y1 - brace_y0
    rect_b = plt.Rectangle(
        (brace_x0, brace_y0), BOX_W + 0.08, brace_h,
        facecolor="none", edgecolor="#888888", linewidth=1.0,
        linestyle="dashed"
    )
    ax.add_patch(rect_b)
    ax.text(brace_x0 - 0.01, brace_y0 + brace_h / 2,
            "Role-specialized\ncritique agents\n(Nash condition)\n\nor\n\nSequential self-\ncritique\n(baseline condition)",
            ha="right", va="center", fontsize=7.5, color="#555555",
            style="italic")

    ax.set_title("Figure 1. Nash-Orchestrated Generation Workflow",
                 fontsize=12, pad=10)
    ax.text(0.50, -0.01,
            "Output metrics scored on a 0 to 1 scale; higher values indicate higher judged quality.",
            ha="center", va="top", fontsize=8, color="#444444",
            transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_radar_data(table2: pd.DataFrame) -> Dict[str, List[float]]:
    metrics = ["Safety", "Efficiency", "Equity", "Composite"]
    out: Dict[str, List[float]] = {}
    for _, row in table2.iterrows():
        strategy = str(row["Strategy"])
        out[strategy] = [parse_mean(row[m]) for m in metrics]
    return out


def generate_figure2(table2_path: Path, output_path: Path) -> None:
    table2 = pd.read_csv(table2_path)
    radar_data = build_radar_data(table2)

    categories = ["Safety", "Efficiency", "Equity", "Composite"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    strategy_order = [
        "Control (Template)",
        "Safety Agent Only",
        "Efficiency Agent Only",
        "Equity Agent Only",
        "Multi-Agent (Nash)",
    ]
    colors = {
        "Control (Template)": "#7f8c8d",
        "Safety Agent Only": "#c0392b",
        "Efficiency Agent Only": "#2471a3",
        "Equity Agent Only": "#1e8449",
        "Multi-Agent (Nash)": "#6c3483",
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for strategy in strategy_order:
        values = radar_data[strategy][:]
        values += values[:1]
        ax.plot(angles, values, color=colors[strategy], linewidth=2, label=strategy)
        ax.fill(angles, values, color=colors[strategy], alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.set_title(
        "Figure 2. Strategy Performance in Safety–Efficiency–Equity–Composite Space",
        pad=30, fontsize=12
    )
    # Place legend below the radar plot to avoid overlap with axis labels.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_into_package(fig1_path: Path, fig2_path: Path, package_dir: Path) -> None:
    pkg_fig_dir = package_dir / "figures"
    pkg_fig_dir.mkdir(parents=True, exist_ok=True)
    dest1 = pkg_fig_dir / "Figure1.png"
    dest2 = pkg_fig_dir / "Figure2.png"
    if dest1.resolve() != fig1_path.resolve():
        dest1.write_bytes(fig1_path.read_bytes())
    if dest2.resolve() != fig2_path.resolve():
        dest2.write_bytes(fig2_path.read_bytes())


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    submission_dir = Path(args.submission_dir).resolve()
    package_dir = Path(args.package_dir).resolve()

    table2_path = project_root / "results" / "writeup_package" / "table2_retrospective_performance.csv"
    if not table2_path.exists():
        raise FileNotFoundError(f"Missing table file: {table2_path}")

    submission_dir.mkdir(parents=True, exist_ok=True)
    fig1_path = submission_dir / "Figure1.png"
    fig2_path = submission_dir / "Figure2.png"

    generate_figure1(fig1_path)
    generate_figure2(table2_path, fig2_path)
    if package_dir.resolve() != submission_dir.resolve():
        copy_into_package(fig1_path, fig2_path, package_dir)
        print(f"wrote {fig1_path}")
        print(f"wrote {fig2_path}")
        print(f"wrote {package_dir / 'figures' / 'Figure1.png'}")
        print(f"wrote {package_dir / 'figures' / 'Figure2.png'}")
    else:
        print(f"wrote {fig1_path}")
        print(f"wrote {fig2_path}")


if __name__ == "__main__":
    main()
