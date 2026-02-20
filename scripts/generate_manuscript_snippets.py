#!/usr/bin/env python3
"""
Generate manuscript-ready table rows and results text from experiment summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_metric(summary: dict, metric: str) -> str:
    mean = summary[f"{metric}_mean"]
    low = summary[f"{metric}_ci_low"]
    high = summary[f"{metric}_ci_high"]
    return f"{mean:.3f} ({low:.3f}-{high:.3f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manuscript snippet from run summary JSON.")
    parser.add_argument("--summary", required=True, help="Path to *_summary.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path. Defaults to alongside summary with *_manuscript_snippet.md",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    payload = json.loads(summary_path.read_text())

    nash = payload["nash_summary_paired"]
    base = payload["compute_matched_summary_paired"]
    comp_stats = payload["paired_statistics"]["composite"]

    n_pairs = int(comp_stats["n_pairs"])
    diff_pp = comp_stats["difference"] * 100

    nash_row = (
        f"| Nash Multi-Model† | {fmt_metric(nash, 'safety')} | {fmt_metric(nash, 'efficiency')} | "
        f"{fmt_metric(nash, 'equity')} | **{fmt_metric(nash, 'composite')}** | {n_pairs} | Real |"
    )
    base_row = (
        f"| Compute-Matched‡ | {fmt_metric(base, 'safety')} | {fmt_metric(base, 'efficiency')} | "
        f"{fmt_metric(base, 'equity')} | **{fmt_metric(base, 'composite')}** | {n_pairs} | Real |"
    )
    advantage_row = (
        f"| *Nash Advantage* | *{(nash['safety_mean']-base['safety_mean'])*100:+.1f}pp* | "
        f"*{(nash['efficiency_mean']-base['efficiency_mean'])*100:+.1f}pp* | "
        f"*{(nash['equity_mean']-base['equity_mean'])*100:+.1f}pp* | "
        f"***{diff_pp:+.1f}pp*** | | *paired t={comp_stats['t_statistic']:.2f}, p={comp_stats['p_value']:.4g}* |"
    )

    paragraph = (
        f"Nash Orchestration achieved Composite Score {nash['composite_mean']:.3f} "
        f"(95% CI {nash['composite_ci_low']:.3f}-{nash['composite_ci_high']:.3f}) versus "
        f"Compute-Matched Self-Critique {base['composite_mean']:.3f} "
        f"(95% CI {base['composite_ci_low']:.3f}-{base['composite_ci_high']:.3f}), "
        f"a {diff_pp:+.1f} percentage point difference in paired analysis "
        f"(N={n_pairs}, t={comp_stats['t_statistic']:.2f}, p={comp_stats['p_value']:.4g})."
    )

    md = [
        "# Manuscript Snippet",
        "",
        "## Table 1 Rows",
        "",
        nash_row,
        base_row,
        advantage_row,
        "",
        "## Results Paragraph",
        "",
        paragraph,
        "",
        "## Reproducibility Metadata",
        "",
        f"- run_name: `{payload['run_name']}`",
        f"- runtime_seconds: `{payload.get('runtime_seconds')}`",
        f"- n_complete_pairs: `{payload.get('n_complete_pairs')}`",
        f"- n_failures_logged: `{payload.get('n_failures_logged')}`",
        f"- models: `{json.dumps(payload.get('models', {}), sort_keys=True)}`",
    ]

    output_path = Path(args.output) if args.output else summary_path.with_name(
        summary_path.name.replace("_summary.json", "_manuscript_snippet.md")
    )
    output_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

