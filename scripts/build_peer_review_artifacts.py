#!/usr/bin/env python3
"""
Build reproducible manuscript tables from local project artifacts.

This script computes all table values from raw cohort data and experiment outputs.
It avoids hand-entered numbers so manuscript values can be regenerated exactly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.care_plan_generator import RetrospectiveEvaluator


METRICS = ["safety", "efficiency", "equity", "composite"]
RETRO_STRATEGY_ORDER = [
    "Control (Template)",
    "Safety Agent Only",
    "Efficiency Agent Only",
    "Equity Agent Only",
    "Multi-Agent (Nash)",
]


@dataclass
class ConditionSummary:
    n: int
    means: Dict[str, float]
    ci_lows: Dict[str, float]
    ci_highs: Dict[str, float]
    stds: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build peer-review manuscript tables from reproducible outputs."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Path to packaging/causal_orchestration_mhealth.",
    )
    parser.add_argument(
        "--cohort-path",
        type=str,
        default=None,
        help="Path to cohort CSV/CSV.GZ for Table 2 retrospective analysis. Defaults to data/real_cohort_experiment_eligible.csv.gz.",
    )
    parser.add_argument(
        "--real-cohort-path",
        type=str,
        default=None,
        help="Path to real cohort parquet for Table 1 demographics. If provided, Table 1 uses real demographics. Defaults to data/real_cohort_analytic.parquet if it exists.",
    )
    parser.add_argument(
        "--controlled-run-name",
        type=str,
        default="real_cohort_n200_seed42",
        help="Run name for controlled results to report in the manuscript draft.",
    )
    parser.add_argument(
        "--ongoing-run-name",
        type=str,
        default="pilot_peerready_n10_20260217",
        help="Second run name to include in the reproducibility status table.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated tables. Defaults to results/writeup_package.",
    )
    return parser.parse_args()


def ci95(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = values.astype(float)
    n = len(values)
    mean = float(np.mean(values))
    if n <= 1:
        return mean, 0.0, mean, mean
    std = float(np.std(values, ddof=1))
    half = 1.96 * std / np.sqrt(n)
    return mean, std, mean - half, mean + half


def summarize_condition_df(df: pd.DataFrame) -> ConditionSummary:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    ci_lows: Dict[str, float] = {}
    ci_highs: Dict[str, float] = {}
    for metric in METRICS:
        mean, std, low, high = ci95(df[metric].to_numpy())
        means[metric] = mean
        stds[metric] = std
        ci_lows[metric] = low
        ci_highs[metric] = high
    return ConditionSummary(
        n=int(len(df)),
        means=means,
        ci_lows=ci_lows,
        ci_highs=ci_highs,
        stds=stds,
    )


def format_ci(mean: float, low: float, high: float) -> str:
    return f"{mean:.3f} ({low:.3f}-{high:.3f})"


def build_table1_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Build Table 1 demographics.

    Accepts either:
    - Real cohort parquet (columns: age, female, race_ethnicity, state, sdoh_count,
      charlson_proxy, ed_visits_study_period, ip_admissions_study_period)
    - Legacy synthetic cohort CSV (columns: age, female, race, charlson, social_needs,
      baseline_risk, outcome)
    """
    n = len(df)
    age_mean = float(df["age"].mean())
    age_sd = float(df["age"].std(ddof=1))
    female_n = int(df["female"].sum())
    female_pct = 100.0 * female_n / n

    is_real = "race_ethnicity" in df.columns

    rows: List[Dict[str, str]] = [
        {"Characteristic": "Population size", "Value": f"{n:,}"},
        {"Characteristic": "Age, mean (SD), years", "Value": f"{age_mean:.1f} ({age_sd:.1f})"},
        {"Characteristic": "Female sex, n (%)", "Value": f"{female_n:,} ({female_pct:.1f}%)"},
    ]

    if is_real:
        # ---- Real cohort ----
        race_col = "race_ethnicity"
        race_order = [
            "Black/African American",
            "White",
            "Hispanic",
            "Asian",
            "American Indian or Alaska Native",
            "Native Hawaiian or Other Pacific Islander",
            "Other",
            "Unknown",
        ]
        race_counts = df[race_col].value_counts(dropna=False)
        for race in race_order:
            count = int(race_counts.get(race, 0))
            rows.append({
                "Characteristic": f"Race/ethnicity: {race}, n (%)",
                "Value": f"{count:,} ({100.0 * count / n:.1f}%)",
            })

        if "state" in df.columns:
            for state in ["VA", "WA"]:
                count = int((df["state"] == state).sum())
                rows.append({
                    "Characteristic": f"State: {state}, n (%)",
                    "Value": f"{count:,} ({100.0 * count / n:.1f}%)",
                })

        sdoh_count = df.get("sdoh_count", pd.Series(dtype=float))
        if len(sdoh_count) > 0:
            any_sdoh_n = int((sdoh_count > 0).sum())
            rows.append({
                "Characteristic": "Any SDOH care goal, n (%)",
                "Value": f"{any_sdoh_n:,} ({100.0 * any_sdoh_n / n:.1f}%)",
            })
            rows.append({
                "Characteristic": "SDOH goal count, mean (SD)",
                "Value": f"{float(sdoh_count.mean()):.2f} ({float(sdoh_count.std(ddof=1)):.2f})",
            })

        charlson_col = "charlson_proxy" if "charlson_proxy" in df.columns else None
        if charlson_col:
            cp = df[charlson_col]
            any_clin_n = int((cp > 0).sum())
            rows.append({
                "Characteristic": "Any clinical care goal, n (%)",
                "Value": f"{any_clin_n:,} ({100.0 * any_clin_n / n:.1f}%)",
            })

        ed_col = "ed_visits_study_period" if "ed_visits_study_period" in df.columns else None
        ip_col = "ip_admissions_study_period" if "ip_admissions_study_period" in df.columns else None
        if ed_col:
            ed = df[ed_col]
            rows.append({
                "Characteristic": "ED visits during study period, mean (SD)",
                "Value": f"{float(ed.mean()):.2f} ({float(ed.std(ddof=1)):.2f})",
            })
        if ip_col:
            ip = df[ip_col]
            rows.append({
                "Characteristic": "IP admissions during study period, mean (SD)",
                "Value": f"{float(ip.mean()):.2f} ({float(ip.std(ddof=1)):.2f})",
            })

    else:
        # ---- Legacy synthetic cohort ----
        baseline_risk_mean = float(df["baseline_risk"].mean())
        baseline_risk_sd = float(df["baseline_risk"].std(ddof=1))
        charlson_mean = float(df["charlson"].mean())
        charlson_sd = float(df["charlson"].std(ddof=1))
        social_needs_mean = float(df["social_needs"].mean())
        social_needs_sd = float(df["social_needs"].std(ddof=1))
        outcome_n = int(df["outcome"].sum())
        outcome_pct = 100.0 * outcome_n / n

        race_counts = df["race"].value_counts(dropna=False)
        for race, count in race_counts.items():
            rows.append({
                "Characteristic": f"Race: {race}, n (%)",
                "Value": f"{int(count):,} ({100.0 * float(count) / n:.1f}%)",
            })
        rows.extend([
            {"Characteristic": "Baseline risk score, mean (SD)",
             "Value": f"{baseline_risk_mean:.3f} ({baseline_risk_sd:.3f})"},
            {"Characteristic": "Charlson Comorbidity Index, mean (SD)",
             "Value": f"{charlson_mean:.2f} ({charlson_sd:.2f})"},
            {"Characteristic": "Social needs count, mean (SD)",
             "Value": f"{social_needs_mean:.2f} ({social_needs_sd:.2f})"},
            {"Characteristic": "Observed outcome event, n (%)",
             "Value": f"{outcome_n:,} ({outcome_pct:.1f}%)"},
        ])

    return pd.DataFrame(rows)


def build_table2_retrospective(df: pd.DataFrame) -> pd.DataFrame:
    strategies = {
        "Control (Template)": RetrospectiveEvaluator.evaluate_control_template(df),
        "Safety Agent Only": RetrospectiveEvaluator.evaluate_safety_agent(df),
        "Efficiency Agent Only": RetrospectiveEvaluator.evaluate_efficiency_agent(df),
        "Equity Agent Only": RetrospectiveEvaluator.evaluate_equity_agent(df),
        "Multi-Agent (Nash)": RetrospectiveEvaluator.evaluate_nash_orchestration(df),
    }

    rows: List[Dict[str, object]] = []
    for strategy in RETRO_STRATEGY_ORDER:
        safety, efficiency, equity = strategies[strategy]
        frame = pd.DataFrame(
            {
                "safety": safety,
                "efficiency": efficiency,
                "equity": equity,
            }
        )
        frame["composite"] = frame[["safety", "efficiency", "equity"]].mean(axis=1)
        summary = summarize_condition_df(frame)

        rows.append(
            {
                "Strategy": strategy,
                "N": summary.n,
                "Safety": format_ci(
                    summary.means["safety"],
                    summary.ci_lows["safety"],
                    summary.ci_highs["safety"],
                ),
                "Efficiency": format_ci(
                    summary.means["efficiency"],
                    summary.ci_lows["efficiency"],
                    summary.ci_highs["efficiency"],
                ),
                "Equity": format_ci(
                    summary.means["equity"],
                    summary.ci_lows["equity"],
                    summary.ci_highs["equity"],
                ),
                "Composite": format_ci(
                    summary.means["composite"],
                    summary.ci_lows["composite"],
                    summary.ci_highs["composite"],
                ),
            }
        )

    return pd.DataFrame(rows)


def build_controlled_tables(results_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(results_csv)

    # Deduplicate: experiment restarts may produce multiple rows per (patient_id, condition).
    # Take the mean of metric columns for each (patient_id, condition) pair.
    metric_cols = ["safety", "efficiency", "equity", "composite"]
    df = df.groupby(["patient_id", "condition"], as_index=False)[metric_cols].mean()

    nash = df[df["condition"] == "nash"][
        ["patient_id"] + metric_cols
    ].copy()
    baseline = df[df["condition"] == "compute_matched"][
        ["patient_id"] + metric_cols
    ].copy()
    merged = nash.merge(baseline, on="patient_id", suffixes=("_nash", "_baseline"))
    paired_ids = set(merged["patient_id"].astype(str))
    paired_df = df[df["patient_id"].astype(str).isin(paired_ids)].copy()

    by_condition_rows: List[Dict[str, object]] = []
    for condition, label in [("nash", "Nash Orchestration"), ("compute_matched", "Compute-Matched Baseline")]:
        cond_df = paired_df[paired_df["condition"] == condition]
        summary = summarize_condition_df(cond_df)
        by_condition_rows.append(
            {
                "Condition": label,
                "N_pairs": int(summary.n),
                "Safety": format_ci(
                    summary.means["safety"], summary.ci_lows["safety"], summary.ci_highs["safety"]
                ),
                "Efficiency": format_ci(
                    summary.means["efficiency"],
                    summary.ci_lows["efficiency"],
                    summary.ci_highs["efficiency"],
                ),
                "Equity": format_ci(
                    summary.means["equity"], summary.ci_lows["equity"], summary.ci_highs["equity"]
                ),
                "Composite": format_ci(
                    summary.means["composite"],
                    summary.ci_lows["composite"],
                    summary.ci_highs["composite"],
                ),
            }
        )

    paired_rows: List[Dict[str, object]] = []
    for metric in METRICS:
        diff = merged[f"{metric}_nash"].astype(float) - merged[f"{metric}_baseline"].astype(float)
        mean, std, low, high = ci95(diff.to_numpy())
        if len(diff) > 1:
            t_stat, p_val = stats.ttest_rel(
                merged[f"{metric}_nash"].astype(float),
                merged[f"{metric}_baseline"].astype(float),
            )
            cohens_d = float(mean / std) if std > 0 else float("nan")
        else:
            t_stat, p_val, cohens_d = float("nan"), float("nan"), float("nan")

        paired_rows.append(
            {
                "Metric": metric.capitalize(),
                "N_pairs": int(len(diff)),
                "Difference_Nash_minus_Baseline": mean,
                "Difference_95CI_low": low,
                "Difference_95CI_high": high,
                "paired_t": float(t_stat),
                "p_value": float(p_val),
                "cohens_d_paired": float(cohens_d),
            }
        )

    summary = {
        "n_results_rows": int(len(df)),
        "n_pairs": int(len(merged)),
        "condition_counts": {
            "nash": int(len(nash)),
            "compute_matched": int(len(baseline)),
        },
    }
    return pd.DataFrame(by_condition_rows), pd.DataFrame(paired_rows), summary


def run_status(results_dir: Path, run_name: str) -> Dict[str, object]:
    summary_path = results_dir / f"{run_name}_summary.json"
    config_path = results_dir / f"{run_name}_config.json"
    results_path = results_dir / f"{run_name}_results.csv"
    failures_path = results_dir / f"{run_name}_failures.csv"

    out: Dict[str, object] = {
        "run_name": run_name,
        "summary_exists": summary_path.exists(),
        "config_exists": config_path.exists(),
        "results_exists": results_path.exists(),
        "failures_exists": failures_path.exists(),
    }

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        out.update(
            {
                "status": "complete",
                "n_patients_requested": int(summary.get("n_patients_requested", 0)),
                "n_complete_pairs": int(summary.get("n_complete_pairs", 0)),
                "n_results_rows": int(summary.get("n_results_rows", 0)),
                "n_failures_logged": int(summary.get("n_failures_logged", 0)),
                "runtime_seconds": float(summary.get("runtime_seconds", float("nan"))),
                "models": summary.get("models", {}),
            }
        )
        return out

    if results_path.exists():
        df = pd.read_csv(results_path)
        nash_ids = set(df[df["condition"] == "nash"]["patient_id"].astype(str))
        baseline_ids = set(df[df["condition"] == "compute_matched"]["patient_id"].astype(str))
        pairs = nash_ids & baseline_ids
        failures = 0
        if failures_path.exists():
            try:
                failures = int(len(pd.read_csv(failures_path)))
            except Exception:  # noqa: BLE001
                failures = -1
        out.update(
            {
                "status": "running_or_partial",
                "n_patients_requested": np.nan,
                "n_complete_pairs": int(len(pairs)),
                "n_results_rows": int(len(df)),
                "n_failures_logged": failures,
                "runtime_seconds": np.nan,
                "models": {},
            }
        )
        return out

    out.update(
        {
            "status": "not_started",
            "n_patients_requested": np.nan,
            "n_complete_pairs": 0,
            "n_results_rows": 0,
            "n_failures_logged": 0,
            "runtime_seconds": np.nan,
            "models": {},
        }
    )
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    results_dir = project_root / "results"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (results_dir / "writeup_package")
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = Path(args.cohort_path).resolve() if args.cohort_path else (project_root / "data" / "real_cohort_experiment_eligible.csv.gz")
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")

    cohort_df = pd.read_csv(cohort_path)

    # Table 1: use real cohort parquet if available, else fall back to synthetic cohort
    repo_root = project_root.parent.parent  # waymark-local root
    default_real_cohort = repo_root / "data" / "real_cohort_analytic.parquet"
    real_cohort_path = Path(args.real_cohort_path).resolve() if args.real_cohort_path else default_real_cohort
    if real_cohort_path.exists():
        print(f"  Using real cohort for Table 1: {real_cohort_path}")
        table1_df = pd.read_parquet(real_cohort_path)
    else:
        print(f"  Real cohort not found at {real_cohort_path}; using synthetic cohort for Table 1.")
        table1_df = cohort_df

    table1 = build_table1_demographics(table1_df)
    table2 = build_table2_retrospective(cohort_df)

    controlled_results = results_dir / f"{args.controlled_run_name}_results.csv"
    if not controlled_results.exists():
        raise FileNotFoundError(
            f"Controlled run results not found: {controlled_results}. "
            "Provide --controlled-run-name for an available run."
        )
    table3, table4, controlled_summary = build_controlled_tables(controlled_results)

    run_rows = [
        run_status(results_dir, args.controlled_run_name),
        run_status(results_dir, args.ongoing_run_name),
    ]
    table5 = pd.DataFrame(run_rows)
    if "runtime_seconds" in table5.columns:
        table5["runtime_hours"] = table5["runtime_seconds"] / 3600.0

    table1_path = output_dir / "table1_demographics.csv"
    table2_path = output_dir / "table2_retrospective_performance.csv"
    table3_path = output_dir / "table3_controlled_condition_performance.csv"
    table4_path = output_dir / "table4_controlled_paired_differences.csv"
    table5_path = output_dir / "table5_reproducibility_status.csv"

    table1.to_csv(table1_path, index=False)
    table2.to_csv(table2_path, index=False)
    table3.to_csv(table3_path, index=False)
    table4.to_csv(table4_path, index=False)
    table5.to_csv(table5_path, index=False)

    manuscript_numbers = {
        "cohort_n": int(len(cohort_df)),
        "controlled_run_name": args.controlled_run_name,
        "ongoing_run_name": args.ongoing_run_name,
        "controlled_summary": controlled_summary,
        "table_paths": {
            "table1": str(table1_path),
            "table2": str(table2_path),
            "table3": str(table3_path),
            "table4": str(table4_path),
            "table5": str(table5_path),
        },
    }
    numbers_path = output_dir / "manuscript_numbers.json"
    numbers_path.write_text(json.dumps(manuscript_numbers, indent=2), encoding="utf-8")

    print("writeup package generated:")
    print(f"  {table1_path}")
    print(f"  {table2_path}")
    print(f"  {table3_path}")
    print(f"  {table4_path}")
    print(f"  {table5_path}")
    print(f"  {numbers_path}")


if __name__ == "__main__":
    main()
