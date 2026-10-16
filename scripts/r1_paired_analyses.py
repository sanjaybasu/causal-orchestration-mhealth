#!/usr/bin/env python3
"""
Paired analyses for SAP v1 sections 5, 6 and 9: win/tie/loss, Wilcoxon, subgroup
differences with interaction tests. Works on the February two-arm run and on the
three-arm run.

  python scripts/r1_paired_analyses.py --feb                                  # February run
  python scripts/r1_paired_analyses.py --three-arm r1_three_arm_n200          # three-arm run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r1_analysis"
METRICS = ["safety", "efficiency", "equity", "composite"]
TIE = 0.005


def paired_block(x: pd.Series, y: pd.Series) -> dict:
    d = (x - y).astype(float)
    n = len(d)
    t, p = stats.ttest_rel(x, y)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    try:
        w_p = float(stats.wilcoxon(d[d.abs() >= 1e-12]).pvalue)
    except ValueError:
        w_p = float("nan")
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "ci_low": float(d.mean() - tcrit * se),
        "ci_high": float(d.mean() + tcrit * se),
        "t": float(t), "p": float(p),
        "cohens_d": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
        "wilcoxon_p": w_p,
        "improved": int((d >= TIE).sum()),
        "tied": int((d.abs() < TIE).sum()),
        "worse": int((d <= -TIE).sum()),
        "median": float(d.median()),
        "p25": float(d.quantile(0.25)),
        "p75": float(d.quantile(0.75)),
        "p05": float(d.quantile(0.05)),
        "p95": float(d.quantile(0.95)),
    }


def subgroup_vars(ids: pd.Index) -> pd.DataFrame:
    c = pd.read_parquet(ROOT / "data" / "real_cohort_analytic.parquet").set_index("patient_id").loc[ids]
    race = c["race_ethnicity"].astype(str)
    out = pd.DataFrame(index=c.index)
    out["race3"] = np.where(race.str.contains("Black"), "Black", np.where(race.str.contains("White"), "White", "Other"))
    out["state"] = c["state"].astype(str)
    out["any_social_need"] = np.where(c["sdoh_count"] >= 1, "Yes", "No")
    out["social_count3"] = pd.cut(c["sdoh_count"], [-1, 0, 1, 99], labels=["0", "1", ">=2"]).astype(str)
    return out


def subgroup_table(diff: pd.DataFrame, sub: pd.DataFrame) -> dict:
    res = {}
    df = diff.join(sub)
    for var in sub.columns:
        res[var] = {"levels": {}, "interaction": {}}
        for lvl, g in df.groupby(var):
            res[var]["levels"][lvl] = {}
            for m in ["equity", "composite"]:
                v = g[m]
                n = len(v)
                se = v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
                tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
                res[var]["levels"][lvl][m] = {"n": n, "mean_diff": float(v.mean()),
                                              "ci_low": float(v.mean() - tcrit * se),
                                              "ci_high": float(v.mean() + tcrit * se)}
        for m in ["equity", "composite"]:
            fit = smf.ols(f"{m} ~ C({var})", data=df).fit()
            res[var]["interaction"][m] = {"F": float(fit.fvalue), "p": float(fit.f_pvalue), "df_num": float(fit.df_model)}
    return res


def feb() -> dict:
    d = pd.read_csv(ROOT / "results" / "real_cohort_n200_seed42_results.csv")
    out = {"rows": len(d), "patients_with_repeat_generations": int(
        (d.groupby(["patient_id", "condition"]).size() > 1).groupby(level=0).any().sum())}
    for rule in ["mean", "first"]:
        if rule == "mean":
            x = d.groupby(["patient_id", "condition"])[METRICS].mean().reset_index()
        else:
            x = d.drop_duplicates(["patient_id", "condition"], keep="first")
        w = x.pivot(index="patient_id", columns="condition", values=METRICS)
        out[rule] = {m: paired_block(w[m]["nash"], w[m]["compute_matched"]) for m in METRICS}
        if rule == "mean":
            diff = pd.DataFrame({m: w[m]["nash"] - w[m]["compute_matched"] for m in METRICS})
            out["mean"]["subgroups"] = subgroup_table(diff, subgroup_vars(diff.index))
            diff.reset_index().to_csv(OUT / "feb_paired_differences.csv", index=False)
    nash = d[d.condition == "nash"]
    out["judge_resolution"] = {
        "nash_composite_eq_0.75": int((nash.composite.round(3) == 0.75).sum()),
        "nash_rows": len(nash),
        "distinct_composite_values": int(d.composite.round(3).nunique()),
    }
    return out


def three_arm(run: str) -> dict:
    files = sorted((ROOT / "results").glob(f"{run}_shard*of*_results.csv")) or [ROOT / "results" / f"{run}_results.csv"]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert not d.duplicated(["patient_id", "arm"]).any()
    w = d.pivot(index="patient_id", columns="arm", values=METRICS).dropna()
    out = {"n_complete_triplets": len(w),
           "calls_identical_across_arms": bool(d.groupby("arm")[["n_generation_calls", "n_selection_judge_calls",
                                                                  "n_outcome_judge_calls"]].nunique().eq(1).all().all()
                                               and d[["n_generation_calls", "n_selection_judge_calls",
                                                      "n_outcome_judge_calls"]].nunique().eq(1).all()),
           "below_disagreement_by_arm": d.groupby("arm")["below_disagreement"].mean().to_dict()}
    for a, b in [("N", "B"), ("N", "A")]:
        out[f"{a}-{b}"] = {m: paired_block(w[m][a], w[m][b]) for m in METRICS}
    secondary_p = [out["N-B"][m]["p"] for m in ["safety", "efficiency", "equity"]] + \
                  [out["N-A"][m]["p"] for m in METRICS]
    order = np.argsort(secondary_p)
    holm = np.empty(len(secondary_p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(secondary_p) - rank) * secondary_p[i]))
        holm[i] = running
    out["holm_secondary"] = dict(zip(["N-B safety", "N-B efficiency", "N-B equity",
                                      "N-A safety", "N-A efficiency", "N-A equity", "N-A composite"],
                                     map(float, holm)))
    diff = pd.DataFrame({m: w[m]["N"] - w[m]["B"] for m in METRICS})
    out["N-B"]["subgroups"] = subgroup_table(diff, subgroup_vars(diff.index))
    diff.reset_index().to_csv(OUT / f"{run}_NB_paired_differences.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feb", action="store_true")
    ap.add_argument("--three-arm")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.feb:
        (OUT / "feb_paired_analyses.json").write_text(json.dumps(feb(), indent=2))
    if args.three_arm:
        (OUT / f"{args.three_arm}_paired_analyses.json").write_text(json.dumps(three_arm(args.three_arm), indent=2))


if __name__ == "__main__":
    main()
