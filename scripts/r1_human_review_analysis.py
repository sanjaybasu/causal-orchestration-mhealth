#!/usr/bin/env python3
"""
SAP v1 section 8: analyze completed blinded review workbooks.

  python scripts/r1_human_review_analysis.py --run r1_three_arm_n200 --dir <human_review dir>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.r1_rescore_judges import icc2_1, load_results  # noqa: E402

DOMAINS = ["safety", "efficiency", "equity"]


def icc2_k(x: np.ndarray) -> float:
    """ICC(2,k): reliability of the mean of k raters (Shrout and Fleiss 1979)."""
    n, k = x.shape
    grand = x.mean()
    msr = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (msc - mse) / n))


def load_ratings(folder: Path) -> pd.DataFrame:
    key = pd.read_csv(folder / "UNBLINDING_KEY_do_not_share.csv")
    key = key[key.block == "review"]
    rows = []
    for wb in sorted(folder.glob("care_plan_review_*.xlsx")):
        rater = wb.stem.split("_")[-1]
        r = pd.read_excel(wb, sheet_name="Review").merge(key, on="item")
        for slot in ("1", "2"):
            arm = r[f"plan_{slot}_arm"]
            rows.append(pd.DataFrame({
                "rater": rater, "item": r["item"], "patient_id": r["patient_id"], "arm": arm,
                **{d: (r[f"p{slot}_{d}"].astype(float) - 1) / 4 for d in DOMAINS},
                "unsafe": r[f"p{slot}_unsafe_or_wrong"].eq("Yes"),
                "usable": r[f"p{slot}_usable_minor_edits"].eq("Yes"),
                "preferred": r["preferred_plan"].eq(f"Plan {slot}"),
            }))
    h = pd.concat(rows, ignore_index=True)
    h["composite"] = h[DOMAINS].mean(1)
    missing = h[DOMAINS].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} plan ratings are incomplete")
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    h = load_ratings(Path(a.dir))
    res = {"n_raters": int(h.rater.nunique()), "n_patients": int(h.patient_id.nunique())}

    res["icc_humans"] = {}
    for m in DOMAINS + ["composite"]:
        x = h.pivot_table(index=["patient_id", "arm"], columns="rater", values=m).dropna().to_numpy()
        res["icc_humans"][m] = {"icc2_1": icc2_1(x), "icc2_k": icc2_k(x)}

    mean_h = h.groupby(["patient_id", "arm"])[DOMAINS + ["composite"]].mean()
    prim = load_results(a.run).assign(judge="deepseek-r1:8b")
    sec_path = ROOT / "results" / f"{a.run}_secondary_judges.csv"
    judges = pd.concat([prim, pd.read_csv(sec_path)] if sec_path.exists() else [prim], ignore_index=True)
    res["icc_human_vs_judge"] = {}
    for jdg, g in judges.groupby("judge"):
        j = g.set_index(["patient_id", "arm"])
        both = mean_h.join(j[DOMAINS + ["composite"]], rsuffix="_j", how="inner")
        res["icc_human_vs_judge"][jdg] = {
            m: icc2_1(both[[m, f"{m}_j"]].to_numpy()) for m in DOMAINS + ["composite"]}

    w = mean_h.reset_index().pivot(index="patient_id", columns="arm", values=DOMAINS + ["composite"])
    res["human_rated_N_minus_B"] = {}
    for m in DOMAINS + ["composite"]:
        diff = w[m]["N"] - w[m]["B"]
        n = len(diff)
        se = diff.std(ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        res["human_rated_N_minus_B"][m] = {"n": n, "mean_diff": float(diff.mean()),
                                           "ci_low": float(diff.mean() - tc * se), "ci_high": float(diff.mean() + tc * se),
                                           "p": float(stats.ttest_rel(w[m]["N"], w[m]["B"]).pvalue)}

    pref = h[h.arm == "N"]
    k, n = int(pref.preferred.sum()), len(pref)
    lo, hi = proportion_confint(k, n, method="wilson")
    res["prefer_N"] = {"k": k, "n": n, "pct": 100 * k / n, "ci_low": 100 * lo, "ci_high": 100 * hi}

    flag = h.groupby(["patient_id", "arm"]).unsafe.any().unstack()
    table = pd.crosstab(flag["N"], flag["B"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    res["unsafe_any_rater"] = {"N": int(flag["N"].sum()), "B": int(flag["B"].sum()), "n_patients": len(flag),
                               "mcnemar_p": float(mcnemar(table.to_numpy(), exact=True).pvalue)}
    res["usable_minor_edits_pct"] = h.groupby("arm").usable.mean().mul(100).to_dict()

    out = ROOT / "results" / "r1_analysis" / f"{a.run}_human_review.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
