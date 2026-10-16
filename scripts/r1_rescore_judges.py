#!/usr/bin/env python3
"""
SAP v1 section 7: rescore every selected plan with secondary outcome judges and
report judge agreement.

  python scripts/r1_rescore_judges.py --run r1_three_arm_n200            # rescore (resumable)
  python scripts/r1_rescore_judges.py --run r1_three_arm_n200 --analyze  # ICC + per-judge effects
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_ollama_open_source_experiments import OllamaClient, build_patient_context, evaluate_plan  # noqa: E402
from scripts.run_r1_three_arm import load_patients  # noqa: E402

SECONDARY_JUDGES = ["gpt-oss:20b", "llama3.1:8b"]
# gpt-oss ignores think=false in Ollama 0.13.1; give it low reasoning effort and room to answer.
JUDGE_SETTINGS = {"gpt-oss:20b": {"num_predict": 1024, "think": "low"}, "llama3.1:8b": {}}
METRICS = ["safety", "efficiency", "equity", "composite"]


def load_results(run: str) -> pd.DataFrame:
    files = sorted((ROOT / "results").glob(f"{run}_shard*of*_results.csv")) or [ROOT / "results" / f"{run}_results.csv"]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert not d.duplicated(["patient_id", "arm"]).any()
    return d


def icc2_1(x: np.ndarray) -> float:
    """Shrout-Fleiss ICC(2,1): two-way random effects, absolute agreement, single rater. x: n targets x k raters."""
    n, k = x.shape
    grand = x.mean()
    msr = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def rescore(run: str, base_url: str) -> None:
    d = load_results(run)
    ctx = {str(r["patient_id"]): build_patient_context(r) for _, r in load_patients(200).iterrows()}
    out = ROOT / "results" / f"{run}_secondary_judges.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.patient_id, prev.arm, prev.judge))
    client = OllamaClient(base_url=base_url, timeout_seconds=300)
    for judge in SECONDARY_JUDGES:
        for _, r in d.iterrows():
            key = (r.patient_id, r.arm, judge)
            if key in done:
                continue
            s = evaluate_plan(client, judge, ctx[str(r.patient_id)], r.plan, judge_retries=4, **JUDGE_SETTINGS[judge])
            row = {"patient_id": r.patient_id, "arm": r.arm, "judge": judge,
                   **{m: s[m] for m in ["safety", "efficiency", "equity"]}}
            row["composite"] = float(np.mean([s[m] for m in ["safety", "efficiency", "equity"]]))
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(judge, r.patient_id[-6:], r.arm, round(row["composite"], 3), flush=True)


def analyze(run: str) -> None:
    prim = load_results(run)[["patient_id", "arm"] + METRICS].assign(judge="deepseek-r1:8b")
    sec = pd.read_csv(ROOT / "results" / f"{run}_secondary_judges.csv")
    allj = pd.concat([prim, sec], ignore_index=True)
    judges = ["deepseek-r1:8b"] + SECONDARY_JUDGES
    res = {"icc2_1": {}, "effect_by_judge": {}, "sign_agreement_NB_composite": {}}
    for m in METRICS:
        wide = allj.pivot_table(index=["patient_id", "arm"], columns="judge", values=m).dropna()
        res["icc2_1"][m] = {"all_three": icc2_1(wide[judges].to_numpy())}
        for i in range(3):
            for j in range(i + 1, 3):
                res["icc2_1"][m][f"{judges[i]} vs {judges[j]}"] = icc2_1(wide[[judges[i], judges[j]]].to_numpy())
    signs = {}
    for jdg in judges:
        w = allj[allj.judge == jdg].pivot(index="patient_id", columns="arm", values=METRICS).dropna()
        res["effect_by_judge"][jdg] = {}
        for m in METRICS:
            diff = w[m]["N"] - w[m]["B"]
            t, p = stats.ttest_rel(w[m]["N"], w[m]["B"])
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            tc = stats.t.ppf(0.975, len(diff) - 1)
            res["effect_by_judge"][jdg][m] = {"n": len(diff), "mean_diff": float(diff.mean()),
                                              "ci_low": float(diff.mean() - tc * se), "ci_high": float(diff.mean() + tc * se),
                                              "p": float(p)}
        signs[jdg] = np.sign((w["composite"]["N"] - w["composite"]["B"]).round(3))
    s = pd.DataFrame(signs).dropna()
    for i in range(3):
        for j in range(i + 1, 3):
            res["sign_agreement_NB_composite"][f"{judges[i]} vs {judges[j]}"] = float((s[judges[i]] == s[judges[j]]).mean())
    out = ROOT / "results" / "r1_analysis" / f"{run}_judge_agreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--ollama-base-url", default="http://127.0.0.1:11435")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.run)
    else:
        rescore(a.run, a.ollama_base_url)


if __name__ == "__main__":
    main()
