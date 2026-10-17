#!/usr/bin/env python3
"""
SAP v1 Addendum 3/3a: single-pass reference. For each of the 200 patients, write one new
draft with the generator's draft prompt and score it with the primary outcome judge; then
report Nash final plan minus single-pass draft. Descriptive; not compute-matched.

  python scripts/r1_single_pass.py --run r1_three_arm_n200            # generate + score (resumable)
  python scripts/r1_single_pass.py --run r1_three_arm_n200 --analyze
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

from run_ollama_open_source_experiments import OllamaClient, build_patient_context, evaluate_plan, generate_initial_plan  # noqa: E402
from scripts.r1_rescore_judges import load_results  # noqa: E402
from scripts.run_r1_three_arm import MODELS, load_patients  # noqa: E402

METRICS = ["safety", "efficiency", "equity", "composite"]


def generate(base_url: str, out: Path) -> None:
    done = set(pd.read_csv(out).patient_id) if out.exists() else set()
    client = OllamaClient(base_url=base_url, timeout_seconds=240)
    for _, row in load_patients(200).iterrows():
        pid = str(row["patient_id"])
        if pid in done:
            continue
        ctx = build_patient_context(row)
        draft = generate_initial_plan(client, MODELS["generator"], ctx)
        s = evaluate_plan(client, MODELS["outcome_judge"], ctx, draft, judge_retries=4)
        rec = {"patient_id": pid, **{k: s[k] for k in ["safety", "efficiency", "equity"]}, "plan": draft}
        rec["composite"] = float(np.mean([s[k] for k in ["safety", "efficiency", "equity"]]))
        pd.DataFrame([rec]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(pid[-6:], round(rec["composite"], 3), flush=True)


def analyze(run: str, out: Path) -> None:
    sp = pd.read_csv(out).set_index("patient_id")
    assert not sp.index.duplicated().any()
    n = load_results(run)
    n = n[n.arm == "N"].set_index("patient_id")
    ids = n.index.intersection(sp.index)
    res = {"n": len(ids)}
    for m in METRICS:
        d = n.loc[ids, m] - sp.loc[ids, m]
        se = d.std(ddof=1) / np.sqrt(len(d))
        tc = stats.t.ppf(0.975, len(d) - 1)
        res[m] = {"mean_diff": float(d.mean()), "ci_low": float(d.mean() - tc * se),
                  "ci_high": float(d.mean() + tc * se), "p": float(stats.ttest_rel(n.loc[ids, m], sp.loc[ids, m]).pvalue),
                  "single_pass_mean": float(sp.loc[ids, m].mean())}
    path = ROOT / "results" / "r1_analysis" / f"{run}_single_pass.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--ollama-base-url", default="http://127.0.0.1:11435")
    a = ap.parse_args()
    out = ROOT / "results" / f"{a.run}_single_pass.csv"
    analyze(a.run, out) if a.analyze else generate(a.ollama_base_url, out)


if __name__ == "__main__":
    main()
