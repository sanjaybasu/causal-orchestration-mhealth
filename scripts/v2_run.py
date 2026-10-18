#!/usr/bin/env python3
"""
Design v2 runner (development or final). One JSON line per (patient, variant) with every arm's
plan, selection-time coverage, and evaluator verdicts on all checklist items.

  python scripts/v2_run.py --set dev --run v2_dev1                  # development set
  python scripts/v2_run.py --set final --run v2_final               # the 200 patients
  python scripts/v2_run.py --set final --run v2_final --swap race   # counterfactual (Black <-> White)
  python scripts/v2_run.py --set final --run v2_final --swap sex
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_ollama_open_source_experiments import OllamaClient, build_patient_context, generate_initial_plan  # noqa: E402
from scripts.v2_pipeline import (ACTION_BUDGET, MODELS, VERIFIER_SETTINGS, WORD_CAP, checklist_for, confirm_pairs, coverage, flag_unsupported, gap_fill,  # noqa: E402
                                 load_reqmap, propose_actions, render, solve, verify_coverage)

RULES = ["nash", "sum", "maxmin"]
REFINE_ROUNDS = 2


def cohort() -> pd.DataFrame:
    c = pd.read_parquet(ROOT / "data" / "real_cohort_analytic.parquet")
    return c.rename(columns={"race_ethnicity": "race", "charlson_proxy": "charlson", "sdoh_count": "social_needs"})


def patient_set(which: str) -> pd.DataFrame:
    c = cohort().set_index("patient_id")
    final_ids = (ROOT / "data" / "r1_patient_ids.txt").read_text().split()
    if which == "final":
        return c.loc[final_ids].reset_index()
    holdout = (ROOT / "data" / "prompt_dev_exclusion.txt").read_text().split()
    pool = sorted(set(c.index) - set(holdout) - set(final_ids))
    extra = random.Random(7).sample(pool, 100)
    ids_path = ROOT / "data" / "v2_dev_ids.txt"
    if not ids_path.exists():
        ids_path.write_text("\n".join(holdout + extra) + "\n")
    return c.loc[ids_path.read_text().split()].reset_index()


def apply_swap(row: pd.Series, swap: str):
    row = row.copy()
    if swap == "race":
        r = str(row["race"])
        if "Black" in r:
            row["race"] = "White"
        elif r == "White":
            row["race"] = "Black/African American"
        else:
            return None  # no Black/White counterfactual defined
    elif swap == "sex":
        row["female"] = 1 - int(row["female"])
    return row


def refine_arm(client, ctx, visible):
    """R: verifier-guided self-refine. Missing visible items are fed back as text."""
    plan = generate_initial_plan(client, MODELS["renderer"], ctx)
    for _ in range(REFINE_ROUNDS):
        met = evaluate_with_verifier(client, visible, plan)
        missing = [it["requirement"] for it in visible if not met[it["id"]]]
        if not missing:
            break
        prompt = (f"{ctx}\nCurrent care plan:\n{plan}\n\nThe plan does not yet meet these requirements:\n"
                  + "\n".join(f"- {m}" for m in missing)
                  + f"\n\nRevise the plan to meet them with specific actions (who, what, when). Use at most {ACTION_BUDGET} "
                    f"action bullets and at most {WORD_CAP} words in total. Do not start, stop, or change medications.")
        plan = client.chat(model=MODELS["renderer"], user_prompt=prompt,
                           system_prompt="You are a senior care manager writing concise action plans.",
                           num_predict=650, temperature=0.2)
    return plan


def evaluate_with_verifier(client, items, plan):
    out = {}
    for it in items:
        prompt = (f"Care plan:\n{plan}\n\nRequirement: {it['question']}\nCounts as MET: {it['met']}\n"
                  f"Counts as NOT MET: {it['not_met']}\n\nAnswer ONLY yes or no.")
        raw = client.chat(model=MODELS["verifier"], user_prompt=prompt,
                          system_prompt="You are a strict clinical auditor.", temperature=0.0, **VERIFIER_SETTINGS)
        out[it["id"]] = raw.strip().lower().startswith("yes")
    return out


def run_patient(client, row, reqmap):
    ctx = build_patient_context(row)
    items = checklist_for(row, reqmap)
    visible = [it for it in items if not it["eval_only"]]
    calls = {"llm": 0}
    t0 = time.time()

    pool = []
    for role in ["safety", "efficiency", "equity"]:
        for a in propose_actions(client, role, ctx, visible):
            if a.lower() not in {p.lower() for p in pool}:
                pool.append(a)
    shortlist = verify_coverage(client, MODELS["verifier"], visible, pool, settings=VERIFIER_SETTINGS)
    cover = confirm_pairs(client, visible, pool, shortlist)
    missing = [it for it in visible if not cover.get(it["id"])]
    for it, a in zip(missing, gap_fill(client, ctx, missing)):
        pool.append(a)
        got = confirm_pairs(client, [it], pool, {it["id"]: [len(pool) - 1]})
        cover[it["id"]] = cover.get(it["id"], []) + got[it["id"]]
    excluded = flag_unsupported(client, ctx, pool)

    arms = {}
    budget = ACTION_BUDGET  # one cap for all arms; binds only when requirements exceed it
    for rule in RULES:
        chosen = solve(rule, visible, cover, len(pool), excluded, budget=budget)
        plan = render(client, ctx, [pool[a] for a in chosen])
        arms[f"S-{rule}"] = {"chosen": chosen, "selection_coverage": coverage(visible, cover, chosen), "plan": plan}
    arms["R"] = {"plan": refine_arm(client, ctx, visible)}
    arms["single"] = {"plan": generate_initial_plan(client, MODELS["renderer"], ctx)}

    for name, arm in arms.items():  # grading is done separately by the panel (scripts/v2_grader.py)
        arm["words"] = len(arm["plan"].split())
    return {"context": ctx, "budget": budget, "shortlist": shortlist, "n_gap_filled": len(missing), "items": [it["id"] for it in items],
            "visible": [it["id"] for it in visible], "pool": pool, "cover": cover, "excluded": excluded,
            "arms": arms, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "final"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--swap", choices=["none", "race", "sex"], default="none")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ollama-base-url", default="http://127.0.0.1:11435")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    out = ROOT / "results" / f"{a.run}_{a.swap}_shard{si}of{sn}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["patient_id"] for l in out.read_text().splitlines() if l.strip()}
    client = OllamaClient(base_url=a.ollama_base_url, timeout_seconds=300)
    reqmap = load_reqmap()
    pts = patient_set(a.set)
    if a.limit:
        pts = pts.head(a.limit)
    for idx, row in pts.iterrows():
        if idx % sn != si or str(row["patient_id"]) in done:
            continue
        r = row if a.swap == "none" else apply_swap(row, a.swap)
        if r is None:
            continue
        try:
            rec = run_patient(client, r, reqmap)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {idx}: {exc}", flush=True)
            continue
        rec.update(patient_id=str(row["patient_id"]), swap=a.swap)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush(); os.fsync(f.fileno())
        print(idx, rec["seconds"], {k: v.get("selection_coverage") or v["words"] for k, v in rec["arms"].items()}, flush=True)


if __name__ == "__main__":
    main()
