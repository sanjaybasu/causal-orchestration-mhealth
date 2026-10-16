#!/usr/bin/env python3
"""
Three-arm compute-matched paired experiment (SAP v1, notebooks/.../revision_1/SAP_v1.md).

Arms (identical call counts: 15 generation, 4 selection-judge, 1 outcome-judge):
  N  role-specialized critiques, Nash-product selection over K candidates
  B  generic single-model critiques, Nash-product selection over K candidates
  A  role-specialized critiques, arithmetic-mean selection over K candidates

Usage:
  python scripts/run_r1_three_arm.py --run-name r1_three_arm_n200            # full run
  python scripts/run_r1_three_arm.py --run-name r1_smoke --n-patients 2      # smoke test
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_ollama_open_source_experiments import (  # noqa: E402
    ROLE_ORDER,
    ROLE_SYSTEM_PROMPTS,
    OllamaClient,
    build_patient_context,
    evaluate_plan,
    generate_initial_plan,
    generate_role_critique,
)

MODELS = {
    "generator": "llama3.1:8b",
    "safety": "deepseek-r1:8b",
    "efficiency": "llama3.1:8b",
    "equity": "llama3.1:8b",
    "generic_critic": "llama3.1:8b",
    "selection_judge": "llama3.1:8b",
    "outcome_judge": "deepseek-r1:8b",
}
ARMS = ["N", "B", "A"]
N_ROUNDS = 3
K_CANDIDATES = 3
CANDIDATE_TEMPERATURE = 0.7
FLOOR = 0.01
DOMAINS = ["safety", "efficiency", "equity"]

GENERIC_CRITIC_PROMPT = (
    "You are a care plan reviewer. Identify the most important weaknesses in the draft."
)

SYNTHESIS_SYSTEM_PROMPT = "You are a senior care manager revising a care plan using reviewer critiques."


def synthesize(client: OllamaClient, patient_context: str, draft: str, critiques: List[str],
               temperature: float = 0.2, seed=None) -> str:
    """Same unweighted synthesis prompt for every arm."""
    blocks = "\n\n".join(f"- Critique {i + 1}:\n{c}" for i, c in enumerate(critiques))
    user_prompt = (
        f"{patient_context}\n"
        f"Current draft:\n{draft}\n\n"
        f"Reviewer critiques:\n{blocks}\n\n"
        "Rewrite the care plan to address the critiques. Keep the sections "
        "Problem List, Actionable Clinical Actions, SDOH / Access Actions, "
        "Follow-up and Monitoring. Use concise bullets."
    )
    return client.chat(
        model=MODELS["generator"],
        user_prompt=user_prompt,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        num_predict=560,
        temperature=temperature,
        seed=seed,
    )


def critiques_for_arm(client: OllamaClient, arm: str, patient_context: str, draft: str) -> List[str]:
    if arm in ("N", "A"):
        return [generate_role_critique(client, MODELS[role], role, patient_context, draft) for role in ROLE_ORDER]
    user_prompt = (
        f"{patient_context}\n"
        f"Current draft care plan:\n{draft}\n\n"
        "Provide a critique with 5-8 concrete bullet points. "
        "Do not rewrite the full plan."
    )
    return [
        client.chat(model=MODELS["generic_critic"], user_prompt=user_prompt,
                    system_prompt=GENERIC_CRITIC_PROMPT, num_predict=260, temperature=0.2)
        for _ in ROLE_ORDER
    ]


def nash_product(u: Dict[str, float]) -> float:
    """Symmetric Nash product with disagreement point d = 0 (no plan): the geometric mean."""
    return float(np.prod([max(u[k], FLOOR) ** (1 / 3) for k in DOMAINS]))


def run_arm(client: OllamaClient, arm: str, patient_context: str, judge_retries: int) -> Dict[str, object]:
    calls = {"generation": 0, "selection_judge": 0, "outcome_judge": 0}

    draft = generate_initial_plan(client, MODELS["generator"], patient_context)
    calls["generation"] += 1
    d = evaluate_plan(client, MODELS["selection_judge"], patient_context, draft, judge_retries)
    calls["selection_judge"] += 1

    rounds = []
    candidates: List[str] = []
    for r in range(1, N_ROUNDS + 1):
        crit = critiques_for_arm(client, arm, patient_context, draft)
        calls["generation"] += len(crit)
        if r < N_ROUNDS:
            draft = synthesize(client, patient_context, draft, crit)
            calls["generation"] += 1
        else:
            candidates = [
                synthesize(client, patient_context, draft, crit,
                           temperature=CANDIDATE_TEMPERATURE, seed=s)
                for s in range(1, K_CANDIDATES + 1)
            ]
            calls["generation"] += K_CANDIDATES
        rounds.append({"round": r, "critiques": crit})

    cand_scores = []
    for c in candidates:
        cand_scores.append(evaluate_plan(client, MODELS["selection_judge"], patient_context, c, judge_retries))
        calls["selection_judge"] += 1

    if arm == "A":
        utilities = [float(np.mean([u[k] for k in DOMAINS])) for u in cand_scores]
    else:
        utilities = [nash_product(u) for u in cand_scores]
    chosen = int(np.argmax(utilities))  # argmax returns the first max -> lowest seed on ties
    # Descriptive only: whether any candidate scored at or above the draft in all three domains.
    below_disagreement = not any(all(u[k] >= d[k] for k in DOMAINS) for u in cand_scores)

    plan = candidates[chosen]
    outcome = evaluate_plan(client, MODELS["outcome_judge"], patient_context, plan, judge_retries)
    calls["outcome_judge"] += 1

    return {
        "plan": plan,
        "outcome": outcome,
        "draft": draft,
        "draft_selection_scores": d,
        "candidate_scores": cand_scores,
        "candidate_utilities": utilities,
        "chosen_index": chosen,
        "below_disagreement": below_disagreement,
        "calls": calls,
        "rounds": rounds,
        "candidates": candidates,
    }


def load_patients(n_patients: int) -> pd.DataFrame:
    ids_path = ROOT / "data" / "r1_patient_ids.txt"
    if not ids_path.exists():
        prior = pd.read_csv(ROOT / "results" / "real_cohort_n200_seed42_results.csv", usecols=["patient_id"])
        ordered = list(dict.fromkeys(prior["patient_id"].astype(str)))
        ids_path.write_text("\n".join(ordered) + "\n")
    ids = [x for x in ids_path.read_text().split() if x]
    cohort = pd.read_parquet(ROOT / "data" / "real_cohort_analytic.parquet")
    cohort = cohort.rename(columns={"race_ethnicity": "race", "charlson_proxy": "charlson", "sdoh_count": "social_needs"})
    cohort = cohort.set_index("patient_id").loc[ids].reset_index()
    assert len(cohort) == len(ids) == 200, len(cohort)
    return cohort.head(n_patients)


def done_keys(results_csv: Path) -> set:
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv, usecols=["patient_id", "arm"])
    keys = list(zip(df["patient_id"].astype(str), df["arm"]))
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate (patient_id, arm) rows in {results_csv}; refusing to continue.")
    return set(keys)


def append_row(path: Path, row: Dict[str, object]) -> None:
    tmp = path.with_suffix(".tmp")
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n-patients", type=int, default=200)
    ap.add_argument("--judge-retries", type=int, default=4)
    ap.add_argument("--timeout-seconds", type=int, default=240)
    ap.add_argument("--shard", default="0/1", help="i/n: run patients with index %% n == i into a per-shard file.")
    ap.add_argument("--ollama-base-url", default="http://127.0.0.1:11435",
                    help="Pinned Ollama 0.13.1 server (honours think=false, matching the February run).")
    args = ap.parse_args()

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    shard_i, shard_n = map(int, args.shard.split("/"))
    if shard_n > 1:
        args.run_name = f"{args.run_name}_shard{shard_i}of{shard_n}"
    results_csv = out / f"{args.run_name}_results.csv"
    trace_jsonl = out / f"{args.run_name}_trace.jsonl"
    failures = out / f"{args.run_name}_failures.jsonl"

    client = OllamaClient(base_url=args.ollama_base_url, timeout_seconds=args.timeout_seconds)
    installed = set(client.list_models())
    missing = sorted(set(MODELS.values()) - installed)
    if missing:
        raise RuntimeError(f"Missing models: {missing}")

    config = {"models": MODELS, "arms": ARMS, "n_rounds": N_ROUNDS, "k_candidates": K_CANDIDATES,
              "candidate_temperature": CANDIDATE_TEMPERATURE, "floor": FLOOR, "args": vars(args),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"{args.run_name}_config.json").write_text(json.dumps(config, indent=2))

    patients = load_patients(args.n_patients)
    done = done_keys(results_csv)
    print(f"{len(done)} (patient, arm) rows already complete")

    for idx, row in patients.iterrows():
        if idx % shard_n != shard_i:
            continue
        pid = str(row["patient_id"])
        ctx = build_patient_context(row)
        order = ARMS[:]
        random.Random(42 + idx).shuffle(order)
        for arm in order:
            if (pid, arm) in done:
                continue
            t0 = time.time()
            try:
                res = run_arm(client, arm, ctx, args.judge_retries)
            except Exception as exc:  # noqa: BLE001
                with failures.open("a") as f:
                    f.write(json.dumps({"patient_id": pid, "arm": arm, "error": str(exc)[:500]}) + "\n")
                print(f"FAIL {idx} {arm}: {exc}")
                continue
            o = res["outcome"]
            append_row(results_csv, {
                "patient_id": pid, "arm": arm, "arm_order": "".join(order),
                "safety": o["safety"], "efficiency": o["efficiency"], "equity": o["equity"],
                "composite": float(np.mean([o[k] for k in DOMAINS])),
                "chosen_index": res["chosen_index"], "below_disagreement": res["below_disagreement"],
                "n_generation_calls": res["calls"]["generation"],
                "n_selection_judge_calls": res["calls"]["selection_judge"],
                "n_outcome_judge_calls": res["calls"]["outcome_judge"],
                "seconds": round(time.time() - t0, 1),
                "plan": res["plan"],
            })
            with trace_jsonl.open("a") as f:
                f.write(json.dumps({"patient_id": pid, "arm": arm, **{k: v for k, v in res.items() if k != "plan"}}) + "\n")
            done.add((pid, arm))
            print(f"{idx:3d} {arm} composite={np.mean([o[k] for k in DOMAINS]):.3f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
