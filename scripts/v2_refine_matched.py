#!/usr/bin/env python3
"""Compute-matched refine arm (R36): checker-guided refine with more rounds, stopping at the
selection arm's per-patient call count (or when no visible item is missing). Writes v2 record format
(run v2_final, variant matchedR) so the panel grades it with the same items."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from run_ollama_open_source_experiments import build_patient_context, generate_initial_plan  # noqa: E402
from scripts.v2_pipeline import ACTION_BUDGET, MODELS, WORD_CAP, checklist_for, load_reqmap  # noqa: E402
from scripts.v2_run import CountingClient, evaluate_with_verifier, patient_set  # noqa: E402

def refine_to_budget(client, ctx, visible, call_budget):
    plan = generate_initial_plan(client, MODELS["renderer"], ctx); calls = 1; rounds = 0
    while calls + len(visible) + 1 <= call_budget:
        met = evaluate_with_verifier(client, visible, plan); calls += len(visible)
        missing = [it["requirement"] for it in visible if not met[it["id"]]]
        if not missing:
            break
        prompt = (f"{ctx}\nCurrent care plan:\n{plan}\n\nThe plan does not yet meet these requirements:\n"
                  + "\n".join(f"- {m}" for m in missing)
                  + f"\n\nRevise the plan to meet them with specific actions (who, what, when). Use at most {ACTION_BUDGET} "
                    f"action bullets and at most {WORD_CAP} words in total. Do not start, stop, or change medications.")
        plan = client.chat(model=MODELS["renderer"], user_prompt=prompt,
                           system_prompt="You are a senior care manager writing concise action plans.",
                           num_predict=650, temperature=0.2); calls += 1; rounds += 1
    return plan, calls, rounds

def main():
    si, sn = map(int, sys.argv[1].split("/")) if len(sys.argv) > 1 else (0, 1)
    budgets = {}
    for f in (ROOT / "results").glob("v2_final_none_shard*of3.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l); budgets[r["patient_id"]] = r["compute"]["selection_shared"]["calls"]
    out = ROOT / "results" / f"v2_final_matchedR_shard{si}of{sn}.jsonl"
    done = {json.loads(l)["patient_id"] for l in out.read_text().splitlines()} if out.exists() else set()
    client = CountingClient(base_url="http://127.0.0.1:11435", timeout_seconds=300); rm = load_reqmap()
    for idx, row in patient_set("final").iterrows():
        pid = str(row["patient_id"])
        if idx % sn != si or pid in done or pid not in budgets:
            continue
        try:
            ctx = build_patient_context(row); items = checklist_for(row, rm); visible = [i for i in items if not i["eval_only"]]
            plan, calls, rounds = refine_to_budget(client, ctx, visible, budgets[pid])
        except Exception as exc:  # noqa: BLE001
            print("FAIL", idx, str(exc)[:100], flush=True); continue
        rec = {"patient_id": pid, "items": [i["id"] for i in items], "visible": [i["id"] for i in visible],
               "arms": {"R-matched": {"plan": plan, "words": len(plan.split()), "calls": calls, "rounds": rounds,
                                      "call_budget": budgets[pid]}}}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        print(idx, calls, budgets[pid], rounds, flush=True)

if __name__ == "__main__":
    main()
