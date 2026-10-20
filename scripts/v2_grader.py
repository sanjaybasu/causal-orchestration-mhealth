#!/usr/bin/env python3
"""
Design v2 grading panel: several strong local models judge each (plan, checklist item) pair
independently; the panel verdict is the majority. Reports pairwise Cohen's kappa and Fleiss'
kappa with bootstrap CIs (resampling patients), and builds the physician calibration sample.

  python scripts/v2_grader.py --run v2_dev2 --grade        # grade every arm's plan (resumable)
  python scripts/v2_grader.py --run v2_dev2 --agreement    # kappa table
  python scripts/v2_grader.py --run v2_dev2 --calibration-sheet <out.xlsx> --n 200
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_ollama_open_source_experiments import OllamaClient  # noqa: E402
from scripts.v2_pipeline import load_reqmap  # noqa: E402

PANEL = {
    # name: (base_url, model, chat settings)
    "gptoss20b": ("http://127.0.0.1:11435", "gpt-oss:20b", {"num_predict": 2048, "think": "low"}),
    "qwen36_35b": ("http://127.0.0.1:11436", "batiai/qwen3.6-35b:iq4", {"num_predict": 20, "think": False}),
    "medgemma27b": ("http://127.0.0.1:11436", "alibayram/medgemma:27b", {"num_predict": 20, "think": False}),
}


def item_prompt(it: dict, plan: str) -> str:
    return (f"Care plan:\n{plan}\n\nRequirement: {it['question']}\nCounts as MET: {it['met']}\n"
            f"Counts as NOT MET: {it['not_met']}\n\nAnswer ONLY yes or no.")


def judge(client: OllamaClient, model: str, settings: dict, it: dict, plan: str) -> bool:
    raw = client.chat(model=model, user_prompt=item_prompt(it, plan),
                      system_prompt="You are a strict clinical auditor.", temperature=0.0, **settings)
    return raw.strip().lower().lstrip("*\"' ").startswith("yes")


def records(run: str, variant: str = "none"):
    for f in sorted((ROOT / "results").glob(f"{run}_{variant}_shard*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def grade(run: str, variant: str = "none", shard: str = "0/1") -> None:
    si, sn = map(int, shard.split("/"))
    rm = load_reqmap()
    byid = {i["id"]: i for c in rm["categories"].values() for i in c["items"]}
    out = ROOT / "results" / (f"{run}_panel.jsonl" if variant == "none" else f"{run}_{variant}_panel.jsonl")
    done = set()
    if out.exists():
        done = {(r["patient_id"], r["arm"], r["item"]) for r in map(json.loads, out.read_text().splitlines())}
    clients = {k: OllamaClient(base_url=u, timeout_seconds=600) for k, (u, _, _) in PANEL.items()}
    skipped = 0
    with open(out, "a") as f:
        for k, rec in enumerate(records(run, variant)):
            if k % sn != si:
                continue
            for arm, a in rec["arms"].items():
                for iid in rec["items"]:
                    key = (rec["patient_id"], arm, iid)
                    if key in done:
                        continue
                    try:
                        v = {k: judge(clients[k], m, s, byid[iid], a["plan"]) for k, (_, m, s) in PANEL.items()}
                    except Exception as exc:  # noqa: BLE001  (e.g. timeout after laptop sleep); retried next pass
                        skipped += 1
                        print(f"SKIP {rec['patient_id'][-6:]} {arm} {iid}: {str(exc)[:80]}", flush=True)
                        continue
                    f.write(json.dumps({"patient_id": rec["patient_id"], "arm": arm, "item": iid,
                                        "eval_only": byid[iid]["eval_only"], "domain": byid[iid]["domain"],
                                        "votes": v, "majority": sum(v.values()) * 2 > len(v)}) + "\n")
                    f.flush()
    print(f"GRADE PASS DONE skipped={skipped}", flush=True)
    if skipped:
        sys.exit(3)


def remaining(run: str, variant: str = "none") -> int:
    out = ROOT / "results" / (f"{run}_panel.jsonl" if variant == "none" else f"{run}_{variant}_panel.jsonl")
    done = set()
    if out.exists():
        done = {(r["patient_id"], r["arm"], r["item"]) for r in map(json.loads, out.read_text().splitlines())}
    return sum((rec["patient_id"], arm, iid) not in done
               for rec in records(run, variant) for arm in rec["arms"] for iid in rec["items"])


def cohen(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def fleiss(M):
    """M: n x k boolean votes."""
    M = np.asarray(M, float)
    n, k = M.shape
    p1 = M.mean()
    P = ((M.sum(1) ** 2 + (k - M.sum(1)) ** 2) - k) / (k * (k - 1))
    pe = p1 ** 2 + (1 - p1) ** 2
    return float((P.mean() - pe) / (1 - pe)) if pe < 1 else float("nan")


def agreement(run: str, B: int = 1000) -> dict:
    rows = [json.loads(l) for l in (ROOT / "results" / f"{run}_panel.jsonl").read_text().splitlines()]
    names = list(PANEL)
    pts = sorted({r["patient_id"] for r in rows})
    by_pt = {p: [r for r in rows if r["patient_id"] == p] for p in pts}
    rng = random.Random(1)

    def stats(sample_rows):
        V = {k: [r["votes"][k] for r in sample_rows] for k in names}
        out = {f"{a} vs {b}": cohen(V[a], V[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        out["fleiss"] = fleiss(np.array([[r["votes"][k] for k in names] for r in sample_rows]))
        return out

    point = stats(rows)
    boots = []
    for _ in range(B):
        s = [r for p in rng.choices(pts, k=len(pts)) for r in by_pt[p]]
        boots.append(stats(s))
    res = {k: {"kappa": v, "ci_low": float(np.percentile([b[k] for b in boots], 2.5)),
               "ci_high": float(np.percentile([b[k] for b in boots], 97.5))} for k, v in point.items()}
    res["n_decisions"] = len(rows)
    res["n_patients"] = len(pts)
    per_item = {}
    for iid in sorted({r["item"] for r in rows}):
        rs = [r for r in rows if r["item"] == iid]
        if len(rs) >= 10:
            per_item[iid] = {"n": len(rs), "fleiss": fleiss(np.array([[r["votes"][k] for k in names] for r in rs])),
                             "met_rate": float(np.mean([r["majority"] for r in rs]))}
    res["per_item"] = per_item
    (ROOT / "results" / "r1_analysis").mkdir(exist_ok=True)
    (ROOT / "results" / "r1_analysis" / f"{run}_panel_agreement.json").write_text(json.dumps(res, indent=2))
    return res


def calibration_sheet(run: str, path: str, n: int) -> None:
    """Random (plan, item) decisions for physician labeling; panel verdicts are hidden from the sheet."""
    from openpyxl import Workbook
    rm = load_reqmap()
    byid = {i["id"]: i for c in rm["categories"].values() for i in c["items"]}
    rows = [json.loads(l) for l in (ROOT / "results" / f"{run}_panel.jsonl").read_text().splitlines()]
    plans = {(r["patient_id"], arm): a["plan"] for r in records(run) for arm, a in r["arms"].items()}
    rng = random.Random(2026)
    sample = rng.sample(rows, min(n, len(rows)))
    wb = Workbook()
    ws = wb.active
    ws.title = "Calibration"
    ws.append(["row", "care_plan", "requirement", "counts_as_met", "counts_as_not_met", "your_answer (yes/no)"])
    key = []
    for k, r in enumerate(sample, 1):
        it = byid[r["item"]]
        ws.append([k, plans[(r["patient_id"], r["arm"])], it["question"], it["met"], it["not_met"], None])
        key.append({"row": k, **r})
    wb.save(path)
    Path(path).with_suffix(".key.json").write_text(json.dumps(key, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--calibration-sheet")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--variant", default="none")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--check", action="store_true", help="exit 0 if no decisions remain, else 3")
    a = ap.parse_args()
    if a.check:
        n = remaining(a.run, a.variant)
        print(f"{a.run} {a.variant}: {n} decisions remaining", flush=True)
        sys.exit(0 if n == 0 else 3)
    if a.grade:
        grade(a.run, a.variant, a.shard)
    if a.agreement:
        print(json.dumps({k: v for k, v in agreement(a.run).items() if k != "per_item"}, indent=2))
    if a.calibration_sheet:
        calibration_sheet(a.run, a.calibration_sheet, a.n)


if __name__ == "__main__":
    main()
