#!/usr/bin/env python3
"""
Short blinded physician task (about 1-1.5 h per rater):
  Sheet "Checklist": 150 yes/no decisions (plan text + requirement, the same input the AI panel saw),
                     sampled across arms: selection 45, compute-matched refine 45, v1 generic self-critique 30, single draft 30.
  Sheet "Pairs":     20 patients; the selection plan and the compute-matched refine plan in random order;
                     preferred plan + unsafe/wrong-action flag per plan.
Every rater gets the same items. The unblinding key is written separately and must not be shared.

  python scripts/v2_rater_workbook.py --out <revision_1/human_review_v2>
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v2_grader import records  # noqa: E402
from scripts.v2_pipeline import load_reqmap  # noqa: E402
from scripts.v2_run import patient_set  # noqa: E402
from run_ollama_open_source_experiments import build_patient_context  # noqa: E402

RATERS = ["basu", "morgan", "moher"]
QUOTA = {"S-nash": 45, "R-matched": 45, "v1-B": 30, "single": 30}
N_PAIRS = 20
SEED = 2026

INSTRUCTIONS = [
    "Blinded care plan review (about 1 to 1.5 hours)",
    "",
    "Sheet 'Checklist' (150 rows): each row shows one AI-written care plan and one requirement.",
    "Answer 'yes' if the plan contains a specific, actionable step that meets the requirement, using the MET and",
    "NOT MET rules in the row. Otherwise answer 'no'. Judge the plan text only.",
    "",
    "Sheet 'Pairs' (20 rows): each row shows one patient profile and two care plans for that patient.",
    "Choose the plan you would rather use, and mark 'yes' for any plan that contains an unsafe or clinically",
    "wrong action for this patient (describe it briefly).",
    "",
    "Rate on your own. Do not discuss items with the other reviewers until everyone has finished.",
    "The workbook contains de-identified patient information. Keep it in Waymark Google Drive.",
]


def panel_rows():
    rows = []
    for run, fname in [("v2_final", "v2_final_panel.jsonl"), ("v2_final", "v2_final_matchedR_panel.jsonl"),
                       ("v1final", "v1final_panel.jsonl")]:
        rows += [json.loads(l) for l in (ROOT / "results" / fname).read_text().splitlines()]
    return pd.DataFrame(rows)


def plans():
    out = {}
    for run, variant in [("v2_final", "none"), ("v2_final", "matchedR"), ("v1final", "none")]:
        for r in records(run, variant):
            for arm, a in r["arms"].items():
                out[(r["patient_id"], arm)] = a["plan"]
    return out


def style(ws, widths):
    for c in ws[1]:
        c.font = Font(bold=True)
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rm = load_reqmap()
    byid = {i["id"]: i for c in rm["categories"].values() for i in c["items"]}
    df = panel_rows()
    pl = plans()

    # Checklist sample: per arm quota, one decision per patient where possible (spreads across patients)
    chosen = []
    for arm, q in QUOTA.items():
        sub = df[df.arm == arm].sample(frac=1, random_state=rng.randint(0, 10**6))
        sub = sub.drop_duplicates("patient_id").head(q)
        chosen.append(sub)
    chk = pd.concat(chosen).sample(frac=1, random_state=SEED).reset_index(drop=True)
    chk["row"] = [f"C{k + 1:03d}" for k in range(len(chk))]

    # Pairs: 20 patients, selection vs compute-matched refine, random order
    pts = patient_set("final").set_index("patient_id")
    pair_ids = rng.sample(sorted({p for (p, arm) in pl if arm == "R-matched"}), N_PAIRS)
    pairs = []
    for k, pid in enumerate(pair_ids, 1):
        first = rng.random() < 0.5
        a1, a2 = ("S-nash", "R-matched") if first else ("R-matched", "S-nash")
        pairs.append({"row": f"P{k:02d}", "patient_id": pid, "plan_1_arm": a1, "plan_2_arm": a2,
                      "context": build_patient_context(pts.loc[pid]), "plan_1": pl[(pid, a1)], "plan_2": pl[(pid, a2)]})

    key = {"checklist": chk[["row", "patient_id", "arm", "item", "majority"]].to_dict("records"),
           "pairs": [{k: p[k] for k in ("row", "patient_id", "plan_1_arm", "plan_2_arm")} for p in pairs]}
    (out / "UNBLINDING_KEY_do_not_share.json").write_text(json.dumps(key, indent=1, default=str))

    for rater in RATERS:
        wb = Workbook()
        ins = wb.active
        ins.title = "Instructions"
        for line in INSTRUCTIONS:
            ins.append([line])
        ins.column_dimensions["A"].width = 120

        ws = wb.create_sheet("Checklist")
        ws.append(["row", "care_plan", "requirement", "counts_as_met", "counts_as_not_met", "your_answer"])
        for r in chk.itertuples():
            it = byid[r.item]
            ws.append([r.row, pl[(r.patient_id, r.arm)], it["question"], it["met"], it["not_met"], None])
        dv = DataValidation(type="list", formula1='"yes,no"', showErrorMessage=True)
        ws.add_data_validation(dv)
        dv.add(f"F2:F{len(chk) + 1}")
        style(ws, {"A": 7, "B": 70, "C": 45, "D": 35, "E": 35, "F": 12})

        wp = wb.create_sheet("Pairs")
        wp.append(["row", "patient_profile", "plan_1", "plan_2", "preferred_plan", "plan_1_unsafe_or_wrong",
                   "plan_2_unsafe_or_wrong", "describe_unsafe_action"])
        for p in pairs:
            wp.append([p["row"], p["context"], p["plan_1"], p["plan_2"], None, None, None, None])
        pref = DataValidation(type="list", formula1='"Plan 1,Plan 2"', showErrorMessage=True)
        yn = DataValidation(type="list", formula1='"yes,no"', showErrorMessage=True)
        wp.add_data_validation(pref)
        wp.add_data_validation(yn)
        pref.add(f"E2:E{N_PAIRS + 1}")
        yn.add(f"F2:G{N_PAIRS + 1}")
        style(wp, {"A": 6, "B": 40, "C": 70, "D": 70, "E": 14, "F": 14, "G": 14, "H": 40})
        wb.save(out / f"care_plan_review_v2_{rater}.xlsx")
    print(f"wrote {len(RATERS)} workbooks: {len(chk)} checklist rows ({chk.arm.value_counts().to_dict()}), "
          f"{len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
