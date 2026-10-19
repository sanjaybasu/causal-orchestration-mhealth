#!/usr/bin/env python3
"""Convert the v1 three-arm plans (generic self-critique B, prompt-Nash N, additive A) into v2 record format
so the v2 panel grades them on the same checklist. Output: results/v1final_none_shard0of1.jsonl"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.r1_rescore_judges import load_results  # noqa: E402
from scripts.v2_pipeline import checklist_for, load_reqmap  # noqa: E402
from scripts.v2_run import patient_set  # noqa: E402
rm = load_reqmap(); pts = patient_set("final").set_index("patient_id"); d = load_results("r1_three_arm_n200")
out = ROOT / "results" / "v1final_none_shard0of1.jsonl"
with open(out, "w") as f:
    for pid, g in d.groupby("patient_id"):
        items = checklist_for(pts.loc[pid], rm)
        f.write(json.dumps({"patient_id": pid, "items": [i["id"] for i in items],
                            "visible": [i["id"] for i in items if not i["eval_only"]],
                            "arms": {f"v1-{r.arm}": {"plan": r.plan, "words": len(str(r.plan).split())} for r in g.itertuples()}}) + "\n")
print("wrote", out)
