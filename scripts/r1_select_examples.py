#!/usr/bin/env python3
"""
S2 Appendix examples by a fixed rule on the primary-judge Composite difference (Nash minus self-critique):
the two largest gains, two ties (|diff| < 0.005, lowest patient index first), and the two largest losses.
Writes a LOCAL draft (contains patient context; review for identifiability before any use).

  python scripts/r1_select_examples.py --run r1_three_arm_n200 --out <local dir>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_ollama_open_source_experiments import build_patient_context  # noqa: E402
from scripts.r1_rescore_judges import load_results  # noqa: E402
from scripts.run_r1_three_arm import load_patients  # noqa: E402

ARM_NAMES = {"N": "Role-specialized Nash", "B": "Generic self-critique", "A": "Role-specialized additive"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load_results(a.run)
    pts = load_patients(200).reset_index().set_index("patient_id")
    w = d.pivot(index="patient_id", columns="arm", values="composite").dropna()
    w["diff"] = w["N"] - w["B"]
    w["idx"] = pts.loc[w.index, "index"]
    gains = w.sort_values(["diff", "idx"], ascending=[False, True]).head(2)
    losses = w.sort_values(["diff", "idx"], ascending=[True, True]).head(2)
    ties = w[w["diff"].abs() < 0.005].sort_values("idx").head(2)
    chosen = [("Largest gain", gains), ("Tie", ties), ("Largest loss", losses)]

    lines = ["# S2 Appendix. Worked examples (DRAFT: review each context for identifiability)\n"]
    k = 0
    for label, df in chosen:
        for pid in df.index:
            k += 1
            lines.append(f"## Example {k}. {label} (Composite difference {df.loc[pid, 'diff']:+.3f})\n")
            lines.append("Patient context:\n\n```\n" + build_patient_context(pts.loc[pid]) + "```\n")
            for arm in ["N", "B", "A"]:
                r = d[(d.patient_id == pid) & (d.arm == arm)].iloc[0]
                lines.append(f"### {ARM_NAMES[arm]}\n\nSafety {r.safety:.2f}; Efficiency {r.efficiency:.2f}; "
                             f"Equity {r.equity:.2f}; Composite {r.composite:.3f}\n\n```\n{r.plan}\n```\n")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "S2_Appendix_examples_DRAFT.md").write_text("\n".join(lines))
    print("wrote", out / "S2_Appendix_examples_DRAFT.md")


if __name__ == "__main__":
    main()
