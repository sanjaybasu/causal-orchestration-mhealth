#!/usr/bin/env python3
"""Table 1: analytic cohort (N=5,148) and the paired experimental sample (N=200)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r1_analysis"
RACE_ORDER = ["Black or African American", "White", "Hispanic", "Asian", "American Indian or Alaska Native",
              "Native Hawaiian or Other Pacific Islander", "Other", "Unknown"]


def describe(c: pd.DataFrame) -> dict:
    n = len(c)
    pct = lambda k: {"n": int(k), "pct": round(100 * k / n, 1)}  # noqa: E731
    race = c["race_ethnicity"].astype(str)
    return {
        "N": n,
        "age_mean": round(c.age.mean(), 1), "age_sd": round(c.age.std(), 1),
        "female": pct(c.female.sum()),
        "race": {r: pct((race == r).sum()) for r in sorted(race.unique())},
        "state": {s: pct((c.state == s).sum()) for s in sorted(c.state.unique())},
        "any_social_need": pct((c.sdoh_count >= 1).sum()),
        "social_need_count_mean": round(c.sdoh_count.mean(), 2), "social_need_count_sd": round(c.sdoh_count.std(), 2),
        "any_clinical_goal": pct((c.charlson_proxy >= 1).sum()),
        "clinical_goal_count_mean": round(c.charlson_proxy.mean(), 2),
        "ed_mean": round(c.ed_visits_study_period.mean(), 2), "ed_sd": round(c.ed_visits_study_period.std(), 2),
        "ip_mean": round(c.ip_admissions_study_period.mean(), 2), "ip_sd": round(c.ip_admissions_study_period.std(), 2),
    }


c = pd.read_parquet(ROOT / "data" / "real_cohort_analytic.parquet")
ids = (ROOT / "data" / "r1_patient_ids.txt").read_text().split()
OUT.mkdir(parents=True, exist_ok=True)
res = {"cohort": describe(c), "sample": describe(c.set_index("patient_id").loc[ids].reset_index())}
(OUT / "table1.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
