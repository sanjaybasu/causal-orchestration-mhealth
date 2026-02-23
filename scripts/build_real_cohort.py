"""
build_real_cohort.py
====================
Constructs the analytic cohort for the Nash orchestration study from real
Waymark Medicaid care management data.

Outputs
-------
data/real_cohort_analytic.parquet
    Full analytic cohort (N ≈ 6,896) used for Table 1 demographics.

data/real_cohort_experiment_eligible.csv.gz
    Post-hold-out sampling frame with column names compatible with
    run_ollama_open_source_experiments.py.

Usage
-----
Run from the waymark-local root directory:

    python packaging/causal_orchestration_mhealth/scripts/build_real_cohort.py

The script writes both output files and prints a demographic summary.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DATA_IN = ROOT / "data" / "real_inputs"
DATA_OUT = ROOT / "data"
EXCLUSION_FILE = DATA_OUT / "prompt_dev_exclusion.txt"

STUDY_START = date(2023, 4, 1)

# ---------------------------------------------------------------------------
# Category maps
# ---------------------------------------------------------------------------
SDOH_CATEGORIES = {
    "TRANSPORTATION", "FOOD_INSECURITY", "HOUSING_INSECURITY", "FINANCIAL",
    "EMPLOYMENT", "SOCIAL_CONNECTION", "HOUSING_QUALITY_SAFETY", "LEGAL",
    "CHILDCARE", "UTILITIES", "FOOD_DIET_NUTRITION",
}

CLINICAL_CATEGORIES = {
    "MEDICATION_ADHERENCE", "HYPERTENSION", "DIABETES", "MENTAL_HEALTH",
    "DEPRESSION", "ANXIETY", "ASTHMA_COPD", "HEART_FAILURE", "SUBSTANCE_USE",
    "OTHER_MENTAL_BEHAVIORAL", "CARE_FOR_MH_BH", "MEDICATION_OPTIMIZATION",
    "SMOKING_CESSATION", "ALCOHOL_USE",
}

RACE_MAP = {
    "white": "White",
    "black or african american": "Black/African American",
    "hispanic": "Hispanic",
    "asian": "Asian",
    "native hawaiian or other pacific islander": "Native Hawaiian or Other Pacific Islander",
    "american indian or alaska native": "American Indian or Alaska Native",
    "other race": "Other",
    "unknown": "Unknown",
}


# ---------------------------------------------------------------------------
# Step 1: Load eligibility and compute age
# ---------------------------------------------------------------------------
def load_eligibility() -> pd.DataFrame:
    print("Loading eligibility...")
    elig = pd.read_csv(DATA_IN / "eligibility.csv", low_memory=False)

    elig["birth_date"] = pd.to_datetime(elig["birth_date"], errors="coerce")
    elig["death_date"] = pd.to_datetime(elig["death_date"], errors="coerce")

    study_start = pd.Timestamp(STUDY_START)
    elig["age"] = ((study_start - elig["birth_date"]).dt.days / 365.25).round(1)

    # Inclusion: age 18-89, medicaid, no death before study start
    elig = elig[elig["payer_type"] == "medicaid"].copy()
    elig = elig[(elig["age"] >= 18) & (elig["age"] <= 89)].copy()
    elig = elig[elig["death_date"].isna() | (elig["death_date"] >= study_start)].copy()

    elig["female"] = (elig["gender"].str.lower() == "female").astype(int)
    elig["race_ethnicity"] = elig["race"].str.lower().map(RACE_MAP).fillna("Unknown")

    print(f"  Eligibility after filters: {len(elig):,} rows")
    return elig[["person_id", "member_id", "age", "female", "race_ethnicity", "state",
                 "birth_date", "enrollment_start_date", "enrollment_end_date"]].drop_duplicates("member_id")


# ---------------------------------------------------------------------------
# Step 2: Filter to activated members
# ---------------------------------------------------------------------------
def load_activated_members() -> pd.Series:
    print("Loading member status events...")
    status = pd.read_parquet(DATA_IN / "member_status_event.parquet")
    activated = status[status["to_status"] == "ACTIVATED"]["member_id"].unique()
    print(f"  Members ever ACTIVATED: {len(activated):,}")
    return activated


# ---------------------------------------------------------------------------
# Step 3: Load member→patient map
# ---------------------------------------------------------------------------
def load_member_patient_map() -> pd.DataFrame:
    print("Loading member-patient map...")
    mpm = pd.read_csv(DATA_IN / "member_patient_map.csv")
    return mpm[["patient_id", "member_id"]].drop_duplicates()


# ---------------------------------------------------------------------------
# Step 4: Aggregate goals per patient
# ---------------------------------------------------------------------------
def aggregate_goals(patient_ids: set) -> pd.DataFrame:
    print("Loading goals...")
    goals = pd.read_parquet(DATA_IN / "member_goals.parquet")
    goals = goals[~goals["deleted"]]
    goals = goals[goals["patient_id"].isin(patient_ids)]

    def agg_goals(grp: pd.DataFrame) -> pd.Series:
        cats = set(grp["category"].dropna().str.upper())
        sdoh = sorted(cats & SDOH_CATEGORIES)
        clinical = sorted(cats & CLINICAL_CATEGORIES)
        has_nondeft = bool(cats - {"DEFAULT"})
        return pd.Series({
            "sdoh_categories": "|".join(sdoh),
            "clinical_categories": "|".join(clinical),
            "sdoh_count": len(sdoh),
            "charlson_proxy": min(len(clinical), 6),
            "has_nondeft_goal": has_nondeft,
        })

    agg = goals.groupby("patient_id").apply(agg_goals).reset_index()
    print(f"  Patients with ≥1 non-deleted goal: {len(agg):,}")
    return agg


# ---------------------------------------------------------------------------
# Step 5: Filter to patients with ≥1 encounter
# ---------------------------------------------------------------------------
def filter_has_encounter(patient_ids: set) -> set:
    print("Loading encounters...")
    enc = pd.read_parquet(DATA_IN / "encounters.parquet")
    enc = enc[enc["deleted_at"].isna()]
    has_enc = set(enc[enc["patient_id"].isin(patient_ids)]["patient_id"].unique())
    print(f"  Patients with ≥1 non-deleted encounter: {len(has_enc):,}")
    return has_enc


# ---------------------------------------------------------------------------
# Step 6: Hospital utilization in prior 12 months of study start
# ---------------------------------------------------------------------------
def aggregate_hospital_utilization(patient_ids: set) -> pd.DataFrame:
    print("Loading hospital visits...")
    hosp = pd.read_parquet(DATA_IN / "hospital_visits.parquet")
    hosp = hosp[hosp["patient_id"].isin(patient_ids)]
    hosp["admit_date"] = pd.to_datetime(hosp["admit_date"], errors="coerce")

    # Hospital visit data begins June 2023; use all available data from study start onward
    # as a utilization measure. Label columns accordingly.
    cutoff_start = pd.Timestamp(STUDY_START, tz="UTC")
    # Normalize admit_date timezone for comparison
    admit = hosp["admit_date"]
    if admit.dt.tz is None:
        admit = admit.dt.tz_localize("UTC")
    hosp = hosp.copy()
    hosp["admit_date_utc"] = admit
    window = hosp[hosp["admit_date_utc"] >= cutoff_start]

    ed = window[window["patient_class_code"] == "E"].groupby("patient_id").size().rename("ed_visits_study_period")
    ip = window[window["patient_class_code"] == "I"].groupby("patient_id").size().rename("ip_admissions_study_period")

    util = pd.concat([ed, ip], axis=1).reset_index().fillna(0)
    util["ed_visits_study_period"] = util["ed_visits_study_period"].astype(int)
    util["ip_admissions_study_period"] = util["ip_admissions_study_period"].astype(int)
    return util


# ---------------------------------------------------------------------------
# Step 7: Compute baseline_risk from charlson_proxy
# ---------------------------------------------------------------------------
def compute_baseline_risk(charlson_proxy: pd.Series) -> pd.Series:
    # Scale: charlson_proxy 0→0.0, 6→0.15; linear, capped
    return (charlson_proxy / 6.0 * 0.15).round(4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # Step 1: eligible members
    elig = load_eligibility()
    eligible_members = set(elig["member_id"])

    # Step 2: activated members
    activated = load_activated_members()
    eligible_activated = set(activated) & eligible_members
    print(f"  Eligible + activated: {len(eligible_activated):,}")

    # Step 3: member → patient map
    mpm = load_member_patient_map()
    mpm_activated = mpm[mpm["member_id"].isin(eligible_activated)]
    print(f"  Activated members with patient_id: {len(mpm_activated):,}")

    # Step 4: goals (require ≥1 non-deleted goal)
    patient_ids_activated = set(mpm_activated["patient_id"])
    goals_agg = aggregate_goals(patient_ids_activated)
    patients_with_goals = set(goals_agg["patient_id"])

    # Step 5: require ≥1 encounter
    patients_with_enc = filter_has_encounter(patients_with_goals)
    analytic_patients = patients_with_goals & patients_with_enc

    # Step 6: hospital utilization
    util = aggregate_hospital_utilization(analytic_patients)
    util_patients = set(util["patient_id"])

    # ---------------------------------------------------------------------------
    # Assemble analytic cohort
    # ---------------------------------------------------------------------------
    print("\nAssembling analytic cohort...")

    # Join goals → mpm → elig
    cohort = goals_agg[goals_agg["patient_id"].isin(analytic_patients)].copy()
    cohort = cohort.merge(mpm_activated[["patient_id", "member_id"]], on="patient_id", how="left")
    cohort = cohort.merge(
        elig[["member_id", "person_id", "age", "female", "race_ethnicity", "state"]],
        on="member_id", how="left",
    )
    cohort = cohort.merge(util, on="patient_id", how="left")
    cohort["ed_visits_study_period"] = cohort["ed_visits_study_period"].fillna(0).astype(int)
    cohort["ip_admissions_study_period"] = cohort["ip_admissions_study_period"].fillna(0).astype(int)
    cohort["baseline_risk"] = compute_baseline_risk(cohort["charlson_proxy"])

    # Drop rows with missing age (unmapped members)
    cohort = cohort[cohort["age"].notna()].copy()

    print(f"\n  Analytic cohort N = {len(cohort):,}")
    print(f"  Age: mean={cohort['age'].mean():.1f} SD={cohort['age'].std():.1f}")
    print(f"  Female: {cohort['female'].mean()*100:.1f}%")
    print(f"  Race/ethnicity:\n{cohort['race_ethnicity'].value_counts().to_string()}")
    print(f"  State:\n{cohort['state'].value_counts().to_string()}")
    print(f"  SDOH count: mean={cohort['sdoh_count'].mean():.2f} SD={cohort['sdoh_count'].std():.2f}")
    print(f"  Charlson proxy: mean={cohort['charlson_proxy'].mean():.2f}")
    print(f"  ED visits prior 12mo: mean={cohort['ed_visits_study_period'].mean():.2f}")
    print(f"  IP admissions prior 12mo: mean={cohort['ip_admissions_study_period'].mean():.2f}")

    # Save analytic cohort
    analytic_path = DATA_OUT / "real_cohort_analytic.parquet"
    cohort.to_parquet(analytic_path, index=False)
    print(f"\n  Saved analytic cohort → {analytic_path}")

    # ---------------------------------------------------------------------------
    # Prompt development hold-out (seed=0, N=50)
    # ---------------------------------------------------------------------------
    if EXCLUSION_FILE.exists():
        print(f"\n  Prompt-dev hold-out already exists at {EXCLUSION_FILE}; not overwriting.")
        with open(EXCLUSION_FILE) as f:
            holdout_ids = set(line.strip() for line in f if line.strip())
        print(f"  Hold-out contains {len(holdout_ids)} patient_ids.")
    else:
        rng = np.random.default_rng(seed=0)
        holdout_idx = rng.choice(len(cohort), size=min(50, len(cohort)), replace=False)
        holdout_ids = set(cohort.iloc[holdout_idx]["patient_id"].tolist())
        with open(EXCLUSION_FILE, "w") as f:
            for pid in sorted(holdout_ids):
                f.write(pid + "\n")
        print(f"\n  Wrote prompt-dev hold-out ({len(holdout_ids)} patients) → {EXCLUSION_FILE}")

    # ---------------------------------------------------------------------------
    # Experiment-eligible cohort (exclude hold-out)
    # ---------------------------------------------------------------------------
    eligible = cohort[~cohort["patient_id"].isin(holdout_ids)].copy()
    print(f"\n  Experiment-eligible pool (after hold-out exclusion): N = {len(eligible):,}")

    # Rename columns for compatibility with run_ollama_open_source_experiments.py
    # Current runner expects: member_id, age, charlson, social_needs, race, female, baseline_risk
    eligible = eligible.rename(columns={
        "patient_id": "member_id",   # use patient_id as the member identifier
        "charlson_proxy": "charlson",
        "sdoh_count": "social_needs",
        "race_ethnicity": "race",
    })
    # Keep enriched context columns alongside for build_patient_context() upgrade
    # Columns: member_id, age, charlson, social_needs, race, female, baseline_risk,
    #          sdoh_categories, clinical_categories, ed_visits_study_period, ip_admissions_study_period,
    #          state, person_id

    eligible_path = DATA_OUT / "real_cohort_experiment_eligible.csv.gz"
    eligible.to_csv(eligible_path, index=False, compression="gzip")
    print(f"  Saved experiment-eligible cohort → {eligible_path}")

    # ---------------------------------------------------------------------------
    # Summary JSON for manuscript
    # ---------------------------------------------------------------------------
    summary = {
        "analytic_cohort_n": int(len(cohort)),
        "experiment_eligible_n": int(len(eligible)),
        "holdout_n": int(len(holdout_ids)),
        "age_mean": round(float(cohort["age"].mean()), 1),
        "age_sd": round(float(cohort["age"].std()), 1),
        "female_pct": round(float(cohort["female"].mean() * 100), 1),
        "race_distribution": cohort["race_ethnicity"].value_counts().to_dict(),
        "state_distribution": cohort["state"].value_counts().to_dict(),
        "sdoh_count_mean": round(float(cohort["sdoh_count"].mean()), 2),
        "charlson_proxy_mean": round(float(cohort["charlson_proxy"].mean()), 2),
        "ed_visits_study_period_mean": round(float(cohort["ed_visits_study_period"].mean()), 2),
        "ip_admissions_study_period_mean": round(float(cohort["ip_admissions_study_period"].mean()), 2),
        "pct_any_sdoh_goal": round(float((cohort["sdoh_count"] > 0).mean() * 100), 1),
        "pct_any_clinical_goal": round(
            float((cohort["charlson_proxy"] > 0).mean() * 100), 1
        ),
        "study_start": str(STUDY_START),
    }
    summary_path = DATA_OUT / "real_cohort_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved cohort summary → {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
