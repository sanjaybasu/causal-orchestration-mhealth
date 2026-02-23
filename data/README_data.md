# Data Dictionary

## Files in this directory

### `real_cohort_analytic.parquet`

De-identified analytic cohort: 5,148 activated Medicaid care management patients from Virginia and Washington. Study start date: April 1, 2023. Built from PHI-protected source tables by `scripts/build_real_cohort.py`. Used as the source for Table 1 demographics and by `scripts/build_peer_review_artifacts.py`.

| Variable | Type | Description |
|---|---|---|
| `patient_id` | string | De-identified patient identifier |
| `member_id` | string | De-identified member identifier |
| `age` | float | Age in years at study start (April 1, 2023) |
| `female` | integer | Sex indicator (1=female, 0=not female) |
| `race_ethnicity` | string | Race/ethnicity category (standardized to 8 categories) |
| `state` | string | State of enrollment (VA or WA) |
| `sdoh_categories` | string | Pipe-delimited list of documented social need goal categories |
| `clinical_categories` | string | Pipe-delimited list of documented clinical goal categories |
| `sdoh_count` | integer | Count of distinct social need goal categories |
| `charlson_proxy` | integer | Clinical complexity score (count of clinical goal categories, capped at 6) |
| `baseline_risk` | float | Derived from charlson_proxy: charlson_proxy / 6 * 0.15 |
| `ed_visits_study_period` | integer | Emergency department visits from study start onward |
| `ip_admissions_study_period` | integer | Inpatient admissions from study start onward |

Quick check:

```python
import pandas as pd
df = pd.read_parquet("data/real_cohort_analytic.parquet")
assert len(df) == 5148
```

---

### `real_cohort_experiment_eligible.csv.gz`

Post-hold-out sampling frame: 5,098 patients remaining after removing the 50-patient prompt-development hold-out (seed=0). This is the input to the controlled paired experiment.

Column names are remapped for compatibility with the experiment runner:
- `patient_id` → `member_id`
- `charlson_proxy` → `charlson`
- `sdoh_count` → `social_needs`
- `race_ethnicity` → `race`

Quick check:

```python
import pandas as pd
df = pd.read_csv("data/real_cohort_experiment_eligible.csv.gz")
assert len(df) == 5098
```

---

### `prompt_dev_exclusion.txt`

50 patient IDs drawn from the analytic cohort before any model inference (seed=0). These patients are excluded from all confirmatory analyses. This file is a pre-registration record.

---

### `real_cohort_summary.json`

Aggregate cohort statistics used by `scripts/build_peer_review_artifacts.py` to generate Table 1. Contains no patient-level data.

---

## PHI-protected source tables (not in this directory)

The following source tables are required to run `scripts/build_real_cohort.py`. They are stored at `/Users/sanjaybasu/waymark-local/data/real_inputs/` under an institutional data use agreement and are not included in this repository.

| File | Description |
|---|---|
| `eligibility.csv` | Member eligibility records (173,931 rows) |
| `member_status_event.parquet` | Member activation status events |
| `member_patient_map.csv` | Member-to-patient identifier crosswalk |
| `member_goals.parquet` | Care management goals |
| `encounters.parquet` | Clinical encounters |
| `hospital_visits.parquet` | Hospital visit records |
| `outcomes_monthly.parquet` | Monthly outcome measures |
