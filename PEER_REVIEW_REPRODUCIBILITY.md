# Peer-Review Reproducibility Protocol

This protocol defines the commands for generating manuscript artifacts from local files.

## 1. Controlled experiment command (resumable)

```bash
cd causal_orchestration_mhealth

python run_ollama_open_source_experiments.py \
  --n-patients 200 \
  --cohort-path data/real_cohort_experiment_eligible.csv.gz \
  --profile legacy_local \
  --seed 42 \
  --judge-retries 4 \
  --condition-retries 1 \
  --timeout-seconds 90 \
  --max-retries 1 \
  --run-name real_cohort_n200_seed42
```

The run resumes automatically when interrupted. Results and traces are written incrementally.

## 2. Build manuscript table artifacts from completed files

```bash
python scripts/build_peer_review_artifacts.py \
  --real-cohort-path data/real_cohort_analytic.parquet \
  --controlled-run-name real_cohort_n200_seed42 \
  --ongoing-run-name pilot_peerready_n10_20260217
```

This writes:

```
results/writeup_package/table1_demographics.csv
results/writeup_package/table2_retrospective_performance.csv
results/writeup_package/table3_controlled_condition_performance.csv
results/writeup_package/table4_controlled_paired_differences.csv
results/writeup_package/table5_reproducibility_status.csv
results/writeup_package/manuscript_numbers.json
```

## 3. Build timing and score distribution artifact

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
import numpy as np

run = 'real_cohort_n200_seed42'
df = pd.read_csv(f'results/{run}_results.csv')
metrics = ['safety', 'efficiency', 'equity', 'composite']
n = df[df.condition == 'nash'][['patient_id'] + metrics]
b = df[df.condition == 'compute_matched'][['patient_id'] + metrics]
m = n.merge(b, on='patient_id', suffixes=('_nash', '_base'))

out = {'run_name': run, 'paired_n': int(len(m))}
for k in metrics:
    out[k] = {
        'nash_mean': float(m[f'{k}_nash'].mean()),
        'compute_mean': float(m[f'{k}_base'].mean()),
        'delta_mean': float((m[f'{k}_nash'] - m[f'{k}_base']).mean()),
    }
Path(f'results/{run}_distribution_report.json').write_text(json.dumps(out, indent=2))
PY
```

## 4. Required files for manuscript lock

```
results/real_cohort_n200_seed42_config.json
results/real_cohort_n200_seed42_results.csv
results/real_cohort_n200_seed42_trace.jsonl
results/real_cohort_n200_seed42_summary.json
results/real_cohort_n200_seed42_distribution_report.json
results/pilot_peerready_n10_20260217_config.json
results/pilot_peerready_n10_20260217_results.csv
results/pilot_peerready_n10_20260217_summary.json
results/writeup_package/table1_demographics.csv
results/writeup_package/table2_retrospective_performance.csv
results/writeup_package/table3_controlled_condition_performance.csv
results/writeup_package/table4_controlled_paired_differences.csv
results/writeup_package/table5_reproducibility_status.csv
results/writeup_package/manuscript_numbers.json
```

## 5. Optional open-source SOTA profile run

```bash
python run_ollama_open_source_experiments.py \
  --n-patients 200 \
  --cohort-path data/real_cohort_experiment_eligible.csv.gz \
  --profile sota_open_source \
  --seed 42 \
  --judge-retries 4 \
  --condition-retries 1 \
  --timeout-seconds 90 \
  --max-retries 1 \
  --auto-pull-missing \
  --run-name real_cohort_sota_oss_n200_seed42
```

Use this command after all required models are available in `ollama list`.
