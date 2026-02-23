#!/bin/bash
# finalize_after_experiment.sh
# Run this once real_cohort_n200_seed42 reaches 200/200 complete pairs.
# It rebuilds all manuscript tables, updates the manuscript and appendix numbers,
# and regenerates figures. Run from the project root directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="real_cohort_n200_seed42"
# Set MANUSCRIPT and APPENDIX paths. Override with environment variables if needed.
MANUSCRIPT="${MANUSCRIPT:-$PROJECT_ROOT/../../../notebooks/causal_orchestration_mhealth_submission/submission_package_2026-02-18/manuscript.md}"
APPENDIX="${APPENDIX:-$PROJECT_ROOT/../../../notebooks/causal_orchestration_mhealth_submission/submission_package_2026-02-18/appendix.md}"

cd "$PROJECT_ROOT"

# Step 1: Verify 200/200 complete pairs
COMPLETE=$(python3 -c "
import pandas as pd
df = pd.read_csv('results/${RUN_NAME}_results.csv')
nash = set(df[df.condition=='nash']['patient_id'])
base = set(df[df.condition=='compute_matched']['patient_id'])
print(len(nash & base))
")
echo "Complete pairs: $COMPLETE/200"
if [ "$COMPLETE" -lt 200 ]; then
    echo "ERROR: Experiment not yet complete ($COMPLETE/200 pairs). Aborting."
    exit 1
fi

# Step 2: Rebuild manuscript tables
echo "Building manuscript tables..."
python3 scripts/build_peer_review_artifacts.py \
    --cohort-path data/real_cohort_experiment_eligible.csv.gz \
    --real-cohort-path data/real_cohort_analytic.parquet \
    --controlled-run-name "$RUN_NAME" \
    --ongoing-run-name pilot_peerready_n10_20260217

# Step 3: Rebuild distribution report
echo "Building distribution report..."
python3 - <<'PY'
import json, pandas as pd, numpy as np
from pathlib import Path
run = 'real_cohort_n200_seed42'
df = pd.read_csv(f'results/{run}_results.csv')
metrics = ['safety','efficiency','equity','composite']
# Deduplicate: take mean per (patient_id, condition) in case of restart duplicates
df = df.groupby(['patient_id','condition'], as_index=False)[metrics].mean()
n = df[df.condition=='nash'][['patient_id']+metrics].set_index('patient_id')
b = df[df.condition=='compute_matched'][['patient_id']+metrics].set_index('patient_id')
common = n.index.intersection(b.index)
d = n.loc[common] - b.loc[common]
out = {'run_name': run, 'paired_n': int(len(d))}
for k in metrics:
    out[k] = {
        'nash_mean': float(n.loc[common,k].mean()),
        'compute_mean': float(b.loc[common,k].mean()),
        'delta_mean': float(d[k].mean()),
    }
Path(f'results/{run}_distribution_report.json').write_text(json.dumps(out, indent=2))
print(f'Written results/{run}_distribution_report.json')
PY

# Step 4: Regenerate figures
echo "Regenerating figures..."
python3 scripts/generate_submission_figures.py

# Step 5: Print summary for manual manuscript update
echo ""
echo "=== NUMBERS FOR MANUSCRIPT UPDATE ==="
python3 - <<'PY'
import pandas as pd, numpy as np
from scipy import stats
df = pd.read_csv('results/real_cohort_n200_seed42_results.csv')
metrics = ['safety','efficiency','equity','composite']
# Deduplicate: take mean per (patient_id, condition) in case of restart duplicates
df = df.groupby(['patient_id','condition'], as_index=False)[metrics].mean()
n_df = df[df.condition=='nash'][['patient_id']+metrics].set_index('patient_id')
b_df = df[df.condition=='compute_matched'][['patient_id']+metrics].set_index('patient_id')
common = n_df.index.intersection(b_df.index)
print(f"\nN complete pairs: {len(common)}")
print("\nCondition means (95% CI):")
for cond, sdf in [('Nash',n_df.loc[common]),('Baseline',b_df.loc[common])]:
    for m in ['safety','efficiency','equity','composite']:
        mn = sdf[m].mean(); se = sdf[m].std()/np.sqrt(len(sdf))
        print(f"  {cond} {m}: {mn:.3f} ({mn-1.96*se:.3f}–{mn+1.96*se:.3f})")
d = n_df.loc[common] - b_df.loc[common]
print("\nPaired differences:")
for m in ['safety','efficiency','equity','composite']:
    delta = d[m]; se = delta.std()/np.sqrt(len(delta))
    t, p = stats.ttest_1samp(delta, 0)
    cohed = delta.mean()/delta.std()
    print(f"  {m}: {delta.mean():.3f} (95%CI {delta.mean()-1.96*se:.3f} to {delta.mean()+1.96*se:.3f}) t={t:.3f} p={p:.2e} d={cohed:.3f}")
PY

echo ""
echo "=== DONE ==="
echo "1. Update Tables 3-4 in $MANUSCRIPT with the numbers above."
echo "2. Update the Abstract Results paragraph with final Composite, Safety, Efficiency, Equity numbers."
echo "3. Update Appendix A8 table and summary statistics."
echo "4. git add + commit updated manuscript files."
echo "5. Push to GitHub."
