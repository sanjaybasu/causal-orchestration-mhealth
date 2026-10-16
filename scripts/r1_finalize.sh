#!/bin/zsh
# Post-generation pipeline for the three-arm run (SAP v1). Waits for 600 rows, then runs every downstream step.
set -e
cd "$(dirname "$0")/.."
RUN=r1_three_arm_n200
R=/Users/sanjaybasu/waymark-local/notebooks/causal_orchestration_mhealth_submission/revision_1
count() { python3 -c "import pandas as pd,glob;print(sum(len(pd.read_csv(f)) for f in glob.glob('results/${RUN}_shard*of3_results.csv')))"; }
until [ "$(count)" -ge 600 ] || ! pgrep -f "run_r1_three_arm.py --run-name ${RUN}" >/dev/null; do sleep 300; done
echo "generation rows: $(count) at $(date)"
python3 scripts/r1_paired_analyses.py --three-arm $RUN
python3 scripts/r1_build_human_review.py --run $RUN --out $R/human_review
python3 scripts/r1_select_examples.py --run $RUN --out $R/supplement
python3 scripts/r1_dump_prompts.py > results/r1_analysis/prompts_verbatim.md
python3 scripts/r1_figures.py --run $RUN --out $R/figures
python3 scripts/r1_rescore_judges.py --run $RUN
python3 scripts/r1_rescore_judges.py --run $RUN --analyze
echo "finalize done at $(date)"
