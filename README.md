# causal_orchestration_mhealth

Code and reproducible artifacts for evaluating multi-agent care-plan generation with locally hosted open-source models.

## Study

A controlled paired experiment comparing Nash-orchestrated multi-agent language models against a compute-matched sequential self-critique baseline for Medicaid care plan generation. Analytic cohort: 5,148 activated Medicaid care management patients in Virginia and Washington. Confirmatory experiment: 200 patients, seed 42.

## Current artifacts (February 2026)

1. Real cohort derivation: `data/real_cohort_analytic.parquet` (N=5,148), built from PHI-protected source tables by `scripts/build_real_cohort.py`.
2. Controlled paired run: `real_cohort_n200_seed42`, 200 complete pairs.
3. Pilot paired run: `pilot_peerready_n10_20260217`, 10 complete pairs.

## Canonical commands

Run the controlled experiment:

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

Progress is printed to stdout. Capture with `tee`:

```bash
python run_ollama_open_source_experiments.py ... | tee results/run.log
```

Build manuscript-ready tables from completed artifacts:

```bash
python scripts/build_peer_review_artifacts.py \
  --real-cohort-path data/real_cohort_analytic.parquet \
  --controlled-run-name real_cohort_n200_seed42 \
  --ongoing-run-name pilot_peerready_n10_20260217
```

## Output package

```
results/real_cohort_n200_seed42_config.json
results/real_cohort_n200_seed42_results.csv
results/real_cohort_n200_seed42_trace.jsonl
results/real_cohort_n200_seed42_summary.json
results/real_cohort_n200_seed42_distribution_report.json
results/writeup_package/table1_demographics.csv
results/writeup_package/table2_retrospective_performance.csv
results/writeup_package/table3_controlled_condition_performance.csv
results/writeup_package/table4_controlled_paired_differences.csv
results/writeup_package/table5_reproducibility_status.csv
results/writeup_package/manuscript_numbers.json
```

## API keys

Copy `.env.example` to `.env` and supply your own keys. The `.env` file is listed in `.gitignore` and must not be committed to public repositories. The `legacy_local` profile uses Ollama and does not require external API keys.

## Model profiles

`legacy_local` uses locally available Ollama models (deepseek-r1:8b, llama3.1:8b) and is reproducible without external API access.

`sota_open_source` requires pulling additional models via `ollama pull` before execution.

## Cohort derivation

`scripts/build_real_cohort.py` documents the full cohort derivation logic from seven linked source tables. The source tables are PHI-protected and not included in this repository. The derived de-identified cohort files (`data/real_cohort_analytic.parquet`, `data/real_cohort_experiment_eligible.csv.gz`) are the inputs to all analyses.
