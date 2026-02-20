# Experiment Guide

This guide covers the controlled-paired experiment runner and artifact build workflow.

## Canonical runner

Use `run_ollama_open_source_experiments.py` for controlled paired experiments.

```bash
cd causal_orchestration_mhealth
```

Run a pilot (N=10):

```bash
python run_ollama_open_source_experiments.py \
  --n-patients 10 \
  --cohort-path data/real_cohort_experiment_eligible.csv.gz \
  --profile legacy_local \
  --seed 42 \
  --judge-retries 4 \
  --condition-retries 1 \
  --timeout-seconds 90 \
  --max-retries 1 \
  --run-name pilot_n10
```

Run the full paired experiment (N=200):

```bash
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

Progress is printed to stdout. Capture with `tee`:

```bash
python run_ollama_open_source_experiments.py ... | tee results/run.log
```

## Cohort path

Always pass `--cohort-path` to avoid ambiguity:

```
--cohort-path data/real_cohort_experiment_eligible.csv.gz   # de-identified real cohort (N=5,098)
```

The cohort file must contain columns: `member_id`, `age`, `charlson`, `social_needs`, `race`, `female`, `baseline_risk`. Optional columns `sdoh_categories` and `clinical_categories` are used for enriched patient context when present.

## Manuscript table generation

Build table artifacts from a completed run:

```bash
python scripts/build_peer_review_artifacts.py \
  --real-cohort-path data/real_cohort_analytic.parquet \
  --controlled-run-name real_cohort_n200_seed42 \
  --ongoing-run-name pilot_peerready_n10_20260217
```

## Optional SOTA open-source profile

The `sota_open_source` profile requires pulling additional models before execution:

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
