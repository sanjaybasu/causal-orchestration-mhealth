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

## PLOS Digital Health revision (tag pdig-r1-submission)

The revised manuscript reports a compute-matched three-arm experiment. The statistical analysis plan, with its addenda, is in `docs/SAP_v1.md`; `docs/sap_lock.txt` gives the file hashes recorded before each stage. All model calls use Ollama 0.13.1 (`--ollama-base-url`, default `http://127.0.0.1:11435`) with `deepseek-r1:8b` (digest 6995872bfe4c), `llama3.1:8b` (digest 46e0c10c039e), and `gpt-oss:20b`.

Run order: `scripts/run_r1_three_arm.py` → `scripts/r1_rescore_judges.py` → `scripts/r1_single_pass.py` → analysis scripts. `scripts/r1_finalize.sh` runs the post-generation steps in sequence.

| Manuscript item | Script | Output (in `results/`) |
|---|---|---|
| Table 1 (cohort and sample) | `scripts/r1_table1.py` | `r1_analysis/table1.json` (needs the protected cohort file) |
| S1 Appendix Table A4 (cohort flow) | `scripts/build_real_cohort.py` | build log (needs the protected source tables) |
| Table 2 (model calls per arm) | `scripts/run_r1_three_arm.py` (`calls` per row) | `r1_three_arm_n200_shard*of3_results.csv` columns `n_*_calls` |
| Arm means, paired differences, higher/same/lower counts, Holm-adjusted p-values (Results; Tables 3–4) | `scripts/r1_paired_analyses.py --three-arm r1_three_arm_n200` | `r1_analysis/r1_three_arm_n200_paired_analyses.json` |
| Fig 1 (pipeline) | `scripts/r1_figures.py` (`fig1`) | `Fig1.tif` |
| Fig 2 (distribution of paired differences) | `scripts/r1_figures.py --run r1_three_arm_n200` | `Fig2.tif` |
| Fig 3 and S1 Appendix A9 (subgroups) | `scripts/r1_paired_analyses.py`; `scripts/r1_figures.py` | `..._paired_analyses.json` (`N-B.subgroups`); `Fig3.tif` |
| S1 Appendix A8 (judge agreement) | `scripts/r1_rescore_judges.py --analyze` | `r1_analysis/r1_three_arm_n200_judge_agreement.json` |
| Blinded physician review (Results; A8) | `scripts/r1_build_human_review.py`; `scripts/r1_human_review_analysis.py` | `r1_analysis/r1_three_arm_n200_human_review.json` |
| Single-pass reference (SAP Addendum 3a) | `scripts/r1_single_pass.py` | `r1_analysis/r1_three_arm_n200_single_pass.json` |
| S1 Appendix A5 (prompts) | `scripts/r1_dump_prompts.py` | `r1_analysis/prompts_verbatim.md` |
| S1 Appendix A10 (two-arm experiment) | `scripts/r1_paired_analyses.py --feb` | `r1_analysis/feb_paired_analyses.json` |
| S2 Appendix (worked examples) | `scripts/r1_select_examples.py` | local draft only (contains patient context) |

The public score file (S1 Data) has random study numbers in place of patient identifiers. With it, the paired, subgroup, judge-agreement, human-review, and single-pass analyses can be re-run. Table 1, Table A4, and the S2 examples need the protected data and are not reproducible from public files.
