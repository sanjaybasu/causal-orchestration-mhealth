# Statistical analysis plan v1: three-arm paired experiment

Locked 2026-10-16. This plan is fixed before any model call for the three-arm run. The code that implements it is `packaging/causal_orchestration_mhealth/scripts/run_r1_three_arm.py` on branch `pdig-r1`. The commit hash is recorded in `audit/sap_lock.txt`.

## 1. Patients
Use the same 200 patients as the February 2026 paired run. They were drawn with seed 42 from the 5,098 experiment-eligible patients, after the 50-patient prompt-development hold-out (seed 0). Use the frozen ID list `data/r1_patient_ids.txt`, which stays local only.

## 2. Arms
Every arm uses the same models and makes the same number of calls:
- 1 draft by the generator
- 1 selection-judge call that scores the draft
- 3 rounds of 3 critiques and 1 synthesis; the third round synthesis gives K = 3 candidates
- 3 selection-judge calls, one for each candidate
- 1 outcome-judge call on the selected plan

Each arm therefore makes 15 generation calls, 4 selection-judge calls, and 1 outcome-judge call.

| Arm | Critiques | Selection rule |
|---|---|---|
| N (Nash) | Role-specialized: the safety critic is deepseek-r1:8b; the efficiency and equity critics are llama3.1:8b, each with its role system prompt | Nash product |
| B (self-critique) | Generic: three critiques by llama3.1:8b with one generic reviewer prompt | Nash product |
| A (additive) | Role-specialized, the same as N | Arithmetic mean |

Rules that apply to all arms:
- The generator and the synthesizer are llama3.1:8b. All arms use the same synthesis prompt, which carries no weights.
- Generation and critique calls use temperature 0.2. The three candidates use temperature 0.7, with seeds 1, 2, and 3.
- Judge calls use temperature 0.0.
- Each arm runs once per patient. A repeated (patient, arm) pair stops the run.
- The order of the arms is randomized per patient (seed 42 + patient index), so that time drift does not favour one arm.

## 3. Nash-product selection
- The selection judge is llama3.1:8b. It uses the same rubric as the outcome judge.
- Let u_k(c) be the score of candidate c in domain k, for k in {safety, efficiency, equity}. The disagreement point is d_k = 0 ("no plan"), and the weights are α_k = 1/3.
- Arms N and B select the candidate that maximizes ∏_k max(u_k(c), 0.01)^{1/3}, which is the geometric mean of the three domain scores.
- Arm A selects the candidate with the highest arithmetic mean of u_k.
- Ties go to the candidate with the lowest seed.
- The selection judge also scores the draft. This score is descriptive: `below_disagreement` records that no candidate scored at or above the draft in all three domains.

A one-patient smoke test on 2026-10-16 used the draft score as d_k and showed that this choice does not work. The Llama selection judge gave the draft an Equity score of 1.0, so every candidate's product was at the floor, and selection became arbitrary. The rule above was fixed before the confirmatory run. The smoke-test outputs are not analysed.

## 4. Outcomes
- **Primary outcome judge:** deepseek-r1:8b. It uses the same judge prompt as the February run (`evaluate_plan`), with think disabled.
- **Secondary outcome judges:** gpt-oss:20b and llama3.1:8b rescore every selected plan after generation. Llama is the family of the generator and the selection judge, so its scores can show self-preference.
- **Metrics:** Safety, Efficiency, Equity, and Composite, where Composite is the mean of the three.

## 5. Contrasts and inference
- **Primary contrast:** N − B, Composite, primary judge.
- **Secondary contrasts:** N − B for each domain, and N − A for all four metrics.
- **Tests:** a paired t-test with a 95% CI, paired Cohen's d, and a Wilcoxon signed-rank sensitivity test.
- **Win/tie/loss:** a difference with |Δ| < 0.005 counts as a tie. Report counts and percentages out of 200.
- **Multiplicity:** Holm correction across the 7 secondary tests, reported next to the unadjusted p-values.

## 6. Subgroups
The subgroups are descriptive only; the study has no power for them.
- **Variables:** race and ethnicity (Black, White, all other); state (VA, WA); any social need (yes, no); social-needs count (0, 1, ≥2).
- **Estimates:** the N − B difference with its 95% CI, for Equity and Composite, within each subgroup level.
- **Interaction test:** a linear model Δ_i ~ subgroup, with an F-test.

## 7. Judge agreement
- ICC(2,1) between each pair of judges, for each domain, across all 600 plans.
- The rate at which the judges agree on the sign of N − B for each patient.
- The N − B effect under each judge.

## 8. Blinded human review
- **Sample:** 100 patients drawn from the 200 with seed 2026; the N plan and the B plan for each.
- **Raters:** 3 (a physician, a Waymark clinician, and a care manager). Each rater sees the two plans as "Plan 1" and "Plan 2" in random order, with no arm or model labels.
- **Items:** 1–5 anchored scales for Safety, Efficiency, and Equity, rescaled as (x − 1)/4; a forced preference; a flag for an unsafe or wrong action, with free text; and "usable with minor edits" (yes/no).
- **Calibration:** 5 practice pairs from the remaining 100 patients, not analysed.
- **Analysis:**
  - ICC(2,k) across the human raters for each domain
  - ICC between the mean human score and each LLM judge
  - the human-rated N − B difference with a CI clustered by patient
  - the preference rate for N with a Wilson CI
  - the unsafe-flag rate per arm, with a McNemar test

## 9. Secondary analysis of the February run
Report the February two-arm run in S1 Appendix. In that run, the two arms used different numbers of calls, and repeat generations for 121 patients were averaged. Report the averaged result and a keep-first sensitivity analysis.

## Addendum 1 (2026-10-16, before secondary-judge rescoring)
In Ollama 0.13.1, gpt-oss:20b reasons even when the request sets think to false. Its reasoning counts against the 220-token output limit. For gpt-oss only, the secondary-judge rescoring uses reasoning effort "low" and a 1,024-token output limit. The judge prompt, temperature 0.0, and parsing are unchanged. To support this setting, `OllamaClient.chat` and `evaluate_plan` gained optional `think` and `num_predict` arguments. Their defaults reproduce the earlier behaviour exactly, and the running generation processes loaded the module before this change.

## Addendum 2 (2026-10-16, before any human rating)
The three reviewers are physicians: Sanjay Basu (internal medicine; an author), John Morgan (internal medicine), and Christian Moher (family medicine). This replaces the planned mix of a physician, a clinician, and a care manager. Because one reviewer is an author, the analysis also reports ICC among the human reviewers, the human–judge agreement, and the human-rated N − B difference with that reviewer's ratings excluded.

## Addendum 3 (2026-10-16, before any three-arm result is viewed)
This addendum adds a secondary descriptive comparison: one draft with no critique. Each arm writes and stores its own draft (`draft` in the trace). The primary outcome judge (deepseek-r1:8b, same prompt) scores the role-specialized Nash arm's draft for each patient. We report the Nash final plan minus its own draft, for each metric, with a paired t-test and 95% CI. This contrast measures what the critique and selection steps add to a single generation pass. It is not compute-matched and is not a test of the study hypotheses.
