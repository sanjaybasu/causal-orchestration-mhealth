# Causal Orchestration of Frontier Large Language Model Agents
## An Embedded Micro-Randomized Trial

This repository contains the replication code for the manuscript "Causal Orchestration of Frontier Large Language Model Agents Prevents Performance Degradation in Mobile Health Allocation: An Embedded Micro-Randomized Trial".

### Repository Structure

- `src/`: Source code for the multi-agent system and statistical analysis.
  - `run_full_analysis.py`: Main execution script. Orchestrates the agents and computes results.
  - `xlearner.py`: Implementation of the X-Learner metalearner for Heterogeneous Treatment Effect (HTE) estimation.
  - `fairness_metrics.py`: Modules for calculating Algorithmic Fairness metrics (Sensitivity, Calibration, Equalized Odds).
  - `propensity_score_analysis.py`: Propensity score estimation and Inverse Probability Weighting (IPW) logic.
  - `multi_agent_llm_system.py`: (Legacy) Standalone implementation of agent logic.
- `data/`:
  - `synthetic_cohort.csv.gz`: De-identified synthetic dataset matching the statistical properties of the Medicaid cohort (N=127,801).
- `results/`: Directory where analysis outputs (JSON results, figures) are saved.

### Installation

Requires Python 3.10+ and standard scientific libraries.

```bash
pip install numpy pandas scikit-learn xgboost scipy
```

### Reproducing Results

To replicate the manuscript's findings using the synthetic dataset:

```bash
python src/run_full_analysis.py
```

This will:
1. Load the synthetic cohort.
2. Simulate the three agents (Safety, Efficiency, Equity) using proxy logic derived from the frontier models.
3. Apply Nash Bargaining Orchestration to resolve conflicts.
4. Calculate outcomes (RRR, Contact Rate) for single-agent vs multi-agent strategies.
5. Compute Algorithmic Fairness metrics (Sensitivity by subgroup).
6. Save results to `results/real_analysis_results.json`.

### License

This code is provided for academic replication and validation purposes.
