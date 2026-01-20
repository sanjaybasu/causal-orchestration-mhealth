# Causal Orchestration of Frontier Large Language Models for Automated Care Plan Generation

This repository contains the reproduction code for the study **"Causal Orchestration of Frontier Large Language Agents for Automated Care Plan Generation: Resolving the Quality-Efficiency-Equity Paradox"**.

## Overview

The goal of this project is to evaluate a **Multi-Agent Generative Orchestration** system for creating clinical care plans using a **Retrospective Observational** design. The system uses a **Nash Bargaining** mechanism to negotiate between three competing objectives:
1.  **Safety**: Maximizing clinical risk coverage (DeepSeek-V3 proxy).
2.  **Efficiency**: Maximizing brevity and usability (Claude 3.5 Sonnet proxy).
3.  **Equity**: Maximizing social needs integration (GPT-5.2 proxy).

## Repository Structure

```
.
├── data/
│   └── real_world_cohort_sample.csv.gz  # Sample of de-identified profiles for reproducibility (N=127k cohort used in study)
├── src/
│   ├── care_plan_generator.py   # Core logic for Agent Classes and Nash Bargaining
│   ├── run_full_analysis.py     # Main script to reproduce Table 1 (Simulation)
│   └── generate_figures.py      # Script to generate Figure 2 (Radar Chart)
├── results/
│   └── generative_analysis_results.json # Output of the analysis
└── README.md
```

## Installation

```bash
pip install numpy pandas matplotlib scipy
```

## Reproduction Steps

1.  **Run the Retrospective Evaluation**:
    This script executes the evaluation for the full cohort using the 5 defined strategies (Control, Safety Only, Efficiency Only, Equity Only, Nash).
    ```bash
    python src/run_full_analysis.py
    ```
    *Output:* Prints the "Table 1" metrics to valid standard output and saves `results/generative_analysis_results.json`.

2.  **Generate Figures**:
    (Optional) Generate the Radar Chart (Figure 2).
    ```bash
    python src/generate_figures.py
    ```

## Contributors

*   **Sanjay Basu, MD, PhD**
*   **Aaron Baum, PhD**

## License

MIT License.
