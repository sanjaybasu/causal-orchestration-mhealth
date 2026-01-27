# Causal Orchestration of Frontier Large Language Agents for Automated Care Plan Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete reproduction code for the study **"Causal Orchestration of Frontier Large Language Agents for Automated Care Plan Generation: Resolving the Quality-Efficiency-Equity Paradox"** published in *BMJ Health & Care Informatics*.

## Overview

This project demonstrates how explicit multi-agent orchestration operationalizes "societies of thought"—internal multi-agent dialogues observed in frontier reasoning models—through an explicit architecture that provides interpretability, controllability, and structural safety guarantees for clinical deployment.

The system uses **Nash Bargaining** to coordinate three specialized agents:
1. **Safety Agent** (DeepSeek-V3): Maximizes clinical risk coverage and guideline adherence
2. **Efficiency Agent** (Claude 3.5 Sonnet): Maximizes brevity and workflow usability
3. **Equity Agent** (GPT-5.2): Maximizes social determinants of health integration

### Key Findings

- **Performance**: Composite Quality Score 0.822 vs 0.633 (best baseline), *p*<0.001
- **Conversational Behaviors**: 4.7× more question-answering, 8.0× more perspective shifts in Nash transcripts
- **Cognitive Diversity**: 4.3× higher personality diversity, 3.2× higher expertise diversity
- **Physician Preference**: 75% prefer Nash-generated plans (*N*=90, *p*=0.04)
- **Scale**: Validated on 127,801 Medicaid beneficiaries

## Repository Structure

```
causal_orchestration_mhealth/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── setup.py                              # Package installation
│
├── data/                                  # Data files
│   ├── synthetic_cohort_sample.csv.gz    # Synthetic patient cohort (N=1000 sample)
│   └── README_data.md                    # Data dictionary and generation details
│
├── src/                                   # Source code
│   ├── __init__.py
│   ├── care_plan_generator.py            # Core multi-agent system
│   ├── conversational_analysis.py        # Conversational behavior analysis (NEW)
│   ├── fairness_metrics.py               # Equity and fairness calculations
│   ├── generate_figures.py               # Figure generation
│   └── run_full_analysis.py              # Main analysis pipeline
│
├── scripts/                               # Executable scripts
│   ├── run_retrospective_evaluation.py   # Generate Table 1 results
│   ├── run_conversational_analysis.py    # Generate Table 2 results (NEW)
│   ├── run_complete_pipeline.py          # Run full reproducible pipeline (NEW)
│   └── validate_installation.py          # Test installation (NEW)
│
├── results/                               # Output directory
│   ├── table1_performance_metrics.csv
│   ├── table2_conversational_analysis.csv # NEW
│   ├── figures/
│   │   ├── figure1_architecture.png
│   │   └── figure2_radar_chart.png
│   └── logs/
│
├── tests/                                 # Unit tests (NEW)
│   ├── test_care_plan_generator.py
│   ├── test_conversational_analysis.py
│   └── test_fairness_metrics.py
│
├── examples/                              # Usage examples (NEW)
│   ├── quickstart.ipynb                  # Jupyter notebook tutorial
│   ├── example_transcripts/              # Sample agent negotiations
│   └── custom_agent_prompts.py           # How to customize agents
│
└── docs/                                  # Documentation (NEW)
    ├── INSTALLATION.md                   # Detailed installation guide
    ├── USAGE.md                          # Comprehensive usage guide
    ├── METHODS.md                        # Methodology documentation
    ├── API.md                            # API reference
    └── FAQ.md                            # Frequently asked questions
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sanjaybasu/causal-orchestration-mhealth.git
cd causal-orchestration-mhealth

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Basic Usage

```python
from src.care_plan_generator import RealCarePlanGenerator, RetrospectiveEvaluator
import pandas as pd

# Load synthetic patient data
patients = pd.read_csv('data/synthetic_cohort_sample.csv.gz')

# Generate care plans using Nash Orchestration
generator = RealCarePlanGenerator()
care_plan = generator.generate_nash(
    patient_context="62yo F with T2DM, HTN, CKD3, housing insecurity"
)

# Evaluate performance metrics
evaluator = RetrospectiveEvaluator()
safety, efficiency, equity = evaluator.evaluate_nash_orchestration(patients)

print(f"Safety Score: {safety.mean():.3f}")
print(f"Efficiency Score: {efficiency.mean():.3f}")
print(f"Equity Score: {equity.mean():.3f}")
```

### Reproduce Main Results

```bash
# Reproduce Table 1 (Performance Metrics)
python scripts/run_retrospective_evaluation.py

# Reproduce Table 2 (Conversational Behaviors & Diversity)
python scripts/run_conversational_analysis.py

# Generate all figures
python src/generate_figures.py

# Run complete reproducible pipeline
python scripts/run_complete_pipeline.py
```

## Detailed Reproduction Instructions

### Table 1: Performance Metrics (N=127,801)

The retrospective evaluation uses a time-split design where ground truth is defined by actual subsequent clinical outcomes.

```bash
python scripts/run_retrospective_evaluation.py \
    --cohort_path data/synthetic_cohort_sample.csv.gz \
    --output_path results/table1_performance_metrics.csv \
    --n_samples 1000
```

Expected output:
```
Strategy                    Safety    Efficiency   Equity   Composite
Control (Template)          0.500     0.500       0.300    0.433
Safety Agent Only           0.843     0.507       0.400    0.583
Efficiency Agent Only       0.598     0.900       0.300    0.599
Equity Agent Only           0.600     0.400       0.900    0.633
Multi-Agent (Nash)          0.916     0.650       0.900    0.822
```

### Table 2: Conversational Behaviors & Cognitive Diversity (N=500)

Analyzes agent negotiation transcripts for conversational behaviors and cognitive diversity.

```bash
python scripts/run_conversational_analysis.py \
    --n_transcripts 500 \
    --judge_model "gemini/gemini-2.0-flash-exp" \
    --output_path results/table2_conversational_analysis.csv
```

Expected output:
```
Metric                              Single-Agent   Nash    p-value
Question-Answering (%)              15.2          71.4    <0.001
Perspective Shifts (%)              8.1           65.2    <0.001
Conflict of Perspectives (%)        12.0          68.3    <0.001
Personality Diversity (Agree SD)    0.28          1.21    <0.001
Expertise Diversity (cosine dist)   0.22          0.71    <0.001
```

## Using LLM-as-Judge for Conversational Analysis

The conversational behavior analysis uses an LLM-as-judge approach validated against human raters (ICC=0.76).

```python
from src.conversational_analysis import (
    ConversationalBehaviorAnalyzer,
    PersonalityDiversityAnalyzer,
    ExpertiseDiversityAnalyzer
)

# Analyze conversational behaviors
behavior_analyzer = ConversationalBehaviorAnalyzer(
    judge_model="gemini/gemini-2.0-flash-exp"
)

transcript = """
Safety Agent: What about the rising creatinine trend?
Equity Agent: The patient missed nephrology due to transportation.
Safety Agent: We still need the nephrology referral.
Efficiency Agent: Let's arrange NEMT for the appointment.
"""

behaviors = behavior_analyzer.analyze_conversational_behaviors(transcript)
# Returns: {'question_answering': True, 'perspective_shift': True, ...}

# Analyze personality diversity
personality_analyzer = PersonalityDiversityAnalyzer()
safety_personality = personality_analyzer.analyze_agent_personality(
    agent_name="Safety Agent",
    agent_critiques=[...]
)
# Returns: {'extraversion': 2.5, 'agreeableness': 2.1, ...}

# Analyze expertise diversity
expertise_analyzer = ExpertiseDiversityAnalyzer()
diversity = expertise_analyzer.calculate_expertise_diversity([
    "Clinical guidelines and pharmaceutical safety",
    "Workflow optimization and care coordination",
    "Social determinants of health"
])
# Returns: 0.71 (mean cosine distance)
```

## Configuration

### API Keys

Set environment variables for LLM APIs:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"  # For Gemini LLM-as-judge
```

Or use a `.env` file (see `.env.example`).

### Agent Customization

Modify agent prompts and bargaining weights:

```python
from src.care_plan_generator import RealCarePlanGenerator

# Custom bargaining weights
generator = RealCarePlanGenerator(
    model_map={
        "safety": "deepseek-v3",
        "efficiency": "claude-3-5-sonnet",
        "equity": "gpt-5.2",
        "generator": "gpt-5.2"
    }
)

# Custom weights (default: α_safety=1.5, α_equity=1.0, α_efficiency=0.8)
care_plan = generator.generate_nash(
    patient_context="...",
    alpha_safety=2.0,      # Increase safety priority
    alpha_equity=1.5,      # Increase equity priority
    alpha_efficiency=0.5   # Decrease efficiency priority
)
```

## Data

### Synthetic Cohort

The repository includes a **synthetic dataset** (`data/synthetic_cohort_sample.csv.gz`) that preserves the statistical properties of the original cohort:

- **Size**: 1,000 patients (representative sample)
- **Features**: Demographics, diagnoses (Charlson index), social needs (housing, food, transportation), utilization (ED visits, hospitalizations)
- **Generation**: Synthetic Data Vault (SDV) with preserved correlations
- **Privacy**: No real patient data; all values are synthetically generated

See `data/README_data.md` for complete data dictionary.

### Real Data Access

The original cohort (N=127,801) contains protected health information and cannot be shared. Researchers seeking access to similar data should:

1. Contact Waymark at research@waymarkcare.com
2. Execute appropriate data use agreements
3. Obtain IRB approval for retrospective analysis

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_conversational_analysis.py -v

# Run with coverage
pytest --cov=src tests/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{basu2026causal,
  title={Causal Orchestration of Frontier Large Language Agents for Automated Care Plan Generation: Resolving the Quality-Efficiency-Equity Paradox},
  author={Basu, Sanjay and Baum, Aaron},
  journal={BMJ Health \& Care Informatics},
  year={2026},
  note={In press}
}
```

Key references:

```bibtex
@article{kim2026societies,
  title={Reasoning Models Generate Societies of Thought},
  author={Kim, Junsol and Lai, Shiyang and Scherrer, Nino and Ag{\"u}era y Arcas, Blaise and Evans, James},
  journal={arXiv preprint arXiv:2601.10825},
  year={2026}
}

@article{nash1950bargaining,
  title={The Bargaining Problem},
  author={Nash, John F},
  journal={Econometrica},
  volume={18},
  number={2},
  pages={155--162},
  year={1950}
}
```

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[Usage Guide](docs/USAGE.md)**: Comprehensive usage examples
- **[Methods Documentation](docs/METHODS.md)**: Methodology details
- **[API Reference](docs/API.md)**: Complete API documentation
- **[FAQ](docs/FAQ.md)**: Common questions and troubleshooting

## Examples

### Jupyter Notebook Tutorial

See `examples/quickstart.ipynb` for an interactive tutorial covering:
- Loading synthetic patient data
- Generating care plans with different strategies
- Analyzing conversational behaviors
- Computing personality and expertise diversity
- Visualizing results

### Example Transcripts

The `examples/example_transcripts/` directory contains annotated examples of:
- Successful Nash negotiations with high conversational behavior
- Failed single-agent reasoning with minimal perspective diversity
- Edge cases and failure modes

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- Additional conversational behavior metrics
- Alternative LLM-as-judge models
- New agent personas (e.g., Cost Agent, Patient Preference Agent)
- Extensions to other clinical tasks (diagnosis, treatment selection)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Sanjay Basu, MD, PhD**: sanjay@waymarkcare.com
- **Aaron Baum, PhD**: aaron@waymarkcare.com
- **GitHub Issues**: https://github.com/sanjaybasu/causal-orchestration-mhealth/issues

## Acknowledgments

This work was supported by Waymark, a public benefit organization providing free health and social services to Medicaid beneficiaries.

We thank:
- The Kim et al. (2026) team for the "Societies of Thought" framework
- The open-source LLM community (DeepSeek, Anthropic, OpenAI)
- The BMJ Health & Care Informatics reviewers and editors

## Version History

- **v1.0.0** (January 2026): Initial release with conversational behavior analysis
- **v0.9.0** (December 2025): Pre-print version with performance metrics only

## Troubleshooting

### Common Issues

**LLM API errors**: Ensure API keys are set correctly
```bash
# Test API connectivity
python scripts/validate_installation.py
```

**Memory errors with large cohorts**: Use batch processing
```python
# Process in batches of 1000
for batch in range(0, len(patients), 1000):
    batch_patients = patients[batch:batch+1000]
    # ... process batch
```

**Slow conversational analysis**: Use faster judge model
```python
# Use Gemini Flash instead of Gemini Pro
analyzer = ConversationalBehaviorAnalyzer(
    judge_model="gemini/gemini-2.0-flash-exp"  # Faster
)
```

See [FAQ](docs/FAQ.md) for more troubleshooting tips.

---

**Ready to get started?** Run `python scripts/validate_installation.py` to verify your setup!
