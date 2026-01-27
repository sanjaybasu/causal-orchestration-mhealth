# Causal Orchestration of Frontier Large Language Agents for Automated Care Plan Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


This project demonstrates how explicit multi-agent orchestration operationalizes "societies of thought"—internal multi-agent dialogues observed in frontier reasoning models—through an explicit architecture that provides interpretability, controllability, and structural safety guarantees for clinical deployment.

The system uses **Nash Bargaining** to coordinate three specialized agents:
1. **Safety Agent** (DeepSeek-V3): Maximizes clinical risk coverage and guideline adherence
2. **Efficiency Agent** (Claude 3.5 Sonnet): Maximizes brevity and workflow usability
3. **Equity Agent** (GPT-5.2): Maximizes social determinants of health integration

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


## Configuration

### API Keys

Set environment variables for LLM APIs:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"  
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
  year={2026},
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


## License

MIT License. See [LICENSE](LICENSE) for details.


