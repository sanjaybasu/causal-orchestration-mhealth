# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-26

### Added
- **Conversational Behavior Analysis Module** (`src/conversational_analysis.py`)
  - ConversationalBehaviorAnalyzer class for LLM-as-judge classification
  - PersonalityDiversityAnalyzer using BFI-10 framework
  - ExpertiseDiversityAnalyzer using semantic embeddings
  - Validation against human raters (ICC=0.76)

- **New Analysis Scripts**
  - `scripts/run_conversational_analysis.py` - Generate Table 2 results
  - `scripts/run_complete_pipeline.py` - Full reproducible pipeline
  - `scripts/validate_installation.py` - Installation verification

- **Comprehensive Documentation**
  - `docs/METHODS.md` - Detailed methodology documentation
  - `docs/INSTALLATION.md` - Step-by-step installation guide
  - `docs/USAGE.md` - Comprehensive usage examples
  - `docs/API.md` - Complete API reference
  - `docs/FAQ.md` - Troubleshooting and common questions

- **Testing Infrastructure**
  - Unit tests for conversational analysis
  - Unit tests for care plan generator
  - Unit tests for fairness metrics
  - pytest configuration with coverage reporting

- **Examples and Tutorials**
  - `examples/quickstart.ipynb` - Interactive Jupyter tutorial
  - `examples/example_transcripts/` - Annotated negotiation examples
  - `examples/custom_agent_prompts.py` - Agent customization guide

- **Package Infrastructure**
  - `setup.py` for pip installation
  - `requirements.txt` with all dependencies
  - MIT License with citation requirement
  - GitHub badges and comprehensive README

### Changed
- **README.md** - Completely rewritten with:
  - Societies of thought framework explanation
  - Key findings highlighted (4.7×, 8.0×, 4.3×, 3.2× improvements)
  - Detailed reproduction instructions
  - LLM-as-judge usage examples
  - Configuration and customization guides

- **Repository Structure** - Reorganized for better scientific reproducibility:
  - Separated `src/` (source code) from `scripts/` (executables)
  - Added `docs/`, `tests/`, and `examples/` directories
  - Created `results/` directory structure for outputs

### Fixed
- None (initial release with all features)

## [0.9.0] - 2025-12-15 (Pre-print Version)

### Added
- Initial implementation of multi-agent Nash bargaining system
- Core agent personas (Safety, Efficiency, Equity)
- Retrospective evaluation pipeline
- Performance metrics (Safety, Efficiency, Equity scores)
- Basic visualization (radar charts)
- Synthetic data generation

### Known Limitations (addressed in v1.0.0)
- No conversational behavior analysis
- Limited documentation
- No testing infrastructure
- Missing personality/expertise diversity metrics

---

## Future Roadmap

### [1.1.0] - Planned Features
- [ ] Support for additional agent personas (Cost Agent, Patient Preference Agent)
- [ ] Alternative LLM-as-judge models (Claude, GPT-4o)
- [ ] Real-time conversational monitoring dashboard
- [ ] Enhanced fairness metrics (intersectional equity)
- [ ] Docker containerization for reproducibility
- [ ] Integration with popular EHR systems (FHIR API)

### [1.2.0] - Planned Features
- [ ] Extension to diagnostic reasoning tasks
- [ ] Extension to treatment selection tasks
- [ ] Multi-language support (Spanish, Mandarin)
- [ ] Active learning for agent weight optimization
- [ ] Web-based demo interface

### [2.0.0] - Major Redesign (Future)
- [ ] Agent-based simulation environment
- [ ] Reinforcement learning for agent behavior
- [ ] Integration with clinical trial matching
- [ ] HIPAA-compliant hosted API service

---

## Version History

| Version | Date       | Highlights |
|---------|------------|------------|
| 1.0.0   | 2026-01-26 | Conversational analysis, full reproducibility |
| 0.9.0   | 2025-12-15 | Initial pre-print release |

---

## Upgrade Guide

### From 0.9.0 to 1.0.0

**Breaking Changes**: None

**New Features Available**:
```python
# Conversational behavior analysis (NEW in 1.0.0)
from src.conversational_analysis import ConversationalBehaviorAnalyzer

analyzer = ConversationalBehaviorAnalyzer()
behaviors = analyzer.analyze_conversational_behaviors(transcript)
```

**Deprecated**: None

**Migration Steps**:
1. Update dependencies: `pip install -r requirements.txt`
2. No code changes required for existing usage
3. Optional: Add conversational analysis to your workflow

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and development setup.

---

## Citation

If you use this software, please cite:

```bibtex
@article{basu2026causal,
  title={Causal Orchestration of Frontier Large Language Agents for Automated
         Care Plan Generation: Resolving the Quality-Efficiency-Equity Paradox},
  author={Basu, Sanjay and Baum, Aaron},
  journal={BMJ Health \& Care Informatics},
  year={2026},
  note={In press}
}
```
