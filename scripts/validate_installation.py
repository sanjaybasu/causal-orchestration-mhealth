#!/usr/bin/env python3
"""Validate environment setup for running the experiment."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def check_python_version() -> bool:
    print("Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"  FAIL: Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"  OK: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies() -> bool:
    print("\nChecking Python dependencies...")
    required = ["numpy", "pandas", "scipy", "matplotlib", "tqdm"]
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  OK: {package}")
        except ImportError:
            print(f"  FAIL: {package}")
            missing.append(package)

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    return True


def check_ollama() -> bool:
    print("\nChecking Ollama...")
    if not shutil.which("ollama"):
        print("  FAIL: ollama not found on PATH")
        print("  Install from https://ollama.com")
        return False
    print("  OK: ollama found")

    models_needed = ["deepseek-r1:8b", "llama3.1:8b"]
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        available = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  WARN: could not list Ollama models (is the server running?)")
        return True  # non-fatal

    all_found = True
    for model in models_needed:
        tag = model.split(":")[0]
        if tag in available:
            print(f"  OK: {model}")
        else:
            print(f"  MISSING: {model} (run: ollama pull {model})")
            all_found = False
    return all_found


def check_source_files() -> bool:
    print("\nChecking source files...")
    files = [
        "run_ollama_open_source_experiments.py",
        "src/care_plan_generator.py",
        "scripts/build_peer_review_artifacts.py",
        "scripts/generate_submission_figures.py",
    ]
    all_exist = True
    for f in files:
        path = PROJECT_ROOT / f
        if path.exists():
            print(f"  OK: {f}")
        else:
            print(f"  FAIL: {f}")
            all_exist = False
    return all_exist


def check_data_files() -> bool:
    print("\nChecking data files...")
    data_dir = PROJECT_ROOT / "data"
    files = [
        "real_cohort_experiment_eligible.csv.gz",
        "real_cohort_analytic.parquet",
    ]
    all_exist = True
    for f in files:
        path = data_dir / f
        if path.exists():
            print(f"  OK: data/{f}")
        else:
            print(f"  MISSING: data/{f}")
            all_exist = False
    return all_exist


def test_basic_import() -> bool:
    print("\nTesting imports...")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.care_plan_generator import RetrospectiveEvaluator  # noqa: F401
        print("  OK: RetrospectiveEvaluator import")
        return True
    except Exception as e:
        print(f"  FAIL: import failed: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("INSTALLATION VALIDATION")
    print("=" * 60)

    checks = [
        check_python_version(),
        check_dependencies(),
        check_ollama(),
        check_source_files(),
        check_data_files(),
        test_basic_import(),
    ]

    print("\n" + "=" * 60)
    if all(checks):
        print("ALL CHECKS PASSED")
        print("\nTo run the experiment:")
        print("  python run_ollama_open_source_experiments.py \\")
        print("    --n-patients 200 \\")
        print("    --cohort-path data/real_cohort_experiment_eligible.csv.gz \\")
        print("    --profile legacy_local --seed 42 \\")
        print("    --run-name real_cohort_n200_seed42")
    else:
        print("SOME CHECKS FAILED -- see above")
        return 1
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
