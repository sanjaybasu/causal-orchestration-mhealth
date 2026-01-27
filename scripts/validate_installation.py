"""
Validation script to test installation and API connectivity.
Run this after installation to verify everything is set up correctly.
"""

import sys
import os

def check_python_version():
    """Check Python version is 3.10+"""
    print("Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"  ❌ Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies():
    """Check required packages are installed"""
    print("\nChecking dependencies...")
    required = [
        'numpy', 'pandas', 'scipy', 'matplotlib',
        'sklearn', 'litellm', 'tqdm'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    return True


def check_api_keys():
    """Check API keys are configured"""
    print("\nChecking API keys...")
    keys = {
        'OPENAI_API_KEY': 'OpenAI (GPT-5.2)',
        'ANTHROPIC_API_KEY': 'Anthropic (Claude)',
        'DEEPSEEK_API_KEY': 'DeepSeek (DeepSeek-V3)',
        'GOOGLE_API_KEY': 'Google (Gemini)'
    }

    configured = []
    for key, name in keys.items():
        if os.getenv(key):
            print(f"  ✅ {name}")
            configured.append(key)
        else:
            print(f"  ⚠️  {name} (not configured)")

    if not configured:
        print("\n  No API keys found. Set environment variables or use .env file")
        print("  Example: export OPENAI_API_KEY='your-key-here'")
        return False
    return True


def check_source_files():
    """Check source files exist"""
    print("\nChecking source files...")
    files = [
        'src/care_plan_generator.py',
        'src/conversational_analysis.py',
        'src/fairness_metrics.py'
    ]

    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            all_exist = False

    return all_exist


def test_basic_import():
    """Test basic imports work"""
    print("\nTesting basic imports...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.care_plan_generator import RetrospectiveEvaluator
        print("  ✅ RetrospectiveEvaluator import")

        from src.conversational_analysis import ConversationalBehaviorAnalyzer
        print("  ✅ ConversationalBehaviorAnalyzer import")

        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.care_plan_generator import RetrospectiveEvaluator
        import pandas as pd
        import numpy as np

        # Create mock data
        df = pd.DataFrame({
            'baseline_risk': np.random.rand(10),
            'charlson': np.random.randint(0, 5, 10),
            'social_needs': np.random.randint(0, 3, 10)
        })

        # Test evaluation
        evaluator = RetrospectiveEvaluator()
        safety, efficiency, equity = evaluator.evaluate_nash_orchestration(df)

        assert len(safety) == 10, "Safety scores incorrect length"
        assert len(efficiency) == 10, "Efficiency scores incorrect length"
        assert len(equity) == 10, "Equity scores incorrect length"

        print("  ✅ RetrospectiveEvaluator functional")
        return True
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False


def main():
    """Run all validation checks"""
    print("=" * 70)
    print("INSTALLATION VALIDATION")
    print("=" * 70)

    checks = [
        check_python_version(),
        check_dependencies(),
        check_api_keys(),
        check_source_files(),
        test_basic_import(),
        test_basic_functionality()
    ]

    print("\n" + "=" * 70)
    if all(checks):
        print("✅ ALL CHECKS PASSED - Installation is ready!")
        print("\nNext steps:")
        print("  1. Run: python scripts/run_retrospective_evaluation.py")
        print("  2. Run: python scripts/run_conversational_analysis.py")
        print("  3. See examples/quickstart.ipynb for interactive tutorial")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        return 1
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
