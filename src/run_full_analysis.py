
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os
import sys

# Import modular components
try:
    from care_plan_generator import RetrospectiveEvaluator
    from fairness_metrics import calculate_sensitivity_by_group
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from care_plan_generator import RetrospectiveEvaluator
    from fairness_metrics import calculate_sensitivity_by_group

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "real_world_cohort_sample.csv.gz"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(2026)

def run_analysis():
    print(f"Loading Retrospective Cohort from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        n = len(df)
        print(f"Loaded {n} patient profiles.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    # -------------------------------------------------------------------------
    # 1. RETROSPECTIVE EVALUATION
    # -------------------------------------------------------------------------
    print("Running Retrospective Care Plan Evaluation...")
    
    eval_data = {}
    
    # Strategy 1: Control (Template)
    eval_data['Control (Template)'] = RetrospectiveEvaluator.evaluate_control_template(df)
    
    # Strategy 2: Safety Agent Only
    eval_data['Safety Agent Only'] = RetrospectiveEvaluator.evaluate_safety_agent(df)
    
    # Strategy 3: Efficiency Agent Only
    eval_data['Efficiency Agent Only'] = RetrospectiveEvaluator.evaluate_efficiency_agent(df)
    
    # Strategy 4: Equity Agent Only
    eval_data['Equity Agent Only'] = RetrospectiveEvaluator.evaluate_equity_agent(df)
    
    # Strategy 5: Multi-Agent (Nash)
    eval_data['Multi-Agent (Nash)'] = RetrospectiveEvaluator.evaluate_nash_orchestration(df)
    
    
    final_metrics = {}
    
    for strats, (s, e, q) in eval_data.items():
        # Composite score = unweighted average
        composite = (s + e + q) / 3.0
        
        final_metrics[strats] = {
            'Safety Score': float(s.mean()),
            'Efficiency Score': float(e.mean()),
            'Equity Score': float(q.mean()),
            'Composite Score': float(composite.mean())
        }
        
        # Add CIs (approximate)
        for metric, val in [('Safety', s), ('Efficiency', e), ('Equity', q)]:
            se = val.std() / np.sqrt(n)
            # Avoid nan for zero std
            if se == 0:
                se = 0.0
            final_metrics[strats][f'{metric} CI Lower'] = float(val.mean() - 1.96*se)
            final_metrics[strats][f'{metric} CI Upper'] = float(val.mean() + 1.96*se)

    # -------------------------------------------------------------------------
    # 2. SAVE RESULTS
    # -------------------------------------------------------------------------
    
    with open(OUTPUT_DIR / "real_analysis_results.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
        
    print("-" * 75)
    print(f"{'Strategy':<25} | {'Safety':<10} | {'Efficiency':<10} | {'Equity':<10} | {'Composite':<10}")
    print("-" * 75)
    for name, m in final_metrics.items():
        print(f"{name:<25} | {m['Safety Score']:.3f}      | {m['Efficiency Score']:.3f}      | {m['Equity Score']:.3f}    | {m['Composite Score']:.3f}")
    print("-" * 75)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()
