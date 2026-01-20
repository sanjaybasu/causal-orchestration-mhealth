
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os

# Import modular components
try:
    from xlearner import XLearner
    from fairness_metrics import calculate_sensitivity_by_group
    # from propensity_score_analysis import calculate_ipw_weights # Not used in simplified run
except ImportError:
    # Handle running from root vs src
    import sys
    sys.path.append(str(Path(__file__).parent))
    from xlearner import XLearner
    from fairness_metrics import calculate_sensitivity_by_group

# Paths - Relative to script location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_cohort.csv.gz"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed
np.random.seed(2026)

def run_analysis():
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print("Error: Data file not found. Please ensure synthetic_cohort.csv.gz is in data/ directory.")
        return

    df = pd.read_csv(DATA_PATH)
    n = len(df)
    print(f"Loaded {n} members.")
    
    # -------------------------------------------------------------------------
    # 1. AGENT SIMULATION (Vectorized)
    # -------------------------------------------------------------------------
    
    print("Simulating Agents...")
    
    # --- Safety Agent (DeepSeek-V3 proxy) ---
    # Objective: Maximize sensitivity. 
    # Logic: High baseline risk OR high comorbidities OR high recent utilization.
    safety_score = df['baseline_risk'] * 5.0 + (df['charlson'] / 10.0) + (df['prior_utilization'] / 5.0 if 'prior_utilization' in df.columns else 0)
    safety_score = (safety_score - safety_score.min()) / (safety_score.max() - safety_score.min()) # Normalize 0-1
    df['safety_utility'] = safety_score
    df['safety_action'] = df['safety_utility'] > 0.2  # Aggressive contact threshold
    
    # --- Efficiency Agent (Claude 3.5 Sonnet proxy) ---
    # Objective: Maximize ROI.
    cost_effectiveness_threshold = -0.015 # 1.5% ARR required to justify cost
    df['efficiency_utility'] = -df['true_cate'] * 10  # Normalize rough scale
    df['efficiency_utility'] = df['efficiency_utility'].clip(0, 1)
    df['efficiency_action'] = df['true_cate'] < cost_effectiveness_threshold
    
    # --- Equity Agent (GPT-5.2 proxy) ---
    # Objective: Equalized Odds / Demographic Parity.
    equity_boost = np.zeros(n)
    equity_boost[df['race'] == 'Black'] = 0.2
    equity_boost[df['race'] == 'Hispanic'] = 0.15
    
    df['equity_utility'] = 0.5 + equity_boost
    df['equity_utility'] = df['equity_utility'].clip(0, 1)
    df['equity_action'] = df['equity_utility'] > 0.6
    
    
    # -------------------------------------------------------------------------
    # 2. ORCHESTRATION (Nash Bargaining)
    # -------------------------------------------------------------------------
    
    print("Running Nash Bargaining Orchestration...")
    
    def rescale(series):
        return 0.2 + (series * 0.75) # Map 0..1 to 0.2..0.95
        
    p_s = rescale(df['safety_utility'])
    p_e = rescale(df['efficiency_utility'])
    p_eq = rescale(df['equity_utility'])
    
    nash_contact = p_s * p_e * p_eq
    nash_wait = (1-p_s) * (1-p_e) * (1-p_eq)
    
    df['nash_action'] = nash_contact > nash_wait
    
    # -------------------------------------------------------------------------
    # 3. OUTCOME CALCULATION
    # -------------------------------------------------------------------------
    
    # Policies to evaluate
    policies = {
        'Control (Usual Care)': np.zeros(n, dtype=bool),
        'Safety Agent Only': df['safety_action'],
        'Efficiency Agent Only': df['efficiency_action'],
        'Equity Agent Only': df['equity_action'],
        'Multi-Agent (Nash)': df['nash_action']
    }
    
    results = {}
    
    # Pre-fetch arrays for speed
    y1_probs = df['Y1_prob'].values
    y0_probs = df['Y0_prob'].values
    race_vec = df['race'].values
    
    for name, action_vec in policies.items():
        # Calculate expected outcome
        expected_probs = np.where(action_vec, y1_probs, y0_probs)
        event_rate = expected_probs.mean()
        contact_rate = action_vec.mean()
        expected_events = expected_probs.sum()
        
        results[name] = {
            'event_rate': float(event_rate),
            'contact_rate': float(contact_rate),
            'events': float(expected_events),
            'n_contact': int(action_vec.sum())
        }
        
    # Calculate RRR vs Control and CIs
    control_rate = results['Control (Usual Care)']['event_rate']
    
    for name in results:
        rate = results[name]['event_rate']
        rrr = (control_rate - rate) / control_rate
        results[name]['rrr'] = float(rrr)
        
        # Analytic approx for CI
        var_rate = rate * (1 - rate) / n
        var_control = control_rate * (1 - control_rate) / n
        se_log_rr = np.sqrt( var_rate/(rate**2) + var_control/(control_rate**2) )
        
        rr = rate / control_rate
        ci_lower_rr = np.exp(np.log(rr) - 1.96 * se_log_rr)
        ci_upper_rr = np.exp(np.log(rr) + 1.96 * se_log_rr)
        
        results[name]['rrr_ci_lower'] = float(1 - ci_upper_rr)
        results[name]['rrr_ci_upper'] = float(1 - ci_lower_rr)
        
    # -------------------------------------------------------------------------
    # 4. FAIRNESS ANALYSIS 
    # -------------------------------------------------------------------------
    
    print("Calculating Fairness Metrics...")
    
    fairness_res = {}
    for pol_name in ['Efficiency Agent Only', 'Multi-Agent (Nash)']:
        action = policies[pol_name].values
        # Use modular function
        tpr_by_race = calculate_sensitivity_by_group(action, y0_probs, race_vec, threshold=0.0)
        # Convert to float for JSON safety
        fairness_res[pol_name] = {k: float(v) for k, v in tpr_by_race.items()}
    
    results['fairness'] = fairness_res
    
    # Save
    with open(OUTPUT_DIR / "real_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("-" * 40)
    for name in results:
        if 'rrr' in results[name]:
            print(f"{name}:")
            print(f"  RRR: {results[name]['rrr']*100:.1f}% ({results[name]['rrr_ci_lower']*100:.1f}-{results[name]['rrr_ci_upper']*100:.1f})")
            print(f"  Contact Rate: {results[name]['contact_rate']*100:.1f}%")
    print("-" * 40)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()
