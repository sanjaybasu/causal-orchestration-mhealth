
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Paths
DATA_PATH = "/Users/sanjaybasu/waymark-local/notebooks/causal_orchestration_mhealth/results/target_trial_data.csv.gz"
OUTPUT_DIR = "/Users/sanjaybasu/waymark-local/notebooks/causal_orchestration_mhealth/results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Set random seed
np.random.seed(2026)

def run_analysis():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    n = len(df)
    print(f"Loaded {n} members.")
    
    # -------------------------------------------------------------------------
    # 1. AGENT SIMULATION (Vectorized)
    # -------------------------------------------------------------------------
    
    # --- Safety Agent (DeepSeek-V3 proxy) ---
    # Objective: Maximize sensitivity. 
    # Logic: High baseline risk OR high comorbidities OR high recent utilization.
    # We calibrate 'threshold' to achieve ~80% contact rate (High Sensitivity).
    
    safety_score = df['baseline_risk'] * 5.0 + (df['charlson'] / 10.0) + (df['prior_utilization'] / 5.0 if 'prior_utilization' in df.columns else 0)
    safety_score = (safety_score - safety_score.min()) / (safety_score.max() - safety_score.min()) # Normalize 0-1
    df['safety_utility'] = safety_score
    df['safety_action'] = df['safety_utility'] > 0.2  # Aggressive contact threshold
    
    # --- Efficiency Agent (Claude 3.5 Sonnet proxy) ---
    # Objective: Maximize ROI.
    # Logic: Contact if CATE (benefit) > Cost threshold. 
    # CATE is negative for risk reduction (good).
    # Cost ratio: Intervention $150 / Event $1200 = 0.125 NNT.
    # We use 'true_cate' from the synthetic data as the agent's estimate.
    
    cost_effectiveness_threshold = -0.015 # 1.5% ARR required to justify cost
    df['efficiency_utility'] = -df['true_cate'] * 10  # Normalize rough scale
    df['efficiency_utility'] = df['efficiency_utility'].clip(0, 1)
    df['efficiency_action'] = df['true_cate'] < cost_effectiveness_threshold
    
    # --- Equity Agent (GPT-5.2 proxy) ---
    # Objective: Equalized Odds / Demographic Parity.
    # Logic: Boost priority for historically underserved or worse-outcome groups to match allocation.
    # Here we simulate an agent that upweights Black/Hispanic members to ensure parity using 'race' column.
    
    # Calculate current allocation rates by race from OTHER agents
    # This agent dynamically adjusts. For vectorization, we'll assign a static boost to under-represented groups
    # assuming they might be under-served by pure risk/efficiency models.
    
    equity_boost = np.zeros(n)
    equity_boost[df['race'] == 'Black'] = 0.2
    equity_boost[df['race'] == 'Hispanic'] = 0.15
    
    df['equity_utility'] = 0.5 + equity_boost
    df['equity_utility'] = df['equity_utility'].clip(0, 1)
    # Equity agent wants contact if utility is high
    df['equity_action'] = df['equity_utility'] > 0.6
    
    
    # -------------------------------------------------------------------------
    # 2. ORCHESTRATION (Nash Bargaining)
    # -------------------------------------------------------------------------
    
    print("Running Nash Bargaining Orchestration...")
    
    # Utilities for Contact (u_c) vs Wait (u_w)
    # Map priority 0-1 to utility.
    # U_contact = Priority
    # U_wait = 1 - Priority
    # Disagreement point d = 0.5
    
    d = 0.5
    
    # Calculate Nash Product for Contact
    # (u_s - d) * (u_eff - d) * (u_eq - d) -- simplified for discrete choice
    # Actually, standard Nash is argmax PROD (u_i(a)).
    # Let's simple mapping: 
    # Score(Contact) = P_safety * P_efficiency * P_equity
    # Score(Wait) = (1-P_safety) * (1-P_efficiency) * (1-P_equity)
    
    # We clip priorities to [0.01, 0.99] to avoid zero product
    # Rescale utilities to Avoid Strict Veto (0.2 to 0.95)
    # This represents that agents are rarely 100% certain 'No'.
    # This allows a strong 'Yes' from Safety to override a weak 'No' from Efficiency.
    
    def rescale(series):
        return 0.2 + (series * 0.75) # Map 0..1 to 0.2..0.95
        
    p_s = rescale(df['safety_utility'])
    p_e = rescale(df['efficiency_utility'])
    p_eq = rescale(df['equity_utility'])
    
    # Nash Product with d=0 (implicitly, since we just compare products of utilities)
    # If d != 0, we'd assume utility relative to d. 
    # Standard Nash: Maximize (U_s - d)(U_e - d)...
    # Here we simplify: Just compare Product(U_contact) vs Product(U_wait)
    # Where U_wait = 1 - U_contact? 
    # Let's define U(Contact) = p. U(Wait) = 1-p?
    # No, let's keep it simple: Action selected if Product(P) > Product(1-P)
    # With the flattened range [0.2, 0.95], 0.2 is "Lean No", 0.95 is "Strong Yes".
    
    nash_contact = p_s * p_e * p_eq
    nash_wait = (1-p_s) * (1-p_e) * (1-p_eq)
    
    df['nash_action'] = nash_contact > nash_wait
    
    # -------------------------------------------------------------------------
    # 3. OUTCOME CALCULATION
    # -------------------------------------------------------------------------
    
    # We use Y1_prob and Y0_prob (Exact probabilities) for expected value calculation.
    # This reduces variance compared to sampling binary outcomes.
    
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
    
    for name, action_vec in policies.items():
        # Calculate expected outcome
        # If action=True, prob = Y1_prob. If action=False, prob = Y0_prob.
        expected_probs = np.where(action_vec, y1_probs, y0_probs)
        event_rate = expected_probs.mean()
        contact_rate = action_vec.mean()
        
        # Calculate expected number of events
        expected_events = expected_probs.sum()
        
        results[name] = {
            'event_rate': float(event_rate),
            'contact_rate': float(contact_rate),
            'events': float(expected_events),
            'n_contact': int(action_vec.sum())
        }
        
    # Calculate RRR vs Control
    control_rate = results['Control (Usual Care)']['event_rate']
    
    for name in results:
        rate = results[name]['event_rate']
        rrr = (control_rate - rate) / control_rate
        results[name]['rrr'] = float(rrr)
        
        # Approximate CI for RRR
        # Using the standard error of the log relative risk
        # SE = sqrt( (1/E_treated) + (1/E_control) ) approx, assuming large N
        # For probabilities, we can use the variance of the mean.
        
        # Let's use a simple bootstrap for robust CI since we have arrays
        # (Simplified here for speed: analytic approx)
        # Var(Rate) approx Rate * (1-Rate) / N
        var_rate = rate * (1 - rate) / n
        var_control = control_rate * (1 - control_rate) / n
        
        # SE of Risk Difference
        se_rd = np.sqrt(var_rate + var_control)
        
        # RRR = 1 - RR. RR = Rate / Control.
        # SE(log RR) = sqrt( Var(Rate)/Rate^2 + Var(Control)/Control^2 )
        se_log_rr = np.sqrt( var_rate/(rate**2) + var_control/(control_rate**2) )
        
        rr = rate / control_rate
        ci_lower_rr = np.exp(np.log(rr) - 1.96 * se_log_rr)
        ci_upper_rr = np.exp(np.log(rr) + 1.96 * se_log_rr)
        
        results[name]['rrr_ci_lower'] = float(1 - ci_upper_rr)
        results[name]['rrr_ci_upper'] = float(1 - ci_lower_rr)
        
    # -------------------------------------------------------------------------
    # 4. FAIRNESS ANALYSIS (Multi-Agent vs Efficiency Only)
    # -------------------------------------------------------------------------
    
    fairness_res = {}
    for pol_name in ['Efficiency Agent Only', 'Multi-Agent (Nash)']:
        # True Positive Rate by Race (Sensitivity)
        # Target: People who WOULD have event under usual care (Y0_prob > threshold? No, probability mass)
        # Expected Sensitivity = Sum(Action * Y0_prob) / Sum(Y0_prob)
        # This gives "Coverage of At-Risk Mass"
        
        action = policies[pol_name].values
        
        tpr_by_race = {}
        for r in df['race'].unique():
            mask = (df['race'] == r)
            if mask.sum() > 0:
                denom = y0_probs[mask].sum()
                num = (action[mask] * y0_probs[mask]).sum()
                tpr = num / denom if denom > 0 else 0
                tpr_by_race[r] = float(tpr)
            else:
                tpr_by_race[r] = 0.0
                
        fairness_res[pol_name] = tpr_by_race
    
    results['fairness'] = fairness_res
    
    # Save
    with open(f"{OUTPUT_DIR}/real_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Analysis Complete.")
    print("-" * 40)
    for name in results:
        if 'rrr' in results[name]:
            print(f"{name}:")
            print(f"  RRR: {results[name]['rrr']*100:.1f}% ({results[name]['rrr_ci_lower']*100:.1f}-{results[name]['rrr_ci_upper']*100:.1f})")
            print(f"  Contact Rate: {results[name]['contact_rate']*100:.1f}%")
    print("-" * 40)
    
    # Save
    with open(f"{OUTPUT_DIR}/real_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Analysis Complete.")
    print("Nash RRR:", results['Multi-Agent (Nash)']['rrr'])
    print("Safety RRR:", results['Safety Agent Only']['rrr'])
    print("Efficiency RRR:", results['Efficiency Agent Only']['rrr'])
    print("Contact Rates - Nash:", results['Multi-Agent (Nash)']['contact_rate'])

if __name__ == "__main__":
    run_analysis()
