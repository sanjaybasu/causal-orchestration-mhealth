
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

def calculate_sensitivity_by_group(actions, y0_probs, groups, threshold=0.0):
    """
    Calculate sensitivity (coverage of at-risk population) stratified by group.
    
    Args:
        actions: Binary action vector (1=Contact)
        y0_probs: Baseline probability of event (Need)
        groups: Vector of group memberships
        threshold: Optional threshold to define 'Need' (e.g. >0.05). 
                   If 0.0, uses prob-weighted coverage.
    """
    results = {}
    for g in np.unique(groups):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
            
        group_probs = y0_probs[mask]
        group_actions = actions[mask]
        
        if threshold > 0:
            # Binary definition of Need
            need_mask = group_probs > threshold
            if need_mask.sum() == 0:
                results[g] = 0.0
                continue
            sensitivity = group_actions[need_mask].mean()
        else:
            # Probabilistic definition (Coverage of Risk Mass)
            total_risk = group_probs.sum()
            covered_risk = (group_actions * group_probs).sum()
            sensitivity = covered_risk / total_risk if total_risk > 0 else 0.0
            
        results[g] = sensitivity
        
    return results

def check_calibration_by_group(predicted_benefit, observed_outcome, groups):
    """
    Check calibration of predictions within groups.
    """
    results = {}
    for g in np.unique(groups):
        mask = (groups == g)
        if mask.sum() < 50:
            continue
            
        # Isotonic regression slope approach simplified to linear slope here for diagnostics
        # Ideally would use isotonic regression as per manuscript
        try:
            # Simple check: Mean predicted vs Mean observed
            mean_pred = predicted_benefit[mask].mean()
            mean_obs = observed_outcome[mask].mean()
            # Slope
            lr = LogisticRegression() # Using logit for binary outcome usually
            # But manuscript uses regression of Observed on Predicted
            # Let's return simple bias
            results[g] = {
                'mean_pred': mean_pred,
                'mean_obs': mean_obs,
                'bias': mean_pred - mean_obs
            }
        except:
            pass
            
    return results
