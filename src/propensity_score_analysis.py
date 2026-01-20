
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def estimate_propensity_scores(df, covariates, treatment_col='treatment'):
    """
    Estimate propensity scores using logistic regression.
    """
    model = LogisticRegression(max_iter=1000, penalty='l2')
    X = df[covariates]
    y = df[treatment_col]
    
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    
    return ps, model

def calculate_ipw_weights(df, ps_col='propensity_score', treatment_col='treatment', stabilize=True):
    """
    Calculate Inverse Probability Weights.
    """
    ps = df[ps_col]
    t = df[treatment_col]
    
    weights = np.where(t == 1, 1/ps, 1/(1-ps))
    
    if stabilize:
        p_t = t.mean()
        stabilizer = np.where(t == 1, p_t, 1-p_t)
        weights = weights * stabilizer
        
    return weights

def trim_weights(df, ps_col, lower=0.01, upper=0.99):
    """
    Symmetric trimming of propensity scores.
    """
    mask = (df[ps_col] >= lower) & (df[ps_col] <= upper)
    return df[mask].copy()
