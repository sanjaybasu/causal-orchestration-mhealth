
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.base import BaseEstimator

class XLearner(BaseEstimator):
    """
    X-Learner implementation for Heterogeneous Treatment Effect estimation.
    Follows methodology described in Kunzel et al. (2019).
    
    Stages:
    1. Estimate outcome surfaces mu_1(x) and mu_0(x) for treated and control.
    2. Impute counterfactuals to get pseudo-treatment effects D_1 and D_0.
    3. Estimate variation in pseudo-effects tau_1(x) and tau_0(x).
    4. Combine estimates via propensity score weighting: tau(x) = g(x)tau_0(x) + (1-g(x))tau_1(x).
    """
    
    def __init__(self, outcome_learner=None, effect_learner=None, propensity_learner=None):
        self.outcome_learner = outcome_learner if outcome_learner else XGBRegressor(n_estimators=100, max_depth=5)
        self.effect_learner = effect_learner if effect_learner else XGBRegressor(n_estimators=100, max_depth=5)
        self.propensity_learner = propensity_learner if propensity_learner else XGBClassifier(n_estimators=100, max_depth=4)
        
        self.mu0 = None
        self.mu1 = None
        self.tau0 = None
        self.tau1 = None
        self.g = None
        
    def fit(self, X, y, w):
        """
        Fit the X-Learner.
        
        Args:
            X: Covariate matrix (DataFrame or array)
            y: Outcome vector
            w: Treatment indicator vector (0/1)
        """
        # Data splitting
        X0 = X[w == 0]
        y0 = y[w == 0]
        X1 = X[w == 1]
        y1 = y[w == 1]
        
        # Stage 1: Outcome Estimators
        self.mu0 = self.outcome_learner.__class__(**self.outcome_learner.get_params())
        self.mu1 = self.outcome_learner.__class__(**self.outcome_learner.get_params())
        
        self.mu0.fit(X0, y0)
        self.mu1.fit(X1, y1)
        
        # Stage 2: Pseudo-Effects
        # Impute counterfactuals
        d1 = y1 - self.mu0.predict(X1)
        d0 = self.mu1.predict(X0) - y0
        
        # Stage 3: Effect Estimators
        self.tau0 = self.effect_learner.__class__(**self.effect_learner.get_params())
        self.tau1 = self.effect_learner.__class__(**self.effect_learner.get_params())
        
        self.tau0.fit(X0, d0)
        self.tau1.fit(X1, d1)
        
        # Stage 4: Propensity Score
        self.g = self.propensity_learner.__class__(**self.propensity_learner.get_params())
        self.g.fit(X, w)
        
        return self
        
    def predict(self, X):
        """
        Predict CATE for new data.
        """
        tau0_pred = self.tau0.predict(X)
        tau1_pred = self.tau1.predict(X)
        g_pred = self.g.predict_proba(X)[:, 1]
        
        # Weighted combination
        tau_hat = g_pred * tau0_pred + (1 - g_pred) * tau1_pred
        
        return tau_hat
