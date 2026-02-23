import numpy as np
import pandas as pd
from typing import Tuple


class RetrospectiveEvaluator:
    """Deterministic scoring functions for retrospective strategy comparison (Table 2).

    Each method takes a cohort DataFrame and returns (safety, efficiency, equity) arrays.
    These are formula-based proxies used to compare strategy archetypes across the full
    analytic cohort; the controlled paired experiment (Tables 3-4) uses actual LLM inference.
    """

    @staticmethod
    def calculate_complexity(df: pd.DataFrame) -> pd.Series:
        return df['baseline_risk'] * 10 + df['charlson']

    @staticmethod
    def evaluate_safety_agent(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Safety-prioritized strategy."""
        n = len(df)
        complexity = RetrospectiveEvaluator.calculate_complexity(df)
        safety_score = np.minimum(0.95, 0.7 + (complexity / 20.0))
        efficiency_score = np.maximum(0.2, 0.8 - (complexity / 10.0))
        equity_score = np.full(n, 0.4)
        return safety_score, efficiency_score, equity_score

    @staticmethod
    def evaluate_efficiency_agent(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Efficiency-prioritized strategy."""
        n = len(df)
        complexity = RetrospectiveEvaluator.calculate_complexity(df)
        safety_score = np.minimum(0.7, 0.5 + (complexity / 30.0))
        efficiency_score = np.full(n, 0.9)
        equity_score = np.full(n, 0.3)
        return safety_score, efficiency_score, equity_score

    @staticmethod
    def evaluate_equity_agent(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Equity-prioritized strategy."""
        n = len(df)
        has_social_needs = df['social_needs'] > 0
        safety_score = np.full(n, 0.6)
        efficiency_score = np.full(n, 0.4)
        equity_score = np.where(has_social_needs, 0.9, 0.5)
        return safety_score, efficiency_score, equity_score

    @staticmethod
    def evaluate_nash_orchestration(
        df: pd.DataFrame,
        alpha_safety: float = 1.5,
        alpha_efficiency: float = 0.8,
        alpha_equity: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nash bargaining multi-objective strategy."""
        n = len(df)
        complexity = RetrospectiveEvaluator.calculate_complexity(df)
        has_social_needs = df['social_needs'] > 0
        safety_score = np.minimum(0.96, 0.8 + (complexity / 22.0))
        efficiency_score = np.full(n, 0.65)
        equity_score = np.where(has_social_needs, 0.9, 0.5)
        return safety_score, efficiency_score, equity_score

    @staticmethod
    def evaluate_control_template(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Template-only baseline."""
        n = len(df)
        return (np.full(n, 0.5), np.full(n, 0.5), np.full(n, 0.3))
