"""
STPA - Markov Credit Engine
============================
Models borrower credit state transitions using a
Non-Homogeneous Markov Chain with macro stress adjustment.

States:
    0 = EXCELLENT
    1 = GOOD
    2 = FAIR
    3 = STRESSED
    4 = DELINQUENT
    5 = DEFAULT   (absorbing)
    6 = RECOVERED
"""

import numpy as np
import pandas as pd
from typing import Optional


STATES = ["EXCELLENT", "GOOD", "FAIR", "STRESSED", "DELINQUENT", "DEFAULT", "RECOVERED"]
STATE_INDEX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
DEFAULT_STATE = STATE_INDEX["DEFAULT"]

BASE_TRANSITION_MATRIX = np.array([
    [0.80, 0.15, 0.03, 0.01, 0.00, 0.00, 0.01],  # EXCELLENT
    [0.10, 0.70, 0.13, 0.04, 0.02, 0.01, 0.00],  # GOOD
    [0.03, 0.12, 0.60, 0.15, 0.07, 0.03, 0.00],  # FAIR
    [0.01, 0.04, 0.15, 0.50, 0.20, 0.08, 0.02],  # STRESSED
    [0.00, 0.01, 0.05, 0.15, 0.40, 0.35, 0.04],  # DELINQUENT
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],  # DEFAULT (absorbing)
    [0.02, 0.10, 0.30, 0.35, 0.15, 0.05, 0.03],  # RECOVERED
], dtype=float)


class MarkovCreditEngine:
    """
    Non-Homogeneous Markov Chain for credit state modeling.

    Usage:
        engine = MarkovCreditEngine()
        state_dist = engine.predict(initial_state="GOOD", n_months=12)
        pd_score = engine.probability_of_default("STRESSED", 24)
    """

    def __init__(self, transition_matrix: Optional[np.ndarray] = None):
        self.P_base = transition_matrix if transition_matrix is not None else BASE_TRANSITION_MATRIX.copy()
        self.P = self.P_base.copy()

    def apply_macro_stress(
        self,
        unemployment_delta: float = 0.0,
        interest_rate_delta: float = 0.0,
        gdp_growth: float = 0.0
    ) -> "MarkovCreditEngine":
        """Shift transition matrix under macro-economic stress scenario."""
        stress = (
            1.0
            + 2.5 * max(unemployment_delta, 0)
            + 1.5 * max(interest_rate_delta, 0)
            - 1.0 * min(gdp_growth, 0)
        )
        P_stressed = self.P_base.copy()
        for i in range(N_STATES):
            if i == DEFAULT_STATE:
                continue
            P_stressed[i, DEFAULT_STATE] = min(P_stressed[i, DEFAULT_STATE] * stress, 0.95)
        row_sums = P_stressed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.P = P_stressed / row_sums
        return self

    def predict(self, initial_state: str, n_months: int = 24) -> pd.DataFrame:
        """Return state probability distribution for each month (0..n_months)."""
        v = np.zeros(N_STATES)
        v[STATE_INDEX[initial_state]] = 1.0
        history = [v.copy()]
        Pn = np.eye(N_STATES)
        for _ in range(n_months):
            Pn = Pn @ self.P
            history.append((v @ Pn).copy())
        return pd.DataFrame(history, columns=STATES)

    def probability_of_default(self, initial_state: str, n_months: int = 24) -> float:
        """Return probability of reaching DEFAULT state within n_months."""
        dist = self.predict(initial_state, n_months)
        return float(dist.iloc[-1]["DEFAULT"])

    @staticmethod
    def fit_from_data(df: pd.DataFrame, current_col: str = "state", next_col: str = "next_state") -> "MarkovCreditEngine":
        """Estimate transition matrix from observed state-to-state transitions."""
        counts = pd.crosstab(df[current_col], df[next_col])
        counts = counts.reindex(index=STATES, columns=STATES, fill_value=0)
        P = counts.div(counts.sum(axis=1).replace(0, 1), axis=0).values.astype(float)
        return MarkovCreditEngine(transition_matrix=P)

    def stationary_distribution(self) -> dict:
        """Long-run equilibrium state probabilities."""
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stat = np.real(eigenvectors[:, idx])
        stat = np.abs(stat) / np.abs(stat).sum()
        return dict(zip(STATES, stat))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.P, index=STATES, columns=STATES).round(4)
