"""
STPA - Survival Curve Calculator
==================================
Computes and formats survival curves from simulation output.

A survival curve shows P(no default by month t) — widely used
in actuarial science and credit risk management.
"""

import numpy as np
import pandas as pd
from core.diffusion_engine import DEFAULT_THRESHOLD


class SurvivalAnalyzer:
    """
    Survival analysis on simulated borrower paths.

    Usage:
        analyzer = SurvivalAnalyzer()
        curve = analyzer.compute(paths, horizon=24)
        half_life = analyzer.median_survival_time(curve)
    """

    def compute(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute survival curve from simulation paths.

        Args:
            paths: Array of shape (n_simulations, n_steps)

        Returns:
            Array of shape (n_steps,) — P(survived to month t)
        """
        n_sims, n_steps = paths.shape
        survival = np.zeros(n_steps)
        for t in range(n_steps):
            ever_defaulted = (paths[:, :t+1] < DEFAULT_THRESHOLD).any(axis=1)
            survival[t] = 1.0 - ever_defaulted.mean()
        return survival

    def to_dataframe(self, survival: np.ndarray) -> pd.DataFrame:
        """Convert survival array to a labeled DataFrame."""
        months = np.arange(len(survival))
        return pd.DataFrame({
            "month": months,
            "survival_probability": survival,
            "default_probability": 1 - survival,
            "survival_pct": (survival * 100).round(2),
        })

    def median_survival_time(self, survival: np.ndarray) -> float:
        """Month at which survival probability first drops below 50%."""
        below_half = np.where(survival < 0.5)[0]
        return float(below_half[0]) if len(below_half) > 0 else float(len(survival))

    def survival_at_month(self, survival: np.ndarray, month: int) -> float:
        """Return survival probability at a specific month."""
        if month < len(survival):
            return float(survival[month])
        return float(survival[-1])

    def compare_curves(
        self,
        curves: dict,
        horizon: int = 24
    ) -> pd.DataFrame:
        """
        Compare multiple survival curves (e.g. across stress scenarios).

        Args:
            curves: Dict of {label: survival_array}
            horizon: Number of months to include

        Returns:
            DataFrame with columns = labels, rows = months
        """
        data = {}
        for label, curve in curves.items():
            data[label] = curve[:horizon]
        df = pd.DataFrame(data)
        df.index.name = "month"
        return df
