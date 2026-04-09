"""
STPA - Parameter Calibrator
=============================
Estimates stochastic process parameters directly from historical
borrower data — so STPA is data-driven, not hardcoded.

Calibrates:
    - OU Process: θ (mean-reversion), μ (long-run mean), σ (volatility)
    - Markov Chain: transition matrix P from observed state sequences
    - Jump Process: λ (shock rate), severity distribution
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from scipy.optimize import minimize
from scipy.stats import norm

from core.markov_engine import MarkovCreditEngine, STATES
from core.jump_engine import ShockProfile


@dataclass
class CalibratedOUParams:
    theta: float   # Mean-reversion speed
    mu: float      # Long-run mean
    sigma: float   # Volatility
    r_squared: float  # Fit quality


@dataclass
class CalibratedJumpParams:
    lambda_rate: float     # Shocks per year
    shock_mean: float      # Average shock magnitude
    shock_std: float       # Shock magnitude std dev


class ParameterCalibrator:
    """
    Estimates STPA stochastic parameters from borrower time-series data.

    Usage:
        cal = ParameterCalibrator()

        # Calibrate OU from health score time series
        ou = cal.calibrate_ou(health_score_series)

        # Calibrate Markov from state sequences
        markov = cal.calibrate_markov(state_df)

        # Calibrate jumps from shock events
        jump = cal.calibrate_jumps(shock_series)
    """

    # ── OU Process Calibration ─────────────────────────────────────────────────

    def calibrate_ou(self, time_series: np.ndarray, dt: float = 1/12) -> CalibratedOUParams:
        """
        Calibrate OU process parameters via OLS regression on discrete observations.

        Uses the discrete approximation:
            X_{t+1} - X_t = θ(μ - X_t)Δt + σ√Δt ε_t

        Which is equivalent to:
            ΔX = a + b*X_t + noise
            where a = θμΔt, b = -θΔt
        """
        X = np.asarray(time_series, dtype=float)
        X_t = X[:-1]
        dX = np.diff(X)

        # OLS: dX = a + b*X_t
        A = np.column_stack([np.ones_like(X_t), X_t])
        coeffs, residuals, *_ = np.linalg.lstsq(A, dX, rcond=None)
        a, b = coeffs

        theta = -b / dt
        mu = a / (theta * dt) if abs(theta) > 1e-8 else X.mean()
        sigma_est = np.std(dX - (a + b * X_t)) / np.sqrt(dt)

        # R-squared
        y_pred = a + b * X_t
        ss_res = np.sum((dX - y_pred) ** 2)
        ss_tot = np.sum((dX - dX.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return CalibratedOUParams(
            theta=max(theta, 0.01),
            mu=float(np.clip(mu, 0, 100)),
            sigma=max(sigma_est, 0.1),
            r_squared=float(r2)
        )

    def calibrate_ou_batch(self, df: pd.DataFrame, score_col: str = "health_score", id_col: str = "borrower_id") -> pd.DataFrame:
        """Calibrate OU per borrower from panel data."""
        results = []
        for bid, group in df.groupby(id_col):
            series = group.sort_values("month")[score_col].values
            if len(series) < 5:
                continue
            params = self.calibrate_ou(series)
            results.append({
                "borrower_id": bid,
                "theta": params.theta,
                "mu": params.mu,
                "sigma": params.sigma,
                "r_squared": params.r_squared
            })
        return pd.DataFrame(results)

    # ── Markov Chain Calibration ──────────────────────────────────────────────

    def calibrate_markov(
        self,
        df: pd.DataFrame,
        current_col: str = "state",
        next_col: str = "next_state"
    ) -> MarkovCreditEngine:
        """Estimate transition matrix from observed state transitions."""
        return MarkovCreditEngine.fit_from_data(df, current_col, next_col)

    def infer_credit_states(self, df: pd.DataFrame, score_col: str = "health_score") -> pd.Series:
        """Map health scores to discrete Markov states."""
        bins = [0, 20, 35, 50, 65, 80, 100]
        labels = ["DEFAULT", "DELINQUENT", "STRESSED", "FAIR", "GOOD", "EXCELLENT"]
        return pd.cut(df[score_col], bins=bins, labels=labels, include_lowest=True)

    # ── Jump Process Calibration ──────────────────────────────────────────────

    def calibrate_jumps(
        self,
        change_series: np.ndarray,
        threshold_std_multiplier: float = 1.5,
        dt: float = 1/12
    ) -> CalibratedJumpParams:
        """
        Identify jump events in a time series and estimate Poisson parameters.

        Jumps are defined as changes exceeding threshold_std_multiplier * std.
        """
        changes = np.diff(np.asarray(change_series, dtype=float))
        std = changes.std()
        threshold = threshold_std_multiplier * std

        shock_mask = np.abs(changes) > threshold
        shock_magnitudes = changes[shock_mask]
        n_shocks = shock_mask.sum()
        T = len(changes) * dt

        lambda_rate = n_shocks / T if T > 0 else 0.0
        shock_mean = float(shock_magnitudes.mean()) if n_shocks > 0 else 0.0
        shock_std = float(shock_magnitudes.std()) if n_shocks > 1 else abs(shock_mean) * 0.3

        return CalibratedJumpParams(
            lambda_rate=lambda_rate,
            shock_mean=shock_mean,
            shock_std=shock_std
        )

    # ── Segment-based calibration ─────────────────────────────────────────────

    def calibrate_segment_profiles(self, df: pd.DataFrame) -> dict:
        """
        Calibrate different parameter sets for borrower risk segments.
        Returns a dict of {segment_label: {ou_params, jump_params}}
        """
        segments = {}
        if "health_score" not in df.columns:
            return segments

        quantiles = df["health_score"].quantile([0.25, 0.50, 0.75])
        df = df.copy()
        df["segment"] = pd.cut(
            df["health_score"],
            bins=[0, quantiles[0.25], quantiles[0.50], quantiles[0.75], 100],
            labels=["subprime", "high_risk", "medium_risk", "low_risk"],
            include_lowest=True
        )

        for seg, group in df.groupby("segment"):
            scores = group["health_score"].values
            ou = self.calibrate_ou(scores)
            jump = self.calibrate_jumps(scores)
            segments[str(seg)] = {
                "ou": ou,
                "jump": jump,
                "n_borrowers": len(group)
            }
        return segments
