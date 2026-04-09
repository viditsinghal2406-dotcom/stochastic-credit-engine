"""
STPA - Diffusion Engine (Ornstein-Uhlenbeck Process)
======================================================
Models the continuous evolution of a borrower's financial health score
using a mean-reverting stochastic differential equation.

The OU Process:
    dX_t = θ(μ - X_t)dt + σ dW_t

Where:
    X_t  = financial health score at time t  (0–100 scale)
    θ    = mean-reversion speed  (how fast health reverts to long-run mean)
    μ    = long-run mean health score
    σ    = volatility (income instability, spending unpredictability)
    W_t  = Wiener process (standard Brownian motion)

Default is triggered when X_t falls below DEFAULT_THRESHOLD.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


DEFAULT_THRESHOLD = 20.0   # Score below which borrower is in default territory
DT = 1 / 12               # Monthly time step (1/12 of a year)


@dataclass
class OUParams:
    """Parameters for the Ornstein-Uhlenbeck process."""
    X0: float          # Initial health score (0–100)
    mu: float          # Long-run mean health score
    theta: float       # Mean-reversion speed (higher = faster reversion)
    sigma: float       # Volatility of health score
    horizon: int = 24  # Simulation horizon in months


class DiffusionEngine:
    """
    Ornstein-Uhlenbeck diffusion engine for financial health modeling.

    Why OU over GBM?
        - Credit scores mean-revert (good borrowers stay good, bad ones recover/worsen slowly)
        - GBM assumes unbounded random walk — unrealistic for bounded credit scores
        - OU captures the gravitational pull toward a borrower's "true" credit level

    Usage:
        engine = DiffusionEngine()
        path = engine.simulate(OUParams(X0=65, mu=60, theta=0.3, sigma=8))
        pd_prob = engine.default_probability(params, n_simulations=5000)
    """

    def simulate(self, params: OUParams, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate a single OU path for a borrower.

        Returns:
            Array of shape (n_steps,) — health score at each monthly step
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = params.horizon
        X = np.zeros(n_steps + 1)
        X[0] = params.X0

        for t in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(DT))
            drift = params.theta * (params.mu - X[t - 1]) * DT
            diffusion = params.sigma * dW
            X[t] = X[t - 1] + drift + diffusion
            X[t] = np.clip(X[t], 0, 100)  # Bounded score

        return X

    def simulate_batch(self, params: OUParams, n_simulations: int = 10000) -> np.ndarray:
        """
        Vectorized batch simulation — much faster than looping.

        Returns:
            Array of shape (n_simulations, horizon+1)
        """
        n = params.horizon
        X = np.zeros((n_simulations, n + 1))
        X[:, 0] = params.X0

        for t in range(1, n + 1):
            dW = np.random.normal(0, np.sqrt(DT), size=n_simulations)
            drift = params.theta * (params.mu - X[:, t - 1]) * DT
            diffusion = params.sigma * dW
            X[:, t] = np.clip(X[:, t - 1] + drift + diffusion, 0, 100)

        return X

    def default_probability(self, params: OUParams, n_simulations: int = 10000) -> float:
        """
        Estimate probability of ever crossing DEFAULT_THRESHOLD within horizon.

        Returns:
            Float in [0, 1]
        """
        paths = self.simulate_batch(params, n_simulations)
        ever_defaulted = (paths < DEFAULT_THRESHOLD).any(axis=1)
        return float(ever_defaulted.mean())

    def expected_time_to_default(self, params: OUParams, n_simulations: int = 10000) -> float:
        """
        Expected number of months before first crossing DEFAULT_THRESHOLD.
        Returns horizon+1 if default never occurs.

        Returns:
            Float (months)
        """
        paths = self.simulate_batch(params, n_simulations)
        times = []
        for path in paths:
            crossings = np.where(path < DEFAULT_THRESHOLD)[0]
            times.append(crossings[0] if len(crossings) > 0 else params.horizon + 1)
        return float(np.mean(times))

    def path_statistics(self, paths: np.ndarray) -> pd.DataFrame:
        """
        Summary statistics across simulated paths at each time step.

        Returns:
            DataFrame with columns: mean, median, p5, p25, p75, p95
        """
        return pd.DataFrame({
            "mean":   paths.mean(axis=0),
            "median": np.median(paths, axis=0),
            "p5":     np.percentile(paths, 5, axis=0),
            "p25":    np.percentile(paths, 25, axis=0),
            "p75":    np.percentile(paths, 75, axis=0),
            "p95":    np.percentile(paths, 95, axis=0),
        })

    def analytical_mean(self, params: OUParams, t: float) -> float:
        """Closed-form expected value of X(t) under OU process."""
        return params.mu + (params.X0 - params.mu) * np.exp(-params.theta * t)

    def analytical_variance(self, params: OUParams, t: float) -> float:
        """Closed-form variance of X(t) under OU process."""
        return (params.sigma ** 2 / (2 * params.theta)) * (1 - np.exp(-2 * params.theta * t))


# Allow Optional import
from typing import Optional
