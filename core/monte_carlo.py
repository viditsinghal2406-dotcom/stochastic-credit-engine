"""
STPA - Monte Carlo Fusion Layer
================================
Combines the three stochastic engines into a unified simulation framework:
    - Engine 1: Markov Credit FSM  (state transitions)
    - Engine 2: OU Diffusion       (continuous health score drift)
    - Engine 3: Jump Process       (sudden financial shocks)

Runs 10,000 borrower path simulations and aggregates results
into a comprehensive risk profile.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

from core.markov_engine import MarkovCreditEngine, STATES, STATE_INDEX, DEFAULT_THRESHOLD as MARKOV_THRESHOLD
from core.diffusion_engine import DiffusionEngine, OUParams, DEFAULT_THRESHOLD
from core.jump_engine import JumpEngine, ShockProfile, SHOCK_PROFILES


@dataclass
class BorrowerProfile:
    """Complete borrower input profile for STPA simulation."""

    # Identity
    borrower_id: str = "B001"

    # Financial health score (0–100)
    health_score: float = 60.0        # Current score
    long_run_mean: float = 55.0       # Estimated long-run average
    reversion_speed: float = 0.25     # θ — how quickly health reverts
    volatility: float = 8.0           # σ — monthly score volatility

    # Credit state
    initial_state: str = "FAIR"       # Starting Markov state

    # Shock profile key (see SHOCK_PROFILES)
    risk_tier: str = "medium_risk"

    # Macro scenario overrides
    unemployment_delta: float = 0.0
    interest_rate_delta: float = 0.0
    gdp_growth: float = 0.0

    # Simulation config
    horizon_months: int = 24
    n_simulations: int = 10_000


@dataclass
class STPAResult:
    """Full simulation output from Monte Carlo fusion."""

    borrower_id: str

    # Core risk metrics
    pd_score: float              # 0–100 probability of default score
    pd_probability: float        # Raw probability (0–1)
    expected_time_to_default: float  # Months
    risk_tier: str               # LOW / MEDIUM / HIGH / CRITICAL

    # Path statistics
    path_stats: pd.DataFrame     # mean, p5, p25, p75, p95 at each month
    all_paths: np.ndarray        # Full simulation matrix (n_sim x horizon)

    # Markov state distribution
    markov_dist: pd.DataFrame    # State probabilities at each month

    # Survival curve
    survival_curve: np.ndarray   # P(no default) at each month

    # Stress test comparison
    base_pd: float
    stressed_pd: float

    # Metadata
    n_simulations: int
    horizon_months: int
    params: BorrowerProfile


class MonteCarloEngine:
    """
    STPA Monte Carlo Fusion Engine.

    Orchestrates all three stochastic engines to produce a
    comprehensive credit risk analysis for a single borrower.

    Usage:
        engine = MonteCarloEngine()
        result = engine.run(BorrowerProfile(...))
        print(f"PD Score: {result.pd_score:.1f}/100")
    """

    def __init__(self):
        self.diffusion = DiffusionEngine()
        self.jump = JumpEngine()
        self.markov = MarkovCreditEngine()

    def _score_to_risk_tier(self, pd_score: float) -> str:
        if pd_score < 15:   return "LOW"
        elif pd_score < 35: return "MEDIUM"
        elif pd_score < 60: return "HIGH"
        else:               return "CRITICAL"

    def _compute_survival_curve(self, paths: np.ndarray) -> np.ndarray:
        """P(no default by month t) for each t in horizon."""
        n_sims, n_steps = paths.shape
        survived = np.ones(n_steps)
        for t in range(n_steps):
            ever_defaulted = (paths[:, :t+1] < DEFAULT_THRESHOLD).any(axis=1)
            survived[t] = 1.0 - ever_defaulted.mean()
        return survived

    def _run_base_simulation(self, profile: BorrowerProfile) -> np.ndarray:
        """Run base OU + Jump simulation (no macro stress)."""
        ou_params = OUParams(
            X0=profile.health_score,
            mu=profile.long_run_mean,
            theta=profile.reversion_speed,
            sigma=profile.volatility,
            horizon=profile.horizon_months
        )
        base_paths = self.diffusion.simulate_batch(ou_params, profile.n_simulations)
        shock_profile = SHOCK_PROFILES.get(profile.risk_tier, SHOCK_PROFILES["medium_risk"])
        shocked_paths = self.jump.apply_shocks_batch(base_paths, shock_profile)
        return shocked_paths

    def run(self, profile: BorrowerProfile) -> STPAResult:
        """
        Execute full STPA Monte Carlo simulation for a borrower.

        Steps:
            1. Simulate 10k OU diffusion paths
            2. Apply Compound Poisson shocks to each path
            3. Compute PD score from paths
            4. Run Markov chain state distribution
            5. Compute survival curve
            6. Run stressed scenario for comparison
        """

        # ── Step 1+2: Base simulation ─────────────────────────────────────────
        paths = self._run_base_simulation(profile)
        ever_defaulted = (paths < DEFAULT_THRESHOLD).any(axis=1)
        pd_probability = float(ever_defaulted.mean())
        pd_score = pd_probability * 100

        # ── Step 3: Expected time to default ─────────────────────────────────
        times = []
        for path in paths:
            crossings = np.where(path < DEFAULT_THRESHOLD)[0]
            times.append(int(crossings[0]) if len(crossings) > 0 else profile.horizon_months + 1)
        expected_ttd = float(np.mean(times))

        # ── Step 4: Path statistics ───────────────────────────────────────────
        path_stats = self.diffusion.path_statistics(paths)

        # ── Step 5: Markov state distribution ────────────────────────────────
        self.markov.apply_macro_stress(
            unemployment_delta=profile.unemployment_delta,
            interest_rate_delta=profile.interest_rate_delta,
            gdp_growth=profile.gdp_growth
        )
        markov_dist = self.markov.predict(profile.initial_state, profile.horizon_months)

        # ── Step 6: Survival curve ────────────────────────────────────────────
        survival_curve = self._compute_survival_curve(paths)

        # ── Step 7: Stressed PD (recession scenario) ─────────────────────────
        stressed_profile = BorrowerProfile(
            **{**profile.__dict__,
               "unemployment_delta": max(profile.unemployment_delta, 0.05),
               "interest_rate_delta": max(profile.interest_rate_delta, 0.02),
               "gdp_growth": min(profile.gdp_growth, -0.02),
               "n_simulations": 2000}  # Fewer sims for speed
        )
        stressed_paths = self._run_base_simulation(stressed_profile)
        stressed_pd = float((stressed_paths < DEFAULT_THRESHOLD).any(axis=1).mean()) * 100

        return STPAResult(
            borrower_id=profile.borrower_id,
            pd_score=round(pd_score, 2),
            pd_probability=round(pd_probability, 4),
            expected_time_to_default=round(expected_ttd, 1),
            risk_tier=self._score_to_risk_tier(pd_score),
            path_stats=path_stats,
            all_paths=paths,
            markov_dist=markov_dist,
            survival_curve=survival_curve,
            base_pd=round(pd_score, 2),
            stressed_pd=round(stressed_pd, 2),
            n_simulations=profile.n_simulations,
            horizon_months=profile.horizon_months,
            params=profile
        )

    def run_portfolio(self, profiles: list[BorrowerProfile]) -> pd.DataFrame:
        """Run simulation for multiple borrowers and return summary DataFrame."""
        results = []
        for p in profiles:
            r = self.run(p)
            results.append({
                "borrower_id": r.borrower_id,
                "pd_score": r.pd_score,
                "pd_probability": r.pd_probability,
                "risk_tier": r.risk_tier,
                "expected_ttd_months": r.expected_time_to_default,
                "stressed_pd": r.stressed_pd,
            })
        return pd.DataFrame(results)
