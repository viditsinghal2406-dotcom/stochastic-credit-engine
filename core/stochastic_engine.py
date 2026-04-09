"""
Stochastic Engine
=================
High-level simulation orchestrator that wires together:
    DiffusionEngine  (OU process)
    JumpEngine       (Compound Poisson shocks)
    MarkovCreditEngine (discrete state transitions)

Exposes two interfaces:
    run_simulation(params, n_sim, horizon, seed) -> SimulationResult
    simulate_scenario(features, shock_type)      -> SimulationResult

No Monte Carlo bookkeeping lives here — that belongs in risk/risk_metrics.py.
This file only owns path generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from core.diffusion_engine import DiffusionEngine, OUParams, DEFAULT_THRESHOLD
from core.jump_engine import JumpEngine, ShockProfile, SHOCK_PROFILES
from core.markov_engine import MarkovCreditEngine
from core.parameter_mapper import StochasticParams, apply_macro_overlay

logger = logging.getLogger(__name__)

DT = 1 / 12   # monthly


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Raw simulation output: paths matrix + Markov state distribution.
    Downstream risk_metrics.py converts this into risk numbers.
    """
    paths: np.ndarray            # shape (n_sim, horizon+1)  — health scores
    markov_dist: list[dict]      # list[{state: prob}] per month — length horizon+1
    default_threshold: float     # boundary used (= DEFAULT_THRESHOLD = 20)
    n_simulations: int
    horizon_months: int
    params: StochasticParams


# ── Engine ─────────────────────────────────────────────────────────────────────

class StochasticEngine:
    """
    Hybrid stochastic simulation engine.

    dF(t) = θ(μ − F(t))dt + σ dW(t) + dJ(t)

    where dJ(t) is a compound Poisson jump.
    Markov chain runs in parallel to produce discrete state distribution.
    """

    def __init__(self) -> None:
        self._diffusion = DiffusionEngine()
        self._jump      = JumpEngine()

    # ── Primary interface ──────────────────────────────────────────────────────

    def run_simulation(
        self,
        params: StochasticParams,
        n_simulations: int = 5_000,
        horizon: int = 24,
        seed: int | None = 42,
    ) -> SimulationResult:
        """
        Simulate *n_simulations* paths of F(t) over *horizon* months.

        Parameters
        ----------
        params : StochasticParams
            Calibrated stochastic parameters from parameter_mapper.
        n_simulations : int
        horizon : int
            Number of monthly steps to simulate.
        seed : int | None
            For reproducibility.

        Returns
        -------
        SimulationResult
        """
        rng = np.random.default_rng(seed)

        ou_params = OUParams(
            X0=params.health_score,
            mu=params.mu,
            theta=params.theta,
            sigma=params.sigma,
            horizon=horizon,
        )
        profile = _build_shock_profile(params)

        # ── Vectorised OU paths ────────────────────────────────────────────────
        paths = self._diffusion.simulate_batch(ou_params, n_simulations)   # (n_sim, horizon+1)

        # ── Apply jumps vectorised ─────────────────────────────────────────────
        paths = self._jump.apply_shocks_batch(paths, profile)

        # ── Markov distribution (fresh instance per call — avoids shared-state mutation) ─
        markov_engine = MarkovCreditEngine()
        has_macro = (
            params.unemployment_delta != 0.0
            or params.interest_rate_delta != 0.0
            or params.gdp_growth < 0.0
        )
        if has_macro:
            markov_engine.apply_macro_stress(
                params.unemployment_delta,
                params.interest_rate_delta,
                params.gdp_growth,
            )
        markov_df   = markov_engine.predict(params.markov_initial_state, horizon)
        markov_dist = markov_df.to_dict(orient="records")

        return SimulationResult(
            paths=paths,
            markov_dist=markov_dist,
            default_threshold=DEFAULT_THRESHOLD,
            n_simulations=n_simulations,
            horizon_months=horizon,
            params=params,
        )

    # ── Scenario shocks ────────────────────────────────────────────────────────

    def simulate_scenario(
        self,
        features: dict,
        shock_type: str,
        n_simulations: int = 3_000,
        horizon: int = 24,
        seed: int | None = 99,
    ) -> SimulationResult:
        """
        Run a named macro/financial shock scenario on top of the base profile.

        Parameters
        ----------
        features : dict
            Feature dict from :func:`core.financial_analyzer.analyze`.
        shock_type : str
            One of ``"income_drop"``, ``"expense_spike"``, ``"recession"``,
            ``"job_loss"``, ``"medical_emergency"``, ``"windfall"``.
        n_simulations, horizon, seed : as above.

        Returns
        -------
        SimulationResult
        """
        from core.financial_state import build_financial_state
        from core.parameter_mapper import map_features_to_params

        state  = build_financial_state(features)
        params = map_features_to_params(state)
        params = _apply_scenario_shock(params, shock_type)

        return self.run_simulation(params, n_simulations, horizon, seed)


# ── Scenario definitions ───────────────────────────────────────────────────────

_SCENARIO_OVERRIDES: dict[str, dict] = {
    "income_drop": dict(
        sigma_mult=1.5, mu_delta=-12, lambda_neg_add=1.0, theta_mult=0.8
    ),
    "expense_spike": dict(
        sigma_mult=1.8, mu_delta=-8,  lambda_neg_add=0.8
    ),
    "recession": dict(
        unemployment_delta=0.06, interest_rate_delta=0.03, gdp_growth=-0.04
    ),
    "job_loss": dict(
        sigma_mult=2.0, mu_delta=-20, lambda_neg_add=1.5, theta_mult=0.6,
        immediate_shock=-20
    ),
    "medical_emergency": dict(
        sigma_mult=1.3, mu_delta=-10, lambda_neg_add=0.5,
        immediate_shock=-12
    ),
    "windfall": dict(
        sigma_mult=0.9, mu_delta=+10, lambda_neg_add=-0.3, lambda_pos_add=0.3
    ),
}


def _apply_scenario_shock(params: StochasticParams, shock_type: str) -> StochasticParams:
    """
    Return an adjusted StochasticParams for the given shock scenario.
    Macro-level scenarios delegate to apply_macro_overlay.
    """
    from dataclasses import replace

    overrides = _SCENARIO_OVERRIDES.get(shock_type)
    if overrides is None:
        logger.warning("Unknown shock_type '%s', returning base params.", shock_type)
        return params

    # Macro scenario (recession) → use existing overlay function
    if "unemployment_delta" in overrides:
        return apply_macro_overlay(params, **{k: overrides[k] for k in
                                              ["unemployment_delta", "interest_rate_delta", "gdp_growth"]})

    kw = dict(
        sigma=_clamp(params.sigma * overrides.get("sigma_mult", 1.0), 3.0, 28.0),
        mu=_clamp(params.mu + overrides.get("mu_delta", 0), 10.0, 90.0),
        lambda_neg=_clamp(
            params.lambda_neg + overrides.get("lambda_neg_add", 0.0), 0.5, 6.0
        ),
        lambda_pos=_clamp(
            params.lambda_pos + overrides.get("lambda_pos_add", 0.0), 0.1, 1.0
        ),
        theta=_clamp(params.theta * overrides.get("theta_mult", 1.0), 0.05, 0.65),
    )
    if "immediate_shock" in overrides:
        kw["health_score"] = _clamp(params.health_score + overrides["immediate_shock"], 0, 100)

    return replace(params, **kw)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_shock_profile(params: StochasticParams) -> ShockProfile:
    """Build a ShockProfile from StochasticParams, overriding preset if needed."""
    base = SHOCK_PROFILES.get(params.risk_tier, SHOCK_PROFILES["medium_risk"])
    from dataclasses import replace
    return replace(
        base,
        lambda_negative=params.lambda_neg,
        lambda_positive=params.lambda_pos,
        minor_prob=params.minor_prob,
        moderate_prob=params.moderate_prob,
        severe_prob=params.severe_prob,
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))
