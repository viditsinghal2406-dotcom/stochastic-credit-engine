"""
Parameter Mapper
================
Maps a FinancialState to calibrated stochastic model parameters:

    theta  — mean-reversion speed (OU)
    mu     — long-run mean health score (OU)
    sigma  — diffusion volatility (OU)
    lambda_neg / lambda_pos — Poisson shock arrival rates (Jump)
    markov_initial_state    — starting state for Markov chain
    risk_tier               — shock profile key

Mapping rules are derived from financial intuition:
    - Stable salaried income     → high theta (fast reversion)
    - High savings rate          → high mu    (good long-run state)
    - High expense variability   → high sigma (noisy path)
    - Low balance buffer         → high lambda_neg (frequent shocks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.financial_state import FinancialState

logger = logging.getLogger(__name__)

# ── Baseline parameter bounds ──────────────────────────────────────────────────
_THETA_MIN, _THETA_MAX = 0.08, 0.60     # reversion speed
_MU_MIN,    _MU_MAX    = 15.0, 90.0     # long-run mean
_SIGMA_MIN, _SIGMA_MAX = 3.0,  22.0     # volatility
_LAMBDA_NEG_MIN        = 0.6
_LAMBDA_NEG_MAX        = 5.0
_LAMBDA_POS_MIN        = 0.1
_LAMBDA_POS_MAX        = 0.8


@dataclass
class StochasticParams:
    """
    Calibrated stochastic model parameters derived from FinancialState.
    All fields are ready to pass directly into the simulation engines.
    """
    # OU params
    health_score:    float   # F(0) — initial score
    mu:              float   # long-run mean
    theta:           float   # mean-reversion speed
    sigma:           float   # diffusion volatility

    # Jump process
    lambda_neg:      float   # negative shock arrival rate (shocks/year)
    lambda_pos:      float   # positive shock arrival rate
    severe_prob:     float   # probability a negative shock is severe
    moderate_prob:   float   # probability a negative shock is moderate
    minor_prob:      float   # probability a negative shock is minor

    # Classification
    markov_initial_state: str  # EXCELLENT / GOOD / FAIR / STRESSED / DELINQUENT
    risk_tier:            str  # low_risk / medium_risk / high_risk / subprime

    # Macro overlays
    unemployment_delta:   float = 0.0
    interest_rate_delta:  float = 0.0
    gdp_growth:           float = 0.02

    # Provenance
    source_composite: float = 50.0   # F(0) for audit


def map_features_to_params(state: FinancialState) -> StochasticParams:
    """
    Derive stochastic model parameters from a :class:`FinancialState`.

    Parameters
    ----------
    state : FinancialState
        Computed from bank summary + ITR features.

    Returns
    -------
    StochasticParams
    """
    F0 = state.composite_score

    # ── theta: mean-reversion speed ───────────────────────────────────────────
    # Stable income (low CV) + salary detected → revert quickly to mean
    base_theta = 0.20
    theta = base_theta + (1.0 - state.income_cv) * 0.20
    if state.salary_detected:
        theta += 0.08
    if state.balance_ratio >= 2.0:
        theta += 0.05
    theta = _clamp(theta, _THETA_MIN, _THETA_MAX)

    # ── mu: long-run mean ─────────────────────────────────────────────────────
    # Derived from savings & balance quality; penalised by risk flags
    flag_penalty = state.n_risk_flags * 4.0
    mu = (
        0.40 * state.savings_component +
        0.35 * state.balance_component +
        0.25 * state.income_component  -
        flag_penalty
    )
    mu = _clamp(mu, _MU_MIN, _MU_MAX)

    # ── sigma: diffusion volatility ───────────────────────────────────────────
    # High income/expense variability → noisy paths
    base_sigma = 6.0
    sigma = base_sigma + state.income_cv * 8.0 + state.expense_cv * 5.0
    if state.min_balance < 0:
        sigma += 3.0
    sigma = _clamp(sigma, _SIGMA_MIN, _SIGMA_MAX)

    # ── lambda_neg: negative shock rate ──────────────────────────────────────
    # Low balance buffer + many flags → frequent shocks
    base_lambda_neg = 1.0 + state.n_risk_flags * 0.35
    if state.balance_ratio < 0.5:
        base_lambda_neg += 0.8
    elif state.balance_ratio < 1.0:
        base_lambda_neg += 0.3
    lambda_neg = _clamp(base_lambda_neg, _LAMBDA_NEG_MIN, _LAMBDA_NEG_MAX)

    # ── lambda_pos: positive shock rate ──────────────────────────────────────
    lambda_pos = _clamp(0.5 - state.expense_cv * 0.3, _LAMBDA_POS_MIN, _LAMBDA_POS_MAX)
    if state.salary_detected and state.savings_component > 60:
        lambda_pos += 0.15

    # ── Shock severity distribution ───────────────────────────────────────────
    # Worse financial state → higher probability of severe shocks
    severity_factor = _clamp((100.0 - F0) / 100.0)   # 0 (excellent) → 1 (terrible)
    severe_prob   = _clamp(0.05 + severity_factor * 0.20, 0.05, 0.30)
    moderate_prob = _clamp(0.25 + severity_factor * 0.15, 0.20, 0.45)
    minor_prob    = _clamp(1.0 - severe_prob - moderate_prob, 0.30, 0.70)

    # ── Markov initial state ──────────────────────────────────────────────────
    markov_initial = _score_to_markov_state(F0)

    # ── Risk tier ─────────────────────────────────────────────────────────────
    risk_tier = _score_to_risk_tier(F0, state.n_risk_flags, state.income_cv)

    params = StochasticParams(
        health_score=round(F0, 2),
        mu=round(mu, 2),
        theta=round(theta, 4),
        sigma=round(sigma, 2),
        lambda_neg=round(lambda_neg, 3),
        lambda_pos=round(lambda_pos, 3),
        severe_prob=round(severe_prob, 3),
        moderate_prob=round(moderate_prob, 3),
        minor_prob=round(minor_prob, 3),
        markov_initial_state=markov_initial,
        risk_tier=risk_tier,
        source_composite=round(F0, 2),
    )
    logger.debug("StochasticParams: %s", params)
    return params


def apply_macro_overlay(params: StochasticParams,
                        unemployment_delta: float = 0.0,
                        interest_rate_delta: float = 0.0,
                        gdp_growth: float = 0.02) -> StochasticParams:
    """
    Return a new :class:`StochasticParams` with macro stress adjustments.

    Macro stress formula (from Markov engine):
        stress_factor = 1 + 2.5·Δu + 1.5·Δr − gdp

    This amplifies sigma and lambda_neg, and depresses mu.
    """
    from dataclasses import replace
    stress = 1.0 + 2.5 * max(unemployment_delta, 0) \
                 + 1.5 * max(interest_rate_delta, 0) \
                 - min(gdp_growth, 0)

    new_sigma     = _clamp(params.sigma * stress, _SIGMA_MIN, _SIGMA_MAX + 5)
    new_lambda    = _clamp(params.lambda_neg * stress, _LAMBDA_NEG_MIN, _LAMBDA_NEG_MAX + 1)
    new_mu        = _clamp(params.mu / stress, _MU_MIN, _MU_MAX)
    new_theta     = _clamp(params.theta / (stress ** 0.3), _THETA_MIN, _THETA_MAX)

    return replace(params,
                   sigma=round(new_sigma, 2),
                   lambda_neg=round(new_lambda, 3),
                   mu=round(new_mu, 2),
                   theta=round(new_theta, 4),
                   unemployment_delta=unemployment_delta,
                   interest_rate_delta=interest_rate_delta,
                   gdp_growth=gdp_growth)


# ── Private helpers ────────────────────────────────────────────────────────────

def _score_to_markov_state(score: float) -> str:
    if score >= 80:  return "EXCELLENT"
    if score >= 65:  return "GOOD"
    if score >= 48:  return "FAIR"
    if score >= 32:  return "STRESSED"
    if score >= 20:  return "DELINQUENT"
    return "DEFAULT"


def _score_to_risk_tier(score: float, n_flags: int, income_cv: float) -> str:
    if score >= 70 and n_flags == 0 and income_cv < 0.25:
        return "low_risk"
    if score >= 50 and n_flags <= 1:
        return "medium_risk"
    if score >= 30 or n_flags <= 3:
        return "high_risk"
    return "subprime"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))
