"""
Risk Metrics
============
Converts raw simulation paths into interpretable credit risk statistics.

Inputs:  SimulationResult  (from core.stochastic_engine)
Outputs: RiskMetrics dataclass with:

    pd_probability       — fraction of paths that ever breach DEFAULT_THRESHOLD
    pd_score             — pd_probability × 100  (0–100 scale)
    survival_curve       — P(no default by month t)  shape (horizon+1,)
    survival_12m         — survival probability at month 12
    survival_24m         — survival probability at month 24
    expected_ttd         — mean first-passage time (months) to default
    ci_p5, ci_p95        — 5th/95th percentile of final score distribution
    worst_case_curve     — 5th-percentile path at each month (shape horizon+1)
    mean_curve           — mean path at each month
    markov_pd_terminal   — Markov-chain DEFAULT probability at final month
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from core.diffusion_engine import DEFAULT_THRESHOLD
from risk.survival import SurvivalAnalyzer

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class RiskMetrics:
    pd_probability:       float          # P(default) over simulation horizon
    pd_score:             float          # pd_probability * 100
    survival_curve:       np.ndarray     # shape (horizon+1,)
    survival_12m:         float          # S(12)
    survival_24m:         float          # S(24); equals S(horizon) if horizon<24
    expected_ttd:         float          # expected time-to-default (months)
    ci_p5:                float          # 5th-percentile final score
    ci_p95:               float          # 95th-percentile final score
    worst_case_curve:     np.ndarray     # 5th-percentile path  shape (horizon+1,)
    mean_curve:           np.ndarray     # mean path             shape (horizon+1,)
    markov_pd_terminal:   float          # Markov DEFAULT prob at last month
    n_simulations:        int
    horizon_months:       int


# ── Primary interface ─────────────────────────────────────────────────────────

def compute_risk_metrics(
    sim: "SimulationResult",          # noqa: F821 — avoid circular import at module level
) -> RiskMetrics:
    """
    Derive credit risk statistics from a :class:`core.stochastic_engine.SimulationResult`.

    Parameters
    ----------
    sim : SimulationResult
        Raw simulation output from :class:`core.stochastic_engine.StochasticEngine`.

    Returns
    -------
    RiskMetrics
    """
    paths     = sim.paths          # (n_sim, horizon+1)
    threshold = sim.default_threshold
    n_sims, n_cols = paths.shape
    horizon   = n_cols - 1

    # ── Probability of Default ────────────────────────────────────────────────
    ever_defaulted = (paths < threshold).any(axis=1)
    pd_probability = float(ever_defaulted.mean())
    pd_score       = round(pd_probability * 100, 4)

    # ── Survival curve ────────────────────────────────────────────────────────
    analyzer       = SurvivalAnalyzer()
    survival_curve = analyzer.compute(paths)    # shape (horizon+1,)
    survival_12m   = _safe_survival(survival_curve, 12)
    survival_24m   = _safe_survival(survival_curve, 24)

    # ── Expected time to default ──────────────────────────────────────────────
    expected_ttd = _mean_first_passage(paths, threshold, horizon)

    # ── Confidence interval on final score ───────────────────────────────────
    final_scores = paths[:, -1]
    ci_p5  = float(np.percentile(final_scores, 5))
    ci_p95 = float(np.percentile(final_scores, 95))

    # ── Path statistics ───────────────────────────────────────────────────────
    worst_case_curve = np.percentile(paths, 5,  axis=0)
    mean_curve       = paths.mean(axis=0)

    # ── Markov terminal DEFAULT probability ───────────────────────────────────
    # markov_dist[-1] is a dict {state_name: prob} at month horizon
    markov_pd_terminal = 0.0
    if sim.markov_dist:
        last = sim.markov_dist[-1]
        markov_pd_terminal = float(last.get("DEFAULT", 0.0))

    metrics = RiskMetrics(
        pd_probability=round(pd_probability, 6),
        pd_score=round(pd_score, 2),
        survival_curve=survival_curve,
        survival_12m=round(survival_12m, 6),
        survival_24m=round(survival_24m, 6),
        expected_ttd=round(expected_ttd, 2),
        ci_p5=round(ci_p5, 2),
        ci_p95=round(ci_p95, 2),
        worst_case_curve=worst_case_curve,
        mean_curve=mean_curve,
        markov_pd_terminal=round(markov_pd_terminal, 6),
        n_simulations=sim.n_simulations,
        horizon_months=horizon,
    )
    logger.debug("RiskMetrics: pd=%.2f%% survival_24m=%.2f%%",
                 pd_score, survival_24m * 100)
    return metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean_first_passage(
    paths: np.ndarray,
    threshold: float,
    horizon: int,
) -> float:
    """
    Mean first-passage time to default (months).
    Paths that never default contribute *horizon + 1* to the average,
    which makes this a censored mean rather than a conditional mean.
    That is the standard credit-risk convention.
    """
    times = np.full(paths.shape[0], horizon + 1, dtype=float)
    for i, path in enumerate(paths):
        crossing = np.where(path < threshold)[0]
        if len(crossing) > 0:
            times[i] = float(crossing[0])
    return float(times.mean())


def _safe_survival(curve: np.ndarray, month: int) -> float:
    """Return survival probability at *month*, or last available value."""
    if month < len(curve):
        return float(curve[month])
    return float(curve[-1])
