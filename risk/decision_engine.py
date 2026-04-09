"""
Risk Decision Engine (PD-Based)
================================
Produces a final credit decision from simulation-derived PD metrics.

This supersedes the static scoring in ``core/decision_engine.py`` for the
document-analysis pathway by replacing the heuristic credit score with a
probabilistic default rate derived from Monte Carlo simulation.

Decision tiers (align with Basel-III PD bands):
    LOW      PD  < 10 %    — approve
    MEDIUM   10 % ≤ PD < 30 %  — approve with conditions / monitoring
    HIGH     30 % ≤ PD < 60 %  — decline / high-rate offer
    CRITICAL PD ≥ 60 %    — decline

Credit limit formula
--------------------
limit = avg_monthly_inflow × multiplier(survival_24m) × (1 - 0.5 × pd)

Multiplier table:
    survival_24m ≥ 0.85  →  6 ×
    survival_24m ≥ 0.70  →  4 ×
    survival_24m ≥ 0.50  →  2.5 ×
    survival_24m ≥ 0.30  →  1 ×
    otherwise             →  0   (decline)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from risk.risk_metrics import RiskMetrics

logger = logging.getLogger(__name__)

# ── Tier thresholds ───────────────────────────────────────────────────────────
_PD_LOW      = 0.10
_PD_MEDIUM   = 0.30
_PD_HIGH     = 0.60


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class StochasticDecision:
    """
    Credit decision derived from Monte Carlo simulation risk metrics.
    """
    decision:                 str         # APPROVE | REVIEW | REJECT
    risk_level:               str         # LOW | MEDIUM | HIGH | CRITICAL
    pd_score:                 float       # 0–100, higher = worse
    pd_probability:           float       # 0–1
    expected_survival_months: float
    survival_12m:             float
    survival_24m:             float
    confidence_interval:      tuple       # (p5_final_score, p95_final_score)
    credit_limit:             float
    credit_limit_currency:    str = "INR"
    reasons:                  List[str] = field(default_factory=list)
    simulation_summary:       dict       = field(default_factory=dict)


# ── Primary interface ─────────────────────────────────────────────────────────

def decide_from_metrics(
    metrics: RiskMetrics,
    avg_monthly_inflow: float,
    params: "StochasticParams | None" = None,    # noqa: F821
) -> StochasticDecision:
    """
    Derive a credit decision from :class:`risk.risk_metrics.RiskMetrics`.

    Parameters
    ----------
    metrics : RiskMetrics
        Output of :func:`risk.risk_metrics.compute_risk_metrics`.
    avg_monthly_inflow : float
        Borrower's average monthly income (used for credit limit calculation).
    params : StochasticParams, optional
        Original parameters; if provided, adds audit context to summary.

    Returns
    -------
    StochasticDecision
    """
    pd = metrics.pd_probability
    risk_level = _pd_to_risk_level(pd)
    decision   = _risk_to_decision(risk_level)
    reasons    = _build_reasons(metrics, risk_level, params)
    limit      = _compute_credit_limit(avg_monthly_inflow, pd, metrics.survival_24m)

    sim_summary = {
        "n_simulations":      metrics.n_simulations,
        "horizon_months":     metrics.horizon_months,
        "mean_final_score":   round(float(metrics.mean_curve[-1]), 2),
        "worst_case_score":   round(float(metrics.worst_case_curve[-1]), 2),
        "markov_pd_terminal": round(metrics.markov_pd_terminal * 100, 2),
    }
    if params is not None:
        sim_summary["initial_health_score"] = params.health_score
        sim_summary["risk_tier"]            = params.risk_tier
        sim_summary["markov_initial_state"] = params.markov_initial_state

    result = StochasticDecision(
        decision=decision,
        risk_level=risk_level,
        pd_score=metrics.pd_score,
        pd_probability=round(metrics.pd_probability, 6),
        expected_survival_months=metrics.expected_ttd,
        survival_12m=round(metrics.survival_12m * 100, 2),
        survival_24m=round(metrics.survival_24m * 100, 2),
        confidence_interval=(metrics.ci_p5, metrics.ci_p95),
        credit_limit=round(limit, 2),
        reasons=reasons,
        simulation_summary=sim_summary,
    )
    logger.info(
        "StochasticDecision: pd=%.1f%%  decision=%s  limit=%.0f",
        pd * 100, decision, limit,
    )
    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _pd_to_risk_level(pd: float) -> str:
    if pd < _PD_LOW:    return "LOW"
    if pd < _PD_MEDIUM: return "MEDIUM"
    if pd < _PD_HIGH:   return "HIGH"
    return "CRITICAL"


def _risk_to_decision(risk_level: str) -> str:
    return {
        "LOW":      "APPROVE",
        "MEDIUM":   "REVIEW",
        "HIGH":     "REJECT",
        "CRITICAL": "REJECT",
    }[risk_level]


def _compute_credit_limit(
    avg_monthly_inflow: float,
    pd: float,
    survival_24m: float,
) -> float:
    """
    Credit limit = inflow × multiplier × (1 − 0.5 × pd).
    Multiplier table is based on 24-month survival probability.
    """
    if survival_24m >= 0.85:  multiplier = 6.0
    elif survival_24m >= 0.70: multiplier = 4.0
    elif survival_24m >= 0.50: multiplier = 2.5
    elif survival_24m >= 0.30: multiplier = 1.0
    else:                      multiplier = 0.0

    limit = avg_monthly_inflow * multiplier * (1.0 - 0.5 * pd)
    return max(0.0, limit)


def _build_reasons(
    metrics: RiskMetrics,
    risk_level: str,
    params: "StochasticParams | None",
) -> list[str]:
    reasons: list[str] = []

    # PD narrative
    pd_pct = round(metrics.pd_probability * 100, 1)
    reasons.append(f"Monte Carlo PD: {pd_pct}% over {metrics.horizon_months} months.")

    # Survival narratives
    s12 = round(metrics.survival_12m * 100, 1)
    s24 = round(metrics.survival_24m * 100, 1)
    reasons.append(f"12-month survival: {s12}%.  24-month survival: {s24}%.")

    # TTD
    ttd = metrics.expected_ttd
    if ttd <= metrics.horizon_months:
        reasons.append(f"Expected time to financial distress: {ttd:.1f} months.")
    else:
        reasons.append("No distress event expected within the simulation window.")

    # Tail risk
    if metrics.ci_p5 < 20:
        reasons.append(
            f"Tail risk: worst-case score (P5) is {metrics.ci_p5:.1f} — "
            "below default threshold in stress scenarios."
        )

    # Markov Markov
    if metrics.markov_pd_terminal >= 0.25:
        reasons.append(
            f"Markov chain predicts {metrics.markov_pd_terminal*100:.1f}% "
            "probability of DEFAULT state at end of horizon."
        )

    # Risk-level-specific messages
    tier_msgs = {
        "LOW":      "Strong financial profile — qualifies for standard credit offer.",
        "MEDIUM":   "Moderate risk — credit offered with income monitoring clause.",
        "HIGH":     "Elevated probability of default — credit not recommended at standard terms.",
        "CRITICAL": "Severe credit risk — application declined.",
    }
    reasons.append(tier_msgs[risk_level])

    # Params hints
    if params is not None and params.health_score < 35:
        reasons.append(
            f"Initial financial health score is low ({params.health_score:.1f}/100)."
        )

    return reasons
