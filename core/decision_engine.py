"""
Decision Engine
===============
Converts financial features into a credit score, decision, credit limit,
and risk level using a weighted linear scoring model.

Scoring formula
---------------
score = (income_stability   * 0.30)
      + (savings_rate_norm  * 0.25)
      + (avg_balance_score  * 0.20)
      - (expense_ratio      * 0.15)
      - (risk_flag_penalty  * 0.10)

All components are normalised to [0, 1] before weighting.
Final score is scaled to [0, 100].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────

_APPROVE_THRESHOLD = 60     # score >= 60 → APPROVE
_REVIEW_THRESHOLD  = 35     # score >= 35 → REVIEW, else REJECT

# Credit limit multipliers (× avg monthly inflow)
_CREDIT_LIMIT_MULTIPLIERS = {
    "APPROVE": 5.0,
    "REVIEW":  2.0,
    "REJECT":  0.0,
}

# Risk level bands
_RISK_BANDS = [
    (75, "LOW"),
    (50, "MEDIUM"),
    (0,  "HIGH"),
]


# ── Output model ───────────────────────────────────────────────────────────────

@dataclass
class CreditDecision:
    credit_score: int
    decision: str                          # APPROVE | REVIEW | REJECT
    credit_limit: float
    risk_level: str                        # LOW | MEDIUM | HIGH
    reasons: list[str] = field(default_factory=list)
    score_breakdown: dict = field(default_factory=dict)


# ── Public interface ──────────────────────────────────────────────────────────

def decide(features: dict) -> CreditDecision:
    """
    Produce a credit decision from the financial feature vector.

    Parameters
    ----------
    features : dict
        Output of :func:`core.financial_analyzer.analyze`.

    Returns
    -------
    CreditDecision
    """
    components = _score_components(features)
    raw_score = _weighted_sum(components)
    credit_score = int(round(max(0.0, min(100.0, raw_score * 100))))

    decision = _classify(credit_score)
    risk_level = _risk_level(credit_score)
    credit_limit = _credit_limit(decision, features)
    reasons = _explain(features, components, credit_score)

    result = CreditDecision(
        credit_score=credit_score,
        decision=decision,
        credit_limit=round(credit_limit, 2),
        risk_level=risk_level,
        reasons=reasons,
        score_breakdown={k: round(v, 4) for k, v in components.items()},
    )

    logger.debug("CreditDecision: score=%d decision=%s", credit_score, decision)
    return result


# ── Scoring internals ─────────────────────────────────────────────────────────

def _score_components(f: dict) -> dict:
    """
    Extract and normalise scoring components from feature dict.
    All values returned are in [0, 1] (positive = better).
    """
    income_stability  = _clamp(f.get("income_stability", 0.0))
    savings_rate_norm = _clamp(f.get("savings_rate", 0.0))          # already –1…1
    savings_rate_norm = (savings_rate_norm + 1.0) / 2.0             # → 0…1
    avg_balance_score = _clamp(f.get("avg_balance_score", 0.0))
    expense_ratio     = _clamp(f.get("expense_ratio", 1.0))         # high = bad
    n_flags           = f.get("n_risk_flags", 0)
    risk_flag_penalty = _clamp(min(n_flags / 5.0, 1.0))            # 5+ flags → full penalty
    itr_confidence    = _clamp(f.get("itr_confidence", 0.0))        # weight-adjustment bonus

    return {
        "income_stability":  income_stability,
        "savings_rate_norm": savings_rate_norm,
        "avg_balance_score": avg_balance_score,
        "expense_ratio":     expense_ratio,
        "risk_flag_penalty": risk_flag_penalty,
        "itr_confidence":    itr_confidence,
    }


def _weighted_sum(c: dict) -> float:
    """
    Apply scoring weights and return a raw score in [0, 1].

    Weights sum to 1.00 (positive factors) minus negative factors.
    """
    positive = (
        c["income_stability"]  * 0.30 +
        c["savings_rate_norm"] * 0.25 +
        c["avg_balance_score"] * 0.20 +
        c["itr_confidence"]    * 0.05   # small bonus for verifiable ITR
    )
    negative = (
        c["expense_ratio"]     * 0.15 +
        c["risk_flag_penalty"] * 0.10
    )
    # Clamp to avoid floating-point edge cases
    return _clamp(positive - negative + 0.20)   # +0.20 baseline so empty docs don't auto-reject


def _classify(score: int) -> str:
    if score >= _APPROVE_THRESHOLD:
        return "APPROVE"
    if score >= _REVIEW_THRESHOLD:
        return "REVIEW"
    return "REJECT"


def _risk_level(score: int) -> str:
    for threshold, level in _RISK_BANDS:
        if score >= threshold:
            return level
    return "HIGH"


def _credit_limit(decision: str, features: dict) -> float:
    if decision == "REJECT":
        return 0.0
    multiplier = _CREDIT_LIMIT_MULTIPLIERS[decision]
    monthly_inflow = features.get("avg_monthly_inflow", 0.0)
    return monthly_inflow * multiplier


# ── Explanation ────────────────────────────────────────────────────────────────

def _explain(features: dict, components: dict, score: int) -> list[str]:
    """Generate human-readable reason strings."""
    reasons: list[str] = []

    sr = features.get("savings_rate", 0.0)
    if sr >= 0.25:
        reasons.append(f"Strong savings rate of {sr:.0%}.")
    elif sr >= 0.10:
        reasons.append(f"Moderate savings rate of {sr:.0%}; room for improvement.")
    else:
        reasons.append(f"Low savings rate ({sr:.0%}); high spending relative to income.")

    er = features.get("expense_ratio", 1.0)
    if er > 0.90:
        reasons.append("Expense ratio > 90 % — near-complete income consumption.")
    elif er > 0.75:
        reasons.append(f"Elevated expense ratio ({er:.0%}).")

    stab = features.get("income_stability", 0.0)
    if stab >= 0.75:
        reasons.append("Income inflows are highly consistent across months.")
    elif stab < 0.40:
        reasons.append("Irregular income pattern — high month-to-month variability.")

    bal = features.get("avg_balance_score", 0.0)
    if bal >= 0.70:
        reasons.append("Healthy average balance relative to monthly income.")
    elif bal < 0.30:
        reasons.append("Low average bank balance relative to income.")

    if features.get("min_balance", 0.0) < 0:
        reasons.append("Negative balance detected — indicates overdraft usage.")

    itr_conf = features.get("itr_confidence", 0.0)
    if itr_conf < 0.40:
        reasons.append("ITR data partially incomplete; income verification limited.")

    for flag in features.get("risk_flags", []):
        reasons.append(f"Risk flag: {flag.replace('_', ' ').title()}.")

    if not reasons:
        reasons.append(f"Score of {score} based on overall financial profile.")

    return reasons


# ── Utility ────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))
