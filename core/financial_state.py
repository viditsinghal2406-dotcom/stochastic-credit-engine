"""
Financial State Variable — F(t)
================================
Defines the stochastic state variable that represents a borrower's
financial health at any point in time.

F(t) is a composite score [0, 100] derived from four orthogonal
sub-components:

    F(t) = w1·I(t) + w2·B(t) + w3·S(t) + w4·D(t)

Where:
    I(t) = Income stability component     (inflow regularity)
    B(t) = Balance buffer component       (balance relative to outflow)
    S(t) = Savings behaviour component    (surplus retention)
    D(t) = Debt discipline component      (debt service capacity)

This decomposition lets parameter_mapper.py derive OU / jump parameters
from each sub-component independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Weights ────────────────────────────────────────────────────────────────────
_W_INCOME   = 0.30   # income stability
_W_BALANCE  = 0.25   # balance buffer
_W_SAVINGS  = 0.25   # savings behaviour
_W_DEBT     = 0.20   # debt discipline / behaviour


@dataclass
class FinancialState:
    """
    Complete state variable snapshot for one borrower at t=0.

    All sub-scores are in [0, 100]; composite_score is the weighted sum.
    """
    # ── Sub-component scores ──────────────────────────────────────────────────
    income_component:  float = 50.0   # I(t) — regularity and level of inflows
    balance_component: float = 50.0   # B(t) — reserve buffer adequacy
    savings_component: float = 50.0   # S(t) — savings rate quality
    debt_component:    float = 50.0   # D(t) — debt burden and repayment behaviour

    # ── Composite F(t=0) ─────────────────────────────────────────────────────
    composite_score: float = 50.0

    # ── Volatility estimates ─────────────────────────────────────────────────
    income_cv:      float = 0.3    # coefficient of variation of monthly inflow
    expense_cv:     float = 0.3    # estimated expense variability
    balance_ratio:  float = 1.0    # avg_balance / avg_monthly_inflow

    # ── Macro / behavioural flags ─────────────────────────────────────────────
    n_risk_flags:   int   = 0
    salary_detected: bool = False
    itr_confidence: float = 0.0

    # ── Raw inputs (kept for audit) ───────────────────────────────────────────
    avg_monthly_inflow:  float = 0.0
    avg_monthly_outflow: float = 0.0
    avg_balance:         float = 0.0
    min_balance:         float = 0.0
    annual_income_itr:   float = 0.0
    tax_paid:            float = 0.0
    months_covered:      int   = 12
    risk_flags:          list  = field(default_factory=list)


def build_financial_state(features: dict) -> FinancialState:
    """
    Build a :class:`FinancialState` from the feature dict produced by
    :func:`core.financial_analyzer.analyze`.

    Parameters
    ----------
    features : dict
        Normalised feature dictionary (scores in 0-1 range + raw figures).

    Returns
    -------
    FinancialState
    """
    inflow  = features.get("avg_monthly_inflow", 0.0)
    outflow = features.get("avg_monthly_outflow", 0.0)
    avg_bal = features.get("avg_balance", 0.0)
    min_bal = features.get("min_balance", 0.0)
    inflow_cv = 1.0 - features.get("income_stability", 0.5)  # cv ≈ 1-stability

    # ── I(t) : income component ────────────────────────────────────────────────
    # High, stable income → high score
    income_level_score = _log_income_score(inflow)                # 0-100
    income_stab_score  = features.get("income_stability", 0.5) * 100
    income_component   = 0.55 * income_level_score + 0.45 * income_stab_score

    # ── B(t) : balance buffer ──────────────────────────────────────────────────
    if outflow > 0:
        buffer_months = avg_bal / outflow          # how many months of expenses covered
    else:
        buffer_months = avg_bal / max(inflow, 1)
    balance_component = min(100.0, buffer_months * 25)  # 4 months buffer = 100 pts
    if min_bal < 0:
        balance_component = max(0.0, balance_component - 25)

    # ── S(t) : savings component ───────────────────────────────────────────────
    savings_rate = features.get("savings_rate", 0.0)
    savings_component = _clamp((savings_rate + 0.30) / 0.60 * 100)  # -30% → 0, +30% → 100

    # ── D(t) : debt discipline ─────────────────────────────────────────────────
    expense_ratio  = features.get("expense_ratio", 1.0)
    tax_compliance = features.get("tax_compliance", 1.0)
    debt_component = _clamp((1.0 - expense_ratio) * 70 + tax_compliance * 30)

    # ── Composite ──────────────────────────────────────────────────────────────
    composite = (
        _W_INCOME  * income_component  +
        _W_BALANCE * balance_component +
        _W_SAVINGS * savings_component +
        _W_DEBT    * debt_component
    )
    composite = _clamp(composite)

    # ── Expense CV (estimated as expense_ratio variance proxy) ─────────────────
    expense_cv = min(1.0, expense_ratio * inflow_cv + 0.05)

    balance_ratio = (avg_bal / inflow) if inflow > 0 else 0.0

    state = FinancialState(
        income_component=round(income_component, 2),
        balance_component=round(balance_component, 2),
        savings_component=round(savings_component, 2),
        debt_component=round(debt_component, 2),
        composite_score=round(composite, 2),
        income_cv=round(inflow_cv, 4),
        expense_cv=round(expense_cv, 4),
        balance_ratio=round(balance_ratio, 4),
        n_risk_flags=features.get("n_risk_flags", 0),
        salary_detected=features.get("salary_detected", False),
        itr_confidence=features.get("itr_confidence", 0.0),
        avg_monthly_inflow=round(inflow, 2),
        avg_monthly_outflow=round(outflow, 2),
        avg_balance=round(avg_bal, 2),
        min_balance=round(min_bal, 2),
        annual_income_itr=features.get("annual_income_itr", 0.0),
        tax_paid=features.get("tax_paid", 0.0),
        months_covered=features.get("months_covered", 12),
        risk_flags=features.get("risk_flags", []),
    )

    logger.debug("FinancialState: composite=%.1f  I=%.1f  B=%.1f  S=%.1f  D=%.1f",
                 composite, income_component, balance_component,
                 savings_component, debt_component)
    return state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log_income_score(monthly_inflow: float) -> float:
    """Map monthly inflow to a 0–100 score using log scale (INR context)."""
    import math
    if monthly_inflow <= 0:
        return 0.0
    # ₹10k → ~30, ₹50k → ~65, ₹1L → ~80, ₹3L+ → ~100
    score = math.log(monthly_inflow / 5_000 + 1) / math.log(61) * 100
    return min(100.0, max(0.0, score))


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))
