"""
Financial Analyzer
==================
Combines bank-statement summary and ITR data into a normalised feature
dictionary for the decision engine.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_HEALTHY_SAVINGS_RATE = 0.20          # >= 20 % → good saver
_HEALTHY_EXPENSE_RATIO = 0.70         # <= 70 % → acceptable spend
_MIN_AVG_BALANCE_MULTIPLIER = 1.0     # avg_balance >= 1× monthly_inflow → healthy
_BALANCE_STABILITY_CV_THRESHOLD = 0.5 # CV of monthly balance < 0.5 → stable


# ── Public interface ──────────────────────────────────────────────────────────

def analyze(bank_summary: dict, itr_data: dict) -> dict:
    """
    Compute a normalised feature vector for the credit decision engine.

    Parameters
    ----------
    bank_summary : dict
        Output of :func:`core.bank_parser._compute_summary`.
    itr_data : dict
        Output of :func:`core.itr_parser.parse_itr`.

    Returns
    -------
    dict
        Feature dictionary with scores in [0, 1] and risk flags.
    """
    inflow = bank_summary.get("avg_monthly_inflow", 0.0)
    outflow = bank_summary.get("avg_monthly_outflow", 0.0)
    avg_balance = bank_summary.get("avg_balance", 0.0)
    min_balance = bank_summary.get("min_balance", 0.0)
    inflow_cv = bank_summary.get("inflow_cv", 1.0)
    annual_income = itr_data.get("annual_income") or 0.0
    tax_paid = itr_data.get("tax_paid") or 0.0
    months = bank_summary.get("months_covered", 1) or 1

    # ── Core ratios ────────────────────────────────────────────────────────────
    savings_rate = _safe_ratio(inflow - outflow, inflow)          # (0, 1)
    expense_ratio = _safe_ratio(outflow, inflow)                  # (0, 1)

    # ── Income consistency (lower CV = more consistent) ────────────────────────
    income_stability = max(0.0, 1.0 - inflow_cv)                  # (0, 1)

    # ── Balance score (avg_balance vs monthly inflow) ──────────────────────────
    if inflow > 0:
        balance_ratio = avg_balance / inflow
        avg_balance_score = min(1.0, balance_ratio / 3.0)          # saturates at 3× monthly inflow
    else:
        avg_balance_score = 0.0

    # ── Balance stability (min vs avg) ─────────────────────────────────────────
    balance_stability = _safe_ratio(min_balance, avg_balance) if avg_balance > 0 else 0.0
    balance_stability = max(0.0, min(1.0, balance_stability))

    # ── ITR cross-validation ───────────────────────────────────────────────────
    # Compare bank inflow annualised vs declared income
    bank_annual_inflow = inflow * 12
    income_declared_ratio = (
        _safe_ratio(annual_income, bank_annual_inflow)
        if bank_annual_inflow > 0 else 1.0
    )
    # Ideal: ITR income ≈ bank inflow; large divergence is a flag
    itr_mismatch = abs(1.0 - income_declared_ratio) > 0.40

    # ── Tax compliance ─────────────────────────────────────────────────────────
    # Simple heuristic: tax_paid / annual_income should be > 0 for taxable income
    tax_compliance = 1.0
    if annual_income > 250_000:       # above basic exemption (India)
        expected_min_tax = _estimate_min_tax(annual_income)
        if tax_paid < expected_min_tax * 0.5:
            tax_compliance = 0.5      # under-payment flag

    # ── Risk flags ─────────────────────────────────────────────────────────────
    risk_flags: list[str] = []
    if min_balance < 0:
        risk_flags.append("NEGATIVE_BALANCE")
    if savings_rate < 0:
        risk_flags.append("NEGATIVE_SAVINGS")
    if expense_ratio > 0.95:
        risk_flags.append("EXTREME_EXPENSE_RATIO")
    if itr_mismatch:
        risk_flags.append("ITR_BANK_INCOME_MISMATCH")
    if inflow_cv > 0.8:
        risk_flags.append("UNSTABLE_INCOME")
    if tax_compliance < 1.0:
        risk_flags.append("TAX_UNDERPAYMENT")
    if bank_summary.get("months_covered", 0) < 3:
        risk_flags.append("INSUFFICIENT_HISTORY")

    features = {
        # Normalised scores (0–1)
        "savings_rate": round(max(-1.0, min(1.0, savings_rate)), 4),
        "expense_ratio": round(max(0.0, min(1.0, expense_ratio)), 4),
        "income_stability": round(income_stability, 4),
        "avg_balance_score": round(avg_balance_score, 4),
        "balance_stability": round(balance_stability, 4),
        "tax_compliance": round(tax_compliance, 4),
        # Raw figures
        "avg_monthly_inflow": round(inflow, 2),
        "avg_monthly_outflow": round(outflow, 2),
        "avg_balance": round(avg_balance, 2),
        "min_balance": round(min_balance, 2),
        "annual_income_itr": round(annual_income, 2),
        "bank_annual_inflow": round(bank_annual_inflow, 2),
        "tax_paid": round(tax_paid, 2),
        "months_covered": months,
        # Flags
        "risk_flags": risk_flags,
        "n_risk_flags": len(risk_flags),
        "itr_confidence": itr_data.get("_confidence", 0.0),
        "salary_detected": bank_summary.get("salary_detected", False),
        "estimated_monthly_salary": bank_summary.get("estimated_monthly_salary", 0.0),
    }

    logger.debug("Financial features: %s", features)
    return features


# ── STPA integration hook ──────────────────────────────────────────────────────

def simulate_risk(features: dict) -> dict:
    """
    Run a full Monte Carlo stochastic simulation from financial features.

    Builds a :class:`core.financial_state.FinancialState`, derives
    calibrated stochastic params, runs the hybrid OU + Jump + Markov engine,
    computes risk metrics, and returns a structured result dict.

    Parameters
    ----------
    features : dict
        Output of :func:`analyze`.

    Returns
    -------
    dict
        Simulation summary ready for API serialisation.
    """
    try:
        from core.financial_state import build_financial_state
        from core.parameter_mapper import map_features_to_params
        from core.stochastic_engine import StochasticEngine
        from risk.risk_metrics import compute_risk_metrics
        from risk.decision_engine import decide_from_metrics

        state   = build_financial_state(features)
        params  = map_features_to_params(state)
        engine  = StochasticEngine()
        sim     = engine.run_simulation(params)
        metrics = compute_risk_metrics(sim)

        avg_inflow = features.get("avg_monthly_inflow", 0.0)
        stoch_dec  = decide_from_metrics(metrics, avg_inflow, params)

        return {
            # Core risk metrics
            "pd_probability":           round(metrics.pd_probability, 6),
            "pd_score":                 round(metrics.pd_score, 2),
            "risk_level":               stoch_dec.risk_level,
            "decision":                 stoch_dec.decision,
            "credit_limit":             stoch_dec.credit_limit,
            # Survival
            "survival_12m":             round(metrics.survival_12m * 100, 2),
            "survival_24m":             round(metrics.survival_24m * 100, 2),
            "expected_survival_months": round(metrics.expected_ttd, 2),
            # Confidence interval
            "ci_p5":                    round(metrics.ci_p5, 2),
            "ci_p95":                   round(metrics.ci_p95, 2),
            # Path data (lists for JSON)
            "mean_path":                metrics.mean_curve.tolist(),
            "worst_case_path":          metrics.worst_case_curve.tolist(),
            "survival_curve":           metrics.survival_curve.tolist(),
            # Markov
            "markov_pd_terminal":       round(metrics.markov_pd_terminal * 100, 2),
            # Parameters audit
            "initial_health_score":     params.health_score,
            "markov_initial_state":     params.markov_initial_state,
            "risk_tier":                params.risk_tier,
            # Decision details
            "reasons":                  stoch_dec.reasons,
            "simulation_summary":       stoch_dec.simulation_summary,
        }
    except Exception as exc:          # pragma: no cover
        logger.error("simulate_risk failed: %s", exc, exc_info=True)
        return {
            "error": str(exc),
            "stpa_ready": False,
        }


# ── Private helpers ────────────────────────────────────────────────────────────

def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _estimate_min_tax(annual_income: float) -> float:
    """Rough old-regime Indian income tax (for compliance heuristic only)."""
    tax = 0.0
    slabs = [
        (250_000, 0.0),
        (500_000, 0.05),
        (1_000_000, 0.20),
        (float("inf"), 0.30),
    ]
    prev = 250_000
    for upper, rate in slabs:
        if annual_income <= prev:
            break
        taxable = min(annual_income, upper) - prev
        tax += taxable * rate
        prev = upper
    return tax


def _features_to_health_score(features: dict) -> float:
    """Map financial features to a 0–100 health score for STPA."""
    score = 50.0
    score += features.get("savings_rate", 0.0) * 20
    score += features.get("avg_balance_score", 0.0) * 15
    score -= features.get("expense_ratio", 0.5) * 10
    score -= features.get("n_risk_flags", 0) * 5
    return max(0.0, min(100.0, score))


def _features_to_volatility(features: dict) -> float:
    """Map income instability to OU volatility σ."""
    base = 6.0
    cv = 1.0 - features.get("income_stability", 0.5)
    return base + cv * 14.0          # range approximately 6–20


def _features_to_risk_tier(features: dict) -> str:
    n_flags = features.get("n_risk_flags", 0)
    savings = features.get("savings_rate", 0.0)
    if n_flags == 0 and savings >= 0.25:
        return "low_risk"
    if n_flags <= 1 and savings >= 0.10:
        return "medium_risk"
    if n_flags <= 3:
        return "high_risk"
    return "subprime"


# ── Manual form → summary dicts ────────────────────────────────────────────────

def manual_input_to_summaries(form: "ManualFinancialInput") -> tuple[dict, dict]:  # type: ignore[name-defined]
    """
    Convert a :class:`api.schemas.ManualFinancialInput` instance into the
    ``(bank_summary, itr_data)`` dictionaries expected by :func:`analyze`.

    This lets the manual-form path reuse the exact same analysis pipeline
    as the PDF-upload path without duplication.

    Parameters
    ----------
    form : ManualFinancialInput
        Validated Pydantic model from ``POST /analyze/form``.

    Returns
    -------
    tuple[dict, dict]
        ``(bank_summary, itr_data)`` — same shapes as parser outputs.
    """
    total_inflow  = form.avg_monthly_inflow  * form.months_covered
    total_outflow = form.avg_monthly_outflow * form.months_covered

    # Inflow CV is unknown from manual input; assume moderate stability
    if form.avg_monthly_inflow > 0:
        # If salary matches inflow closely, treat as stable
        salary_ratio = (form.estimated_monthly_salary / form.avg_monthly_inflow
                        if form.estimated_monthly_salary else 0.0)
        inflow_cv = 0.15 if salary_ratio > 0.7 else 0.40
    else:
        inflow_cv = 1.0

    bank_summary: dict = {
        "total_inflow":             round(total_inflow, 2),
        "total_outflow":            round(total_outflow, 2),
        "avg_monthly_inflow":       round(form.avg_monthly_inflow, 2),
        "avg_monthly_outflow":      round(form.avg_monthly_outflow, 2),
        "avg_balance":              round(form.avg_balance, 2),
        "min_balance":              round(form.min_balance, 2),
        "max_balance":              round(form.avg_balance * 1.5, 2),   # estimate
        "n_transactions":           0,                                   # not available
        "months_covered":           form.months_covered,
        "salary_detected":          form.salary_detected,
        "estimated_monthly_salary": round(form.estimated_monthly_salary, 2),
        "inflow_cv":                inflow_cv,
    }

    itr_data: dict = {
        "gross_total_income": form.gross_total_income,
        "total_income":       form.annual_income,
        "salary_income":      (form.estimated_monthly_salary * 12
                               if form.estimated_monthly_salary else None),
        "business_income":    None,
        "other_income":       None,
        "exempt_income":      None,
        "tax_payable":        None,
        "tax_paid":           form.tax_paid,
        "refund_due":         None,
        "assessment_year":    form.assessment_year,
        "pan":                None,
        "annual_income":      form.annual_income or form.gross_total_income,
        "_confidence":        _itr_confidence_from_form(form),
        "_raw_text_length":   0,
        "_warnings":          _itr_warnings_from_form(form),
    }

    return bank_summary, itr_data


def _itr_confidence_from_form(form: "ManualFinancialInput") -> float:  # type: ignore[name-defined]
    """Compute ITR confidence based on how many ITR fields were filled."""
    fields = [form.annual_income, form.gross_total_income, form.tax_paid,
              form.assessment_year, form.estimated_monthly_salary]
    filled = sum(1 for f in fields if f is not None and f != 0.0 and f != "")
    return round(filled / len(fields), 2)


def _itr_warnings_from_form(form: "ManualFinancialInput") -> list[str]:  # type: ignore[name-defined]
    warnings: list[str] = []
    if form.annual_income is None:
        warnings.append("annual_income not provided; ITR verification limited.")
    if form.tax_paid is None:
        warnings.append("tax_paid not provided; tax compliance check skipped.")
    if not form.assessment_year:
        warnings.append("assessment_year not provided.")
    return warnings
