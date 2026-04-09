"""
STPA - Risk Oracle
===================
Final risk scoring layer. Converts raw Monte Carlo output into
human-readable risk assessments, scores, and recommendations.

Output structure:
    - PD Score (0–100)
    - Risk Tier: LOW / MEDIUM / HIGH / CRITICAL
    - Risk Label + Explanation
    - Action Recommendation
    - Key Risk Drivers
"""

from dataclasses import dataclass
from typing import List, Dict
from core.monte_carlo import STPAResult


@dataclass
class RiskAssessment:
    """Human-readable risk output from the Oracle."""
    borrower_id: str
    pd_score: float
    pd_probability: float
    risk_tier: str
    risk_label: str
    summary: str
    recommendation: str
    key_drivers: List[str]
    stressed_pd: float
    stress_uplift: float          # How much worse under recession
    expected_ttd_months: float
    survival_12m: float           # P(no default) at 12 months
    survival_24m: float           # P(no default) at 24 months


RISK_TIER_META = {
    "LOW": {
        "label": "✅ Low Risk",
        "summary": "Borrower shows strong financial health with low probability of default.",
        "recommendation": "Approve loan. Standard monitoring sufficient."
    },
    "MEDIUM": {
        "label": "⚠️ Medium Risk",
        "summary": "Borrower has moderate risk. Financial health shows some vulnerability to shocks.",
        "recommendation": "Approve with conditions. Monthly monitoring recommended. Consider lower credit limit."
    },
    "HIGH": {
        "label": "🔴 High Risk",
        "summary": "Elevated default probability. Borrower's financial path shows significant stress under simulation.",
        "recommendation": "Decline or require collateral. Flag for early intervention if existing customer."
    },
    "CRITICAL": {
        "label": "🚨 Critical Risk",
        "summary": "Very high default probability. Financial trajectory crosses default threshold in majority of simulations.",
        "recommendation": "Decline application. If existing customer, initiate proactive restructuring conversation."
    }
}


def _identify_key_drivers(result: STPAResult, params) -> List[str]:
    drivers = []

    if params.health_score < 40:
        drivers.append(f"Low initial health score ({params.health_score:.0f}/100)")
    if params.health_score > params.long_run_mean + 15:
        drivers.append("Health score significantly above long-run mean — reversion risk")
    if params.volatility > 12:
        drivers.append(f"High income/financial volatility (σ={params.volatility:.1f})")
    if params.reversion_speed < 0.15:
        drivers.append("Slow recovery speed — slow to bounce back from shocks")
    if result.stressed_pd - result.base_pd > 20:
        drivers.append(f"High macro sensitivity (+{result.stressed_pd - result.base_pd:.1f}pts under recession)")
    if params.risk_tier in ("high_risk", "subprime"):
        drivers.append(f"High shock exposure profile ({params.risk_tier.replace('_', ' ').title()})")
    if result.expected_time_to_default < 12:
        drivers.append(f"Expected default in under 12 months ({result.expected_time_to_default:.1f}mo)")
    if not drivers:
        drivers.append("No major individual risk drivers identified")

    return drivers


class RiskOracle:
    """
    Translates simulation results into structured risk assessments.

    Usage:
        oracle = RiskOracle()
        assessment = oracle.assess(stpa_result)
    """

    def assess(self, result: STPAResult) -> RiskAssessment:
        meta = RISK_TIER_META[result.risk_tier]
        params = result.params

        survival_12m = float(result.survival_curve[min(12, len(result.survival_curve)-1)])
        survival_24m = float(result.survival_curve[min(24, len(result.survival_curve)-1)])
        stress_uplift = result.stressed_pd - result.base_pd

        drivers = _identify_key_drivers(result, params)

        return RiskAssessment(
            borrower_id=result.borrower_id,
            pd_score=result.pd_score,
            pd_probability=result.pd_probability,
            risk_tier=result.risk_tier,
            risk_label=meta["label"],
            summary=meta["summary"],
            recommendation=meta["recommendation"],
            key_drivers=drivers,
            stressed_pd=result.stressed_pd,
            stress_uplift=round(stress_uplift, 2),
            expected_ttd_months=result.expected_time_to_default,
            survival_12m=round(survival_12m, 4),
            survival_24m=round(survival_24m, 4),
        )

    def assess_portfolio(self, results: list) -> list:
        return [self.assess(r) for r in results]

    def format_report(self, assessment: RiskAssessment) -> str:
        """Returns a formatted text report for the assessment."""
        lines = [
            f"{'='*60}",
            f"  STPA RISK ASSESSMENT — Borrower {assessment.borrower_id}",
            f"{'='*60}",
            f"  Risk Label   : {assessment.risk_label}",
            f"  PD Score     : {assessment.pd_score:.1f} / 100",
            f"  PD Prob      : {assessment.pd_probability*100:.1f}%",
            f"  TTD (months) : {assessment.expected_ttd_months:.1f}",
            f"  Survival 12m : {assessment.survival_12m*100:.1f}%",
            f"  Survival 24m : {assessment.survival_24m*100:.1f}%",
            f"  Stressed PD  : {assessment.stressed_pd:.1f} (+{assessment.stress_uplift:.1f}pts)",
            f"{'─'*60}",
            f"  Summary      : {assessment.summary}",
            f"  Action       : {assessment.recommendation}",
            f"{'─'*60}",
            f"  Key Drivers  :",
        ]
        for d in assessment.key_drivers:
            lines.append(f"    • {d}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)
