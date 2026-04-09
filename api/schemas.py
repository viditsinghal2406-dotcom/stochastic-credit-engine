"""
STPA - API Schemas
===================
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
import re


# ── Input Schemas ──────────────────────────────────────────────────────────────

class BorrowerInput(BaseModel):
    borrower_id: str = Field(default="B001", description="Unique borrower identifier")
    health_score: float = Field(default=60.0, ge=0, le=100, description="Current financial health score (0–100)")
    long_run_mean: float = Field(default=55.0, ge=0, le=100, description="Borrower's long-run average health score")
    reversion_speed: float = Field(default=0.25, ge=0.01, le=5.0, description="OU mean-reversion speed θ")
    volatility: float = Field(default=8.0, ge=0.1, le=50.0, description="Health score volatility σ")
    initial_state: str = Field(default="FAIR", description="Starting Markov credit state")
    risk_tier: str = Field(default="medium_risk", description="Shock profile: low_risk/medium_risk/high_risk/subprime")
    unemployment_delta: float = Field(default=0.0, description="Macro: change in unemployment rate")
    interest_rate_delta: float = Field(default=0.0, description="Macro: change in interest rate")
    gdp_growth: float = Field(default=0.0, description="Macro: GDP growth rate")
    horizon_months: int = Field(default=24, ge=1, le=60, description="Simulation horizon in months")
    n_simulations: int = Field(default=5000, ge=100, le=10000, description="Number of Monte Carlo paths")

    @validator("initial_state")
    def validate_state(cls, v):
        valid = ["EXCELLENT", "GOOD", "FAIR", "STRESSED", "DELINQUENT", "DEFAULT", "RECOVERED"]
        if v.upper() not in valid:
            raise ValueError(f"initial_state must be one of {valid}")
        return v.upper()

    @validator("risk_tier")
    def validate_tier(cls, v):
        valid = ["low_risk", "medium_risk", "high_risk", "subprime"]
        if v not in valid:
            raise ValueError(f"risk_tier must be one of {valid}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "borrower_id": "B001",
                "health_score": 55.0,
                "long_run_mean": 50.0,
                "reversion_speed": 0.3,
                "volatility": 10.0,
                "initial_state": "STRESSED",
                "risk_tier": "high_risk",
                "horizon_months": 24,
                "n_simulations": 5000
            }
        }


class PortfolioInput(BaseModel):
    borrowers: List[BorrowerInput]


# ── Response Schemas ──────────────────────────────────────────────────────────

class SimulationResponse(BaseModel):
    borrower_id: str
    pd_score: float
    pd_probability: float
    risk_tier: str
    risk_label: str
    summary: str
    recommendation: str
    key_drivers: List[str]
    expected_ttd_months: float
    survival_12m: float
    survival_24m: float
    base_pd: float
    stressed_pd: float
    survival_curve: List[float]
    path_mean: List[float]
    path_p5: List[float]
    path_p95: List[float]
    compute_time_ms: float


class StressTestResponse(BaseModel):
    borrower_id: str
    scenarios: List[dict]


class PortfolioResponse(BaseModel):
    n_borrowers: int
    results: List[dict]
    avg_pd_score: float
    portfolio_risk_tier: str


# ── Document Analysis Schemas ─────────────────────────────────────────────────

class BankSummaryOut(BaseModel):
    total_inflow: float
    total_outflow: float
    avg_monthly_inflow: float
    avg_monthly_outflow: float
    avg_balance: float
    min_balance: float
    months_covered: int
    salary_detected: bool
    estimated_monthly_salary: float


class ITRSummaryOut(BaseModel):
    annual_income: Optional[float]
    gross_total_income: Optional[float]
    total_income: Optional[float]
    tax_paid: Optional[float]
    assessment_year: Optional[str]
    itr_confidence: float


class StpaHookOut(BaseModel):
    stpa_ready: bool
    suggested_health_score: float
    suggested_volatility: float
    suggested_risk_tier: str
    suggested_long_run_mean: float
    suggested_reversion_speed: float
    note: str


class StochasticAnalysisOut(BaseModel):
    """Full stochastic simulation results from the Monte Carlo engine."""
    pd_probability:           float
    pd_score:                 float          # 0–100
    risk_level:               str            # LOW / MEDIUM / HIGH / CRITICAL
    decision:                 str            # APPROVE / REVIEW / REJECT
    credit_limit:             float
    survival_12m:             float          # percentage
    survival_24m:             float          # percentage
    expected_survival_months: float
    ci_p5:                    float          # 5th-percentile final score
    ci_p95:                   float          # 95th-percentile final score
    mean_path:                List[float]
    worst_case_path:          List[float]
    survival_curve:           List[float]
    markov_pd_terminal:       float          # %
    initial_health_score:     float
    markov_initial_state:     str
    risk_tier:                str
    reasons:                  List[str]
    simulation_summary:       dict


class AnalyzeResponse(BaseModel):
    credit_score: int
    decision: str                          # APPROVE | REVIEW | REJECT (static)
    credit_limit: float
    risk_level: str                        # LOW | MEDIUM | HIGH
    reasons: List[str]
    score_breakdown: dict
    bank_summary: BankSummaryOut
    itr_summary: ITRSummaryOut
    stpa_hook: Optional[StpaHookOut]       # legacy stub; None when stochastic run
    stochastic_analysis: Optional[StochasticAnalysisOut]  # full MC results
    warnings: List[str]
    input_mode: str                        # "documents" | "form"


# ── Manual Form Input Schema ──────────────────────────────────────────────────

class ManualFinancialInput(BaseModel):
    """
    Manual financial form — alternative to uploading PDFs.
    All monetary fields are in the same currency (e.g. INR / USD).
    """

    # ── Identity (optional, for tracking only) ────────────────────────────────
    applicant_name: Optional[str] = Field(None, max_length=100, description="Full name of applicant")
    applicant_id:   Optional[str] = Field(None, max_length=50,  description="Applicant reference ID")

    # ── Bank statement fields (required) ─────────────────────────────────────
    avg_monthly_inflow:  float = Field(..., gt=0,   description="Average monthly credits/deposits")
    avg_monthly_outflow: float = Field(..., ge=0,   description="Average monthly debits/withdrawals")
    avg_balance:         float = Field(..., ge=0,   description="Average bank balance over the period")
    min_balance:         float = Field(0.0,         description="Minimum balance observed (can be negative for overdraft)")
    months_covered:      int   = Field(12, ge=1, le=120, description="Number of months of bank history provided")

    # ── Salary / income (optional enrichment) ────────────────────────────────
    salary_detected:          bool  = Field(False, description="Whether a regular salary credit is present")
    estimated_monthly_salary: float = Field(0.0, ge=0, description="Estimated monthly salary amount")

    # ── ITR / tax fields (optional) ───────────────────────────────────────────
    annual_income:       Optional[float] = Field(None, ge=0, description="Net taxable income from ITR")
    gross_total_income:  Optional[float] = Field(None, ge=0, description="Gross total income from ITR")
    tax_paid:            Optional[float] = Field(None, ge=0, description="Total tax paid / TDS")
    assessment_year:     Optional[str]   = Field(None, description="Assessment year, e.g. '2024-25'")

    # ── Validators ────────────────────────────────────────────────────────────

    @validator("avg_monthly_outflow")
    def outflow_not_exceed_double_inflow(cls, v, values):
        inflow = values.get("avg_monthly_inflow", 0)
        if inflow > 0 and v > inflow * 3:
            raise ValueError(
                "avg_monthly_outflow cannot exceed 3× avg_monthly_inflow — "
                "please verify the figures."
            )
        return v

    @validator("assessment_year")
    def validate_ay(cls, v):
        if v and not re.fullmatch(r"\d{4}-\d{2,4}", v):
            raise ValueError("assessment_year must be in format YYYY-YY, e.g. '2024-25'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "applicant_name": "Vidit Sharma",
                "applicant_id": "APPL-001",
                "avg_monthly_inflow": 85000,
                "avg_monthly_outflow": 55000,
                "avg_balance": 120000,
                "min_balance": 8000,
                "months_covered": 12,
                "salary_detected": True,
                "estimated_monthly_salary": 80000,
                "annual_income": 980000,
                "gross_total_income": 1020000,
                "tax_paid": 95000,
                "assessment_year": "2024-25"
            }
        }
