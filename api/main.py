"""
STPA - FastAPI Application
===========================
REST API exposing the STPA simulation engine.

Endpoints:
    POST /simulate          — Run full STPA simulation for a borrower
    POST /stress-test       — Run all macro stress scenarios
    POST /portfolio         — Batch simulation for multiple borrowers
    GET  /health            — API health check
    GET  /scenarios         — List available stress scenarios
    POST /analyze           — Creditworthiness analysis: upload Bank PDF + ITR PDF
    POST /analyze/form      — Creditworthiness analysis: fill manual financial form

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import os
import time

from api.schemas import (
    AnalyzeResponse, BankSummaryOut, ITRSummaryOut, StpaHookOut,
    StochasticAnalysisOut, ManualFinancialInput,
    BorrowerInput, SimulationResponse,
    StressTestResponse, PortfolioInput, PortfolioResponse
)
from core.monte_carlo import MonteCarloEngine, BorrowerProfile
from risk.oracle import RiskOracle
from risk.stress_tester import StressTester, SCENARIOS


app = FastAPI(
    title="STPA — Stochastic Path Analyzer",
    description="Credit Default Risk via Stochastic Processes: Markov Chain + OU Diffusion + Jump Process",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize engines ─────────────────────────────────────────────────────────

mc_engine = MonteCarloEngine()
oracle = RiskOracle()
stress_engine = StressTester()


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serve the NovaCred Bank frontend (index.html)."""
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path, media_type="text/html")


# ── Health Check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "STPA API", "version": "1.0.0"}


# ── List Scenarios ─────────────────────────────────────────────────────────────

@app.get("/scenarios")
def list_scenarios():
    return {
        name: {
            "name": s.name,
            "description": s.description,
            "unemployment_delta": s.unemployment_delta,
            "interest_rate_delta": s.interest_rate_delta,
            "gdp_growth": s.gdp_growth,
        }
        for name, s in SCENARIOS.items()
    }


# ── Single Borrower Simulation ─────────────────────────────────────────────────

@app.post("/simulate", response_model=SimulationResponse)
def simulate(borrower: BorrowerInput):
    t0 = time.time()
    try:
        profile = BorrowerProfile(
            borrower_id=borrower.borrower_id,
            health_score=borrower.health_score,
            long_run_mean=borrower.long_run_mean,
            reversion_speed=borrower.reversion_speed,
            volatility=borrower.volatility,
            initial_state=borrower.initial_state,
            risk_tier=borrower.risk_tier,
            unemployment_delta=borrower.unemployment_delta,
            interest_rate_delta=borrower.interest_rate_delta,
            gdp_growth=borrower.gdp_growth,
            horizon_months=borrower.horizon_months,
            n_simulations=borrower.n_simulations,
        )
        result = mc_engine.run(profile)
        assessment = oracle.assess(result)

        return SimulationResponse(
            borrower_id=result.borrower_id,
            pd_score=result.pd_score,
            pd_probability=result.pd_probability,
            risk_tier=result.risk_tier,
            risk_label=assessment.risk_label,
            summary=assessment.summary,
            recommendation=assessment.recommendation,
            key_drivers=assessment.key_drivers,
            expected_ttd_months=result.expected_time_to_default,
            survival_12m=assessment.survival_12m,
            survival_24m=assessment.survival_24m,
            base_pd=result.base_pd,
            stressed_pd=result.stressed_pd,
            survival_curve=result.survival_curve.tolist(),
            path_mean=result.path_stats["mean"].tolist(),
            path_p5=result.path_stats["p5"].tolist(),
            path_p95=result.path_stats["p95"].tolist(),
            compute_time_ms=round((time.time() - t0) * 1000, 1),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Stress Test ────────────────────────────────────────────────────────────────

@app.post("/stress-test", response_model=StressTestResponse)
def stress_test(borrower: BorrowerInput):
    profile = BorrowerProfile(
        borrower_id=borrower.borrower_id,
        health_score=borrower.health_score,
        long_run_mean=borrower.long_run_mean,
        reversion_speed=borrower.reversion_speed,
        volatility=borrower.volatility,
        initial_state=borrower.initial_state,
        risk_tier=borrower.risk_tier,
        horizon_months=borrower.horizon_months,
        n_simulations=min(borrower.n_simulations, 2000),
    )
    results = stress_engine.run_all_scenarios(profile, n_simulations=2000)
    table = stress_engine.summary_table(results)

    return StressTestResponse(
        borrower_id=borrower.borrower_id,
        scenarios=table.to_dict(orient="records")
    )


# ── Portfolio Simulation ───────────────────────────────────────────────────────

@app.post("/portfolio", response_model=PortfolioResponse)
def portfolio(payload: PortfolioInput):
    profiles = [
        BorrowerProfile(
            borrower_id=b.borrower_id,
            health_score=b.health_score,
            long_run_mean=b.long_run_mean,
            reversion_speed=b.reversion_speed,
            volatility=b.volatility,
            initial_state=b.initial_state,
            risk_tier=b.risk_tier,
            n_simulations=min(b.n_simulations, 3000),
        )
        for b in payload.borrowers
    ]
    summary_df = mc_engine.run_portfolio(profiles)
    return PortfolioResponse(
        n_borrowers=len(profiles),
        results=summary_df.to_dict(orient="records"),
        avg_pd_score=round(float(summary_df["pd_score"].mean()), 2),
        portfolio_risk_tier=(
            "CRITICAL" if summary_df["pd_score"].mean() > 60 else
            "HIGH" if summary_df["pd_score"].mean() > 35 else
            "MEDIUM" if summary_df["pd_score"].mean() > 15 else "LOW"
        )
    )


# ── Shared analysis helper ─────────────────────────────────────────────────────

def _run_analysis(bank_summary: dict, itr_data: dict, input_mode: str) -> AnalyzeResponse:
    """
    Shared pipeline for both document-upload and form-based paths.
    Accepts already-parsed bank_summary and itr_data dicts.
    """
    import logging
    from core.financial_analyzer import analyze, simulate_risk
    from core.decision_engine import decide

    log = logging.getLogger(__name__)

    features  = analyze(bank_summary, itr_data)
    decision  = decide(features)
    stoch_raw = simulate_risk(features)

    bank_out = BankSummaryOut(
        total_inflow=bank_summary.get("total_inflow", 0.0),
        total_outflow=bank_summary.get("total_outflow", 0.0),
        avg_monthly_inflow=bank_summary.get("avg_monthly_inflow", 0.0),
        avg_monthly_outflow=bank_summary.get("avg_monthly_outflow", 0.0),
        avg_balance=bank_summary.get("avg_balance", 0.0),
        min_balance=bank_summary.get("min_balance", 0.0),
        months_covered=bank_summary.get("months_covered", 0),
        salary_detected=bank_summary.get("salary_detected", False),
        estimated_monthly_salary=bank_summary.get("estimated_monthly_salary", 0.0),
    )
    itr_out = ITRSummaryOut(
        annual_income=itr_data.get("annual_income"),
        gross_total_income=itr_data.get("gross_total_income"),
        total_income=itr_data.get("total_income"),
        tax_paid=itr_data.get("tax_paid"),
        assessment_year=str(itr_data.get("assessment_year") or ""),
        itr_confidence=itr_data.get("_confidence", 0.0),
    )

    # ── Stochastic analysis output ─────────────────────────────────────────────
    stochastic_out: StochasticAnalysisOut | None = None
    if "error" not in stoch_raw:
        try:
            stochastic_out = StochasticAnalysisOut(**stoch_raw)
        except Exception as exc:  # pragma: no cover
            log.warning("Could not build StochasticAnalysisOut: %s", exc)

    return AnalyzeResponse(
        credit_score=decision.credit_score,
        decision=decision.decision,
        credit_limit=stochastic_out.credit_limit if stochastic_out else decision.credit_limit,
        risk_level=stochastic_out.risk_level if stochastic_out else decision.risk_level,
        reasons=(stochastic_out.reasons if stochastic_out else decision.reasons),
        score_breakdown=decision.score_breakdown,
        bank_summary=bank_out,
        itr_summary=itr_out,
        stpa_hook=None,
        stochastic_analysis=stochastic_out,
        warnings=itr_data.get("_warnings", []),
        input_mode=input_mode,
    )


# ── Document Analysis (PDF upload) ────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Creditworthiness analysis via PDF documents",
    description=(
        "Upload a **Bank Statement PDF** and/or **ITR PDF**. "
        "At least one document is required. "
        "Provide passwords for encrypted PDFs. "
        "Use `POST /analyze/form` to submit figures manually instead."
    ),
    tags=["Document Analysis"],
)
async def analyze_documents(
    bank_pdf:      Optional[UploadFile] = File(None,  description="Bank statement PDF (optional if itr_pdf provided)"),
    itr_pdf:       Optional[UploadFile] = File(None,  description="ITR PDF (optional if bank_pdf provided)"),
    bank_password: Optional[str]        = Form(None,  description="Bank PDF password, if encrypted"),
    itr_password:  Optional[str]        = Form(None,  description="ITR PDF password, if encrypted"),
):
    from utils.pdf_utils import load_pdf, PDFPasswordRequired, PDFPasswordIncorrect, PDFParseError
    from core.bank_parser import parse_bank_statement
    from core.itr_parser import parse_itr

    # ── Require at least one document ─────────────────────────────────────────
    if bank_pdf is None and itr_pdf is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "At least one document is required: provide 'bank_pdf', 'itr_pdf', or both. "
                "To submit figures manually instead, use POST /analyze/form."
            ),
        )

    # ── Load bank PDF ──────────────────────────────────────────────────────────
    bank_summary: dict = {}
    if bank_pdf is not None:
        try:
            bank_pages = load_pdf(await bank_pdf.read(), password=bank_password)
        except PDFPasswordRequired:
            raise HTTPException(status_code=400, detail="Bank PDF is encrypted. Provide bank_password.")
        except PDFPasswordIncorrect:
            raise HTTPException(status_code=401, detail="Incorrect password for bank PDF.")
        except PDFParseError as exc:
            raise HTTPException(status_code=422, detail=f"Bank PDF parse error: {exc}")
        df, bank_summary = parse_bank_statement(bank_pages)
        if df.empty:
            bank_summary.setdefault("_warnings", [])
            bank_summary["_warnings"] = ["No transactions extracted from bank PDF; results may be inaccurate."]

    # ── Load ITR PDF ───────────────────────────────────────────────────────────
    itr_data: dict = {"_confidence": 0.0, "_warnings": [], "annual_income": None}
    if itr_pdf is not None:
        try:
            itr_pages = load_pdf(await itr_pdf.read(), password=itr_password)
        except PDFPasswordRequired:
            raise HTTPException(status_code=400, detail="ITR PDF is encrypted. Provide itr_password.")
        except PDFPasswordIncorrect:
            raise HTTPException(status_code=401, detail="Incorrect password for ITR PDF.")
        except PDFParseError as exc:
            raise HTTPException(status_code=422, detail=f"ITR PDF parse error: {exc}")
        itr_data = parse_itr(itr_pages)

    return _run_analysis(bank_summary, itr_data, input_mode="documents")


# ── Manual Form Analysis ───────────────────────────────────────────────────────

@app.post(
    "/analyze/form",
    response_model=AnalyzeResponse,
    summary="Creditworthiness analysis via manual financial form",
    description=(
        "Submit your financial figures directly — no PDFs needed. "
        "Fields marked **required** must be provided. "
        "The more fields you fill, the more accurate the credit assessment. "
        "Use `POST /analyze` to upload documents instead."
    ),
    tags=["Document Analysis"],
)
def analyze_form(form: ManualFinancialInput):
    from core.financial_analyzer import manual_input_to_summaries

    bank_summary, itr_data = manual_input_to_summaries(form)
    return _run_analysis(bank_summary, itr_data, input_mode="form")
