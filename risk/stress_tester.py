"""
STPA - Stress Tester
=====================
Runs pre-defined macro stress scenarios and compares PD scores
across baseline and stressed conditions.

Scenarios:
    - BASELINE:       Normal economic conditions
    - MILD_RECESSION: Slight economic downturn
    - SEVERE_RECESSION: Deep recession (2008-style)
    - RATE_SHOCK:     Rapid interest rate hike
    - UNEMPLOYMENT_SPIKE: Sudden unemployment surge
    - COMBINED_SHOCK: Multiple stressors simultaneously
"""

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

from core.monte_carlo import MonteCarloEngine, BorrowerProfile, STPAResult


@dataclass
class MacroScenario:
    name: str
    description: str
    unemployment_delta: float = 0.0
    interest_rate_delta: float = 0.0
    gdp_growth: float = 0.0


SCENARIOS: Dict[str, MacroScenario] = {
    "baseline": MacroScenario(
        name="Baseline",
        description="Normal economic conditions — no macro stress applied",
        unemployment_delta=0.0,
        interest_rate_delta=0.0,
        gdp_growth=0.02,
    ),
    "mild_recession": MacroScenario(
        name="Mild Recession",
        description="Moderate slowdown with slight unemployment rise",
        unemployment_delta=0.02,
        interest_rate_delta=0.01,
        gdp_growth=-0.01,
    ),
    "severe_recession": MacroScenario(
        name="Severe Recession",
        description="Deep recession — 2008-style financial crisis conditions",
        unemployment_delta=0.06,
        interest_rate_delta=0.03,
        gdp_growth=-0.04,
    ),
    "rate_shock": MacroScenario(
        name="Rate Shock",
        description="Aggressive central bank rate hikes (200bps+)",
        unemployment_delta=0.01,
        interest_rate_delta=0.04,
        gdp_growth=0.01,
    ),
    "unemployment_spike": MacroScenario(
        name="Unemployment Spike",
        description="Sudden sectoral layoffs and unemployment surge",
        unemployment_delta=0.08,
        interest_rate_delta=0.01,
        gdp_growth=-0.02,
    ),
    "combined_shock": MacroScenario(
        name="Combined Shock",
        description="Stagflation: high rates + high unemployment + contraction",
        unemployment_delta=0.07,
        interest_rate_delta=0.05,
        gdp_growth=-0.03,
    ),
}


class StressTester:
    """
    Runs macro stress scenarios for a borrower and compares outcomes.

    Usage:
        tester = StressTester()
        results = tester.run_all_scenarios(borrower_profile)
        report = tester.summary_table(results)
    """

    def __init__(self):
        self.engine = MonteCarloEngine()

    def run_scenario(
        self,
        profile: BorrowerProfile,
        scenario: MacroScenario,
        n_simulations: int = 3000
    ) -> STPAResult:
        stressed = BorrowerProfile(
            borrower_id=profile.borrower_id,
            health_score=profile.health_score,
            long_run_mean=profile.long_run_mean,
            reversion_speed=profile.reversion_speed,
            volatility=profile.volatility,
            initial_state=profile.initial_state,
            risk_tier=profile.risk_tier,
            unemployment_delta=scenario.unemployment_delta,
            interest_rate_delta=scenario.interest_rate_delta,
            gdp_growth=scenario.gdp_growth,
            horizon_months=profile.horizon_months,
            n_simulations=n_simulations,
        )
        return self.engine.run(stressed)

    def run_all_scenarios(
        self,
        profile: BorrowerProfile,
        n_simulations: int = 3000
    ) -> Dict[str, STPAResult]:
        return {
            name: self.run_scenario(profile, scenario, n_simulations)
            for name, scenario in SCENARIOS.items()
        }

    def summary_table(self, scenario_results: Dict[str, STPAResult]) -> pd.DataFrame:
        rows = []
        for scenario_key, result in scenario_results.items():
            scenario = SCENARIOS[scenario_key]
            rows.append({
                "scenario": scenario.name,
                "description": scenario.description,
                "pd_score": result.pd_score,
                "risk_tier": result.risk_tier,
                "expected_ttd_months": result.expected_time_to_default,
                "survival_12m": round(float(result.survival_curve[min(12, len(result.survival_curve)-1)]) * 100, 1),
            })
        return pd.DataFrame(rows).sort_values("pd_score")
