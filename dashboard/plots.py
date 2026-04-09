"""
STPA - Dashboard Plots
========================
Generates Plotly figures for the STPA dashboard.

Charts:
    1. Fan Chart        — 10k path simulation with confidence bands
    2. Survival Curve   — P(no default) over time
    3. Markov Heatmap   — State transition probability matrix
    4. Stress Bar Chart — PD comparison across scenarios
    5. State Distribution — Markov state evolution over time
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Optional


STPA_COLORS = {
    "primary":  "#00D4FF",
    "danger":   "#FF4444",
    "warning":  "#FFA500",
    "success":  "#00CC66",
    "bg":       "#0A0E1A",
    "surface":  "#111827",
    "text":     "#E2E8F0",
}


def fan_chart(path_stats: pd.DataFrame, title: str = "Financial Health Trajectory") -> go.Figure:
    """Fan chart showing distribution of simulated health score paths."""
    months = list(range(len(path_stats)))

    fig = go.Figure()

    # P5–P95 band (widest)
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=path_stats["p95"].tolist() + path_stats["p5"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th percentile",
        showlegend=True,
    ))

    # P25–P75 band
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=path_stats["p75"].tolist() + path_stats["p25"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th–75th percentile",
        showlegend=True,
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=months,
        y=path_stats["median"],
        line=dict(color=STPA_COLORS["primary"], width=2.5),
        name="Median path",
    ))

    # Mean
    fig.add_trace(go.Scatter(
        x=months,
        y=path_stats["mean"],
        line=dict(color="#FFFFFF", width=1.5, dash="dash"),
        name="Mean path",
    ))

    # Default threshold line
    fig.add_hline(
        y=20, line_dash="dot",
        line_color=STPA_COLORS["danger"],
        annotation_text="Default Threshold",
        annotation_position="right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Months",
        yaxis_title="Financial Health Score",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor=STPA_COLORS["bg"],
        plot_bgcolor=STPA_COLORS["surface"],
        font=dict(color=STPA_COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig


def survival_curve(survival: np.ndarray, title: str = "Survival Curve — P(No Default)") -> go.Figure:
    """Survival probability curve over simulation horizon."""
    months = list(range(len(survival)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months,
        y=(survival * 100).tolist(),
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0, 204, 102, 0.15)",
        line=dict(color=STPA_COLORS["success"], width=3),
        name="Survival Probability",
    ))

    fig.add_hline(y=50, line_dash="dot", line_color=STPA_COLORS["warning"],
                  annotation_text="50% survival")

    fig.update_layout(
        title=title,
        xaxis_title="Months",
        yaxis_title="P(No Default) %",
        yaxis=dict(range=[0, 105]),
        paper_bgcolor=STPA_COLORS["bg"],
        plot_bgcolor=STPA_COLORS["surface"],
        font=dict(color=STPA_COLORS["text"]),
    )
    return fig


def markov_heatmap(transition_matrix: np.ndarray, states: list) -> go.Figure:
    """Heatmap of the Markov transition matrix."""
    fig = go.Figure(go.Heatmap(
        z=transition_matrix,
        x=states,
        y=states,
        colorscale=[
            [0, STPA_COLORS["bg"]],
            [0.5, "#1E40AF"],
            [1, STPA_COLORS["primary"]]
        ],
        text=np.round(transition_matrix, 3).tolist(),
        texttemplate="%{text}",
        showscale=True,
    ))

    fig.update_layout(
        title="Markov State Transition Matrix",
        xaxis_title="To State",
        yaxis_title="From State",
        paper_bgcolor=STPA_COLORS["bg"],
        plot_bgcolor=STPA_COLORS["surface"],
        font=dict(color=STPA_COLORS["text"]),
    )
    return fig


def stress_comparison(scenario_table: pd.DataFrame) -> go.Figure:
    """Bar chart comparing PD scores across macro stress scenarios."""
    colors = []
    for pd in scenario_table["pd_score"]:
        if pd < 15:   colors.append(STPA_COLORS["success"])
        elif pd < 35: colors.append(STPA_COLORS["warning"])
        elif pd < 60: colors.append("#FF6B35")
        else:         colors.append(STPA_COLORS["danger"])

    fig = go.Figure(go.Bar(
        x=scenario_table["scenario"],
        y=scenario_table["pd_score"],
        marker_color=colors,
        text=[f"{v:.1f}" for v in scenario_table["pd_score"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="PD Score Across Macro Stress Scenarios",
        xaxis_title="Scenario",
        yaxis_title="PD Score (0–100)",
        yaxis=dict(range=[0, 105]),
        paper_bgcolor=STPA_COLORS["bg"],
        plot_bgcolor=STPA_COLORS["surface"],
        font=dict(color=STPA_COLORS["text"]),
    )
    return fig


def state_distribution_area(markov_dist: pd.DataFrame) -> go.Figure:
    """Stacked area chart of Markov state distribution over time."""
    state_colors = {
        "EXCELLENT":   "#00CC66",
        "GOOD":        "#66CC88",
        "FAIR":        "#FFAA00",
        "STRESSED":    "#FF8C00",
        "DELINQUENT":  "#FF5555",
        "DEFAULT":     "#CC0000",
        "RECOVERED":   "#4499FF",
    }

    fig = go.Figure()
    months = list(range(len(markov_dist)))

    for state in markov_dist.columns:
        fig.add_trace(go.Scatter(
            x=months,
            y=(markov_dist[state] * 100).tolist(),
            stackgroup="one",
            name=state,
            line=dict(color=state_colors.get(state, "#888888"), width=0),
            fillcolor=state_colors.get(state, "#888888"),
        ))

    fig.update_layout(
        title="Markov State Distribution Over Time",
        xaxis_title="Month",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor=STPA_COLORS["bg"],
        plot_bgcolor=STPA_COLORS["surface"],
        font=dict(color=STPA_COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig
