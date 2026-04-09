"""
Bank Statement Parser
=====================
Parses raw text extracted from a bank-statement PDF into a structured
DataFrame and computes key financial aggregates.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from utils.cleaners import clean_line, detect_dr_cr, parse_amount, parse_date, split_columns

logger = logging.getLogger(__name__)

# ── Salary detection heuristics ───────────────────────────────────────────────

_SALARY_KEYWORDS = re.compile(
    r"\b(salary|sal|payroll|pay|stipend|wages|neft.*sal|sal.*neft)\b",
    re.IGNORECASE,
)

# ── Minimum-column line filter ────────────────────────────────────────────────

_MIN_COLS = 3          # must find at least this many columns to consider a row


# ── Public interface ──────────────────────────────────────────────────────────

def parse_bank_statement(pages: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Parse bank statement pages and return:

    Parameters
    ----------
    pages : list[str]
        Raw page texts from :func:`utils.pdf_utils.load_pdf`.

    Returns
    -------
    transactions : pd.DataFrame
        Columns: date, description, debit, credit, balance
    summary : dict
        Aggregated statistics (inflow, outflow, avg_balance, …)
    """
    raw_rows = _extract_rows(pages)
    df = _build_dataframe(raw_rows)
    df = _infer_direction(df)
    summary = _compute_summary(df)
    return df, summary


# ── Row extraction ─────────────────────────────────────────────────────────────

def _extract_rows(pages: list[str]) -> list[dict]:
    """Iterate over lines and extract transaction-like rows."""
    rows: list[dict] = []
    for page in pages:
        for line in page.splitlines():
            line = clean_line(line)
            if not line:
                continue
            row = _parse_line(line)
            if row:
                rows.append(row)
    return rows


def _parse_line(line: str) -> Optional[dict]:
    """
    Attempt to parse a single text line as a bank transaction.

    Strategy
    --------
    1. Find a date anywhere in the line.
    2. Find 2–3 currency amounts (debit / credit / balance).
    3. Treat the text between date and first amount as description.
    """
    date_str = parse_date(line)
    if not date_str:
        return None

    # Strip date from line to isolate rest
    date_pattern = re.compile(
        r"\b\d{1,2}[/\-]\d{2}[/\-]\d{2,4}\b"
        r"|\b\d{4}[/\-]\d{2}[/\-]\d{2}\b"
        r"|\b\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\b"
    )
    rest = date_pattern.sub("", line, count=1).strip()

    # Find all currency amounts in rest
    amounts = re.findall(r"[\d,]+\.\d{2}", rest)
    if len(amounts) < 2:
        return None

    # Parse amounts to float
    float_amounts = []
    for a in amounts:
        v = parse_amount(a)
        if v is not None:
            float_amounts.append(v)

    # Remove amounts from rest → description
    description = re.sub(r"[\d,]+\.\d{2}", "", rest).strip()
    description = re.sub(r"\s+", " ", description)

    # Assign debit / credit / balance heuristically
    # Common layout: [debit_or_blank]  [credit_or_blank]  [balance]
    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None

    direction = detect_dr_cr(line)

    if len(float_amounts) >= 3:
        debit, credit, balance = float_amounts[0], float_amounts[1], float_amounts[2]
    elif len(float_amounts) == 2:
        balance = float_amounts[-1]
        if direction == "DR":
            debit = float_amounts[0]
        elif direction == "CR":
            credit = float_amounts[0]
        else:
            # Guess: larger amount is balance
            if float_amounts[0] > float_amounts[1]:
                balance = float_amounts[0]
                debit = float_amounts[1]
            else:
                balance = float_amounts[1]
                debit = float_amounts[0]

    return {
        "date": date_str,
        "description": description,
        "debit": debit,
        "credit": credit,
        "balance": balance,
    }


# ── DataFrame construction ─────────────────────────────────────────────────────

def _build_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "description", "debit", "credit", "balance"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Direction inference ────────────────────────────────────────────────────────

def _infer_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    When debit and credit columns are ambiguous (all in one column),
    use balance delta to re-assign direction.
    """
    if df.empty or "balance" not in df.columns:
        return df

    df = df.copy()
    balance_valid = df["balance"].notna()
    if balance_valid.sum() < 2:
        return df

    delta = df["balance"].diff()

    # Where credit is NaN but balance rose → it was a credit
    mask_cr = balance_valid & df["credit"].isna() & (delta > 0)
    df.loc[mask_cr, "credit"] = df.loc[mask_cr, "debit"]
    df.loc[mask_cr, "debit"] = np.nan

    # Where debit is NaN but balance fell → it was a debit
    mask_dr = balance_valid & df["debit"].isna() & (delta < 0)
    df.loc[mask_dr, "debit"] = df.loc[mask_dr, "credit"]
    df.loc[mask_dr, "credit"] = np.nan

    return df


# ── Summary computation ────────────────────────────────────────────────────────

def _compute_summary(df: pd.DataFrame) -> dict:
    """Return aggregated financial statistics from transaction DataFrame."""
    if df.empty:
        return _empty_summary()

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    monthly_inflow = df.groupby("month")["credit"].sum().fillna(0)
    monthly_outflow = df.groupby("month")["debit"].sum().fillna(0)

    avg_inflow = float(monthly_inflow.mean()) if not monthly_inflow.empty else 0.0
    avg_outflow = float(monthly_outflow.mean()) if not monthly_outflow.empty else 0.0
    avg_balance = float(df["balance"].mean()) if df["balance"].notna().any() else 0.0
    min_balance = float(df["balance"].min()) if df["balance"].notna().any() else 0.0
    max_balance = float(df["balance"].max()) if df["balance"].notna().any() else 0.0

    # Salary detection
    salary_rows = df[df["description"].str.contains(_SALARY_KEYWORDS, na=False)]
    salary_credits = salary_rows["credit"].dropna()
    monthly_salary = float(salary_credits.mean()) if not salary_credits.empty else 0.0
    salary_detected = not salary_credits.empty

    # Inflow consistency: coefficient of variation of monthly inflow
    inflow_cv = _coefficient_of_variation(monthly_inflow)

    return {
        "total_inflow": float(df["credit"].sum(skipna=True)),
        "total_outflow": float(df["debit"].sum(skipna=True)),
        "avg_monthly_inflow": avg_inflow,
        "avg_monthly_outflow": avg_outflow,
        "avg_balance": avg_balance,
        "min_balance": min_balance,
        "max_balance": max_balance,
        "n_transactions": len(df),
        "months_covered": int(df["month"].nunique()),
        "salary_detected": salary_detected,
        "estimated_monthly_salary": monthly_salary,
        "inflow_cv": inflow_cv,  # lower = more consistent
    }


def _empty_summary() -> dict:
    return {
        "total_inflow": 0.0, "total_outflow": 0.0,
        "avg_monthly_inflow": 0.0, "avg_monthly_outflow": 0.0,
        "avg_balance": 0.0, "min_balance": 0.0, "max_balance": 0.0,
        "n_transactions": 0, "months_covered": 0,
        "salary_detected": False, "estimated_monthly_salary": 0.0,
        "inflow_cv": 1.0,
    }


def _coefficient_of_variation(series: pd.Series) -> float:
    """CV = std / mean; returns 1.0 (max instability) if mean is zero."""
    mean = series.mean()
    if mean == 0 or len(series) < 2:
        return 1.0
    return float(series.std() / mean)
