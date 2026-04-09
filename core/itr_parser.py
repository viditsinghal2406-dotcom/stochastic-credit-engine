"""
ITR Parser
==========
Extracts income and tax fields from Indian Income Tax Return (ITR) PDF text.
Handles ITR-1 (Sahaj), ITR-2, and generic AIS/26AS layouts via keyword patterns.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from utils.cleaners import normalise_whitespace, parse_amount

logger = logging.getLogger(__name__)


# ── Field patterns ─────────────────────────────────────────────────────────────
# Each entry: (canonical_key, list_of_regex_patterns)
# Patterns are tried in order; first match wins.

_FIELD_PATTERNS: list[tuple[str, list[str]]] = [
    ("gross_total_income", [
        r"gross\s+total\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"total\s+gross\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"GTI[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("total_income", [
        r"total\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"net\s+taxable\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"taxable\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("salary_income", [
        r"income\s+from\s+salary[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"salary[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("business_income", [
        r"income\s+from\s+business[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"profit\s+(?:&|and)\s+gains[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"PGBP[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("other_income", [
        r"income\s+from\s+other\s+sources[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"other\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("exempt_income", [
        r"exempt\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"exempted\s+income[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("tax_payable", [
        r"tax\s+payable[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"tax\s+liability[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("tax_paid", [
        r"taxes\s+paid[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"tax\s+paid[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"TDS[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"advance\s+tax[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("refund_due", [
        r"refund\s+due[^\d]*([\d,]+(?:\.\d{1,2})?)",
        r"refund\s+amount[^\d]*([\d,]+(?:\.\d{1,2})?)",
    ]),
    ("assessment_year", [
        r"assessment\s+year\s*[:\-]?\s*(\d{4}[\-\/]\d{2,4})",
        r"A\.?Y\.?\s*(\d{4}[\-\/]\d{2,4})",
    ]),
    ("pan", [
        r"\b([A-Z]{5}\d{4}[A-Z])\b",
    ]),
]


# ── Public interface ───────────────────────────────────────────────────────────

def parse_itr(pages: list[str]) -> dict:
    """
    Extract key income & tax fields from ITR PDF pages.

    Parameters
    ----------
    pages : list[str]
        Raw page texts from :func:`utils.pdf_utils.load_pdf`.

    Returns
    -------
    dict
        Keys from *_FIELD_PATTERNS* plus ``_confidence`` (0–1 float),
        ``_raw_text_length`` (int), and ``_warnings`` (list[str]).
    """
    full_text = "\n".join(pages)
    full_text = normalise_whitespace(full_text)

    result: dict = {}
    warnings: list[str] = []

    for key, patterns in _FIELD_PATTERNS:
        value = _extract_field(full_text, patterns)
        result[key] = value
        if value is None:
            warnings.append(f"Could not extract: {key}")

    result["_confidence"] = _compute_confidence(result)
    result["_raw_text_length"] = len(full_text)
    result["_warnings"] = warnings

    # Derived convenience field
    result["annual_income"] = _resolve_annual_income(result)

    logger.debug(
        "ITR parse: confidence=%.2f, annual_income=%s",
        result["_confidence"],
        result.get("annual_income"),
    )
    return result


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_field(text: str, patterns: list[str]) -> Optional[float | str]:
    """Try each pattern in order; return the first successful parse."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            # Return as string for non-numeric fields (year, PAN)
            if re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", raw):
                return parse_amount(raw)
            return raw
    return None


def _resolve_annual_income(result: dict) -> Optional[float]:
    """
    Choose the most authoritative income figure available.
    Priority: total_income > gross_total_income > salary_income + other_income.
    """
    if result.get("total_income"):
        return result["total_income"]
    if result.get("gross_total_income"):
        return result["gross_total_income"]
    components = [
        result.get("salary_income") or 0.0,
        result.get("business_income") or 0.0,
        result.get("other_income") or 0.0,
    ]
    total = sum(components)
    return total if total > 0 else None


def _compute_confidence(result: dict) -> float:
    """
    Fraction of non-None fields among the primary fields.
    assessment_year and pan are bonus, not counted in denominator.
    """
    primary = [
        "gross_total_income", "total_income", "salary_income",
        "tax_paid", "tax_payable",
    ]
    found = sum(1 for k in primary if result.get(k) is not None)
    return round(found / len(primary), 2)
