"""
Text Cleaners
=============
Regex-based utilities for normalising raw PDF text before parsing.
"""

from __future__ import annotations

import re
from typing import Optional


# ── Amount parsing ─────────────────────────────────────────────────────────────

_AMOUNT_RE = re.compile(r"[\d,]+(?:\.\d{1,2})?")


def parse_amount(text: str) -> Optional[float]:
    """
    Extract the first valid currency amount from a string.

    Examples
    --------
    >>> parse_amount("1,23,456.78")
    123456.78
    >>> parse_amount("Dr 5,000.00")
    5000.0
    >>> parse_amount("n/a")
    None
    """
    text = text.replace(" ", "")
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


# ── Date parsing ───────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    r"\b(\d{2})[/\-](\d{2})[/\-](\d{4})\b",   # DD/MM/YYYY or DD-MM-YYYY
    r"\b(\d{2})[/\-](\d{2})[/\-](\d{2})\b",    # DD/MM/YY
    r"\b(\d{4})[/\-](\d{2})[/\-](\d{2})\b",    # YYYY-MM-DD
    r"\b(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\b",# D Mon YYYY
]

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_date(text: str) -> Optional[str]:
    """
    Return an ISO-format date string (YYYY-MM-DD) from arbitrary date text.
    Returns None if no recognisable date is found.
    """
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        groups = m.groups()
        try:
            if len(groups) == 3:
                g0, g1, g2 = groups
                # YYYY-MM-DD
                if len(g0) == 4:
                    return f"{g0}-{int(g1):02d}-{int(g2):02d}"
                # D Mon YYYY
                if not g1.isdigit():
                    month = _MONTH_MAP.get(g1.lower())
                    if month:
                        year = int(g2)
                        if year < 100:
                            year += 2000
                        return f"{year}-{month:02d}-{int(g0):02d}"
                # DD/MM/YY or DD/MM/YYYY
                day, month, year = int(g0), int(g1), int(g2)
                if year < 100:
                    year += 2000
                return f"{year}-{month:02d}-{day:02d}"
        except (ValueError, AttributeError):
            continue
    return None


# ── String normalisers ─────────────────────────────────────────────────────────

def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_chars(text: str, keep: str = r"a-zA-Z0-9 ./@\-") -> str:
    """Strip characters not in *keep* character class."""
    return re.sub(fr"[^{keep}]", " ", text)


def clean_line(line: str) -> str:
    """Normalise a single text line for downstream parsing."""
    line = normalise_whitespace(line)
    line = re.sub(r"\s{2,}", "  ", line)  # keep double-space as column separator
    return line


# ── Table row splitter ─────────────────────────────────────────────────────────

def split_columns(line: str, min_gap: int = 2) -> list[str]:
    """
    Split a PDF table row into columns using whitespace gaps of ≥ *min_gap* spaces.

    Parameters
    ----------
    line : str
        A single normalised text line.
    min_gap : int
        Minimum consecutive spaces to treat as a column boundary.
    """
    pattern = fr" {{{min_gap},}}"
    parts = re.split(pattern, line.strip())
    return [p.strip() for p in parts if p.strip()]


# ── Debit / Credit direction markers ──────────────────────────────────────────

_DR_RE = re.compile(r"\b(dr|dbt|debit|withdrawal|w/d)\b", re.IGNORECASE)
_CR_RE = re.compile(r"\b(cr|crd|credit|deposit)\b", re.IGNORECASE)


def detect_dr_cr(text: str) -> Optional[str]:
    """
    Return 'DR', 'CR', or None based on keywords in the text.
    """
    if _DR_RE.search(text):
        return "DR"
    if _CR_RE.search(text):
        return "CR"
    return None
