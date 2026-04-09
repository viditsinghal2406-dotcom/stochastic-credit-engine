"""
PDF Utilities
=============
Handles PDF loading, decryption, and raw text extraction.
Supports both pikepdf (preferred) and PyPDF2 as fallback.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Attempt imports ────────────────────────────────────────────────────────────

try:
    import pikepdf
    _PIKEPDF = True
except ImportError:
    _PIKEPDF = False

try:
    import PyPDF2
    _PYPDF2 = True
except ImportError:
    _PYPDF2 = False

if not _PIKEPDF and not _PYPDF2:
    raise ImportError(
        "No PDF library found. Install either 'pikepdf' or 'PyPDF2':\n"
        "  pip install pikepdf   (recommended)\n"
        "  pip install PyPDF2    (fallback)"
    )


# ── Exceptions ─────────────────────────────────────────────────────────────────

class PDFPasswordRequired(Exception):
    """Raised when an encrypted PDF is opened without a password."""


class PDFPasswordIncorrect(Exception):
    """Raised when the supplied password does not decrypt the PDF."""


class PDFParseError(Exception):
    """Raised for any other PDF read failure."""


# ── Core function ──────────────────────────────────────────────────────────────

def load_pdf(file: bytes | str, password: Optional[str] = None) -> list[str]:
    """
    Load a PDF file and return a list of page texts.

    Parameters
    ----------
    file : bytes | str
        Raw bytes of the PDF, or a filesystem path string.
    password : str, optional
        Plaintext password for encrypted PDFs.

    Returns
    -------
    list[str]
        One string per page; empty string if a page has no extractable text.

    Raises
    ------
    PDFPasswordRequired  – PDF is encrypted and no password was supplied.
    PDFPasswordIncorrect – Supplied password did not open the PDF.
    PDFParseError        – Any other unrecoverable read error.
    """
    if isinstance(file, str):
        with open(file, "rb") as fh:
            file = fh.read()

    if _PIKEPDF:
        return _load_with_pikepdf(file, password)
    return _load_with_pypdf2(file, password)


# ── pikepdf implementation ─────────────────────────────────────────────────────

def _load_with_pikepdf(data: bytes, password: Optional[str]) -> list[str]:
    """Extract text via pikepdf + pdfminer."""
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
    except ImportError:
        raise ImportError("pdfminer.six is required for text extraction: pip install pdfminer.six")

    buf = io.BytesIO(data)
    try:
        if password:
            pdf = pikepdf.open(buf, password=password)
        else:
            pdf = pikepdf.open(buf)
    except pikepdf.PasswordError:
        if password is None:
            raise PDFPasswordRequired("PDF is encrypted. Provide a password.")
        raise PDFPasswordIncorrect("Incorrect password for the PDF.")
    except Exception as exc:
        raise PDFParseError(f"pikepdf could not open PDF: {exc}") from exc

    # Re-save to a clean buffer so pdfminer can read it
    clean_buf = io.BytesIO()
    pdf.save(clean_buf)
    pdf.close()
    clean_buf.seek(0)

    pages: list[str] = []
    for page_num in range(len(pikepdf.open(io.BytesIO(data if password is None else clean_buf.read())).pages)):
        out = io.StringIO()
        clean_buf.seek(0)
        extract_text_to_fp(
            clean_buf,
            out,
            page_numbers=[page_num],
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        pages.append(out.getvalue())

    return pages


# ── PyPDF2 implementation ──────────────────────────────────────────────────────

def _load_with_pypdf2(data: bytes, password: Optional[str]) -> list[str]:
    """Extract text via PyPDF2."""
    buf = io.BytesIO(data)
    try:
        reader = PyPDF2.PdfReader(buf)
    except Exception as exc:
        raise PDFParseError(f"PyPDF2 could not open PDF: {exc}") from exc

    if reader.is_encrypted:
        if password is None:
            raise PDFPasswordRequired("PDF is encrypted. Provide a password.")
        result = reader.decrypt(password)
        if result == 0:
            raise PDFPasswordIncorrect("Incorrect password for the PDF.")

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    return pages


# ── Helper ─────────────────────────────────────────────────────────────────────

def pages_to_text(pages: list[str]) -> str:
    """Concatenate all page texts into a single string."""
    return "\n".join(pages)
