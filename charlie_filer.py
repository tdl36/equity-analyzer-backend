#!/usr/bin/env python3
"""Charlie Downloads Filer -- routes downloaded research PDFs to STOCKS/<TICKER>/.

The last manual step in the research pipeline is "save this AlphaSense PDF into
the right iCloud folder". Everything after that is already automatic:
charlie_watcher.py uploads anything landing in STOCKS/<TICKER>/, and the backend
turns documents into a thesis. This module removes that step -- download a report
anywhere, and it gets filed by ticker.

WHY THIS IS CONSERVATIVE
------------------------------------------------------------------------------
Moving a file out of ~/Downloads without being asked is the kind of automation
that is infuriating when it guesses wrong: the file appears to vanish. So the
bar for acting is deliberately high.

A bare ticker match is NOT enough. "CVS receipt.pdf" contains a real ticker and
is obviously not research. Filing therefore requires either:

  * a ticker token in the filename AND a research signal (transcript, earnings,
    initiation, 10-K, Q3, FY26, price target, ...), or
  * an explicit exchange tag in the document text, e.g. "(NYSE: CVS)", which is
    close to conclusive on its own.

Anything weaker is logged and left alone. A file the analyst has to move by hand
costs seconds; a file silently moved to the wrong ticker corrupts a thesis.

The known-ticker universe comes from the STOCKS folders that already exist, so
it reflects actual coverage rather than a hardcoded list that would drift.
"""

import re
import shutil
from pathlib import Path

# Words that mark a PDF as sell-side / company research rather than, say, a
# receipt or a boarding pass that happens to contain three capital letters.
RESEARCH_HINTS = (
    "transcript", "earnings", "call", "initiation", "initiating", "coverage",
    "downgrade", "upgrade", "price target", "estimates", "outlook", "guidance",
    "10-k", "10q", "10-q", "8-k", "annual report", "investor day", "analyst day",
    "results", "quarterly", "preview", "recap", "note", "research", "equity",
    "alphasense", "conference", "prepared remarks", "q&a", "fy2", "management",
)

# "(NYSE: CVS)", "NASDAQ:AAPL", "NYSE - DE" — near-conclusive on its own.
EXCHANGE_RE = re.compile(
    r"\b(?:NYSE|NASDAQ|AMEX|NYSEARCA|OTC)\b\s*[:\-]?\s*([A-Z]{1,5})\b"
)

QUARTER_RE = re.compile(r"\b(?:[1-4]Q|Q[1-4])\s?(?:FY)?\s?\d{2,4}\b", re.I)


def known_tickers(stocks_dir: Path) -> set[str]:
    """Tickers Charlie already covers, taken from existing STOCKS folders."""
    if not stocks_dir or not stocks_dir.exists():
        return set()
    out = set()
    for child in stocks_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(".") or name in {"Processed", "Prior Versions"}:
            continue
        if 1 <= len(name) <= 5 and name.isalpha():
            out.add(name.upper())
    return out


def _tokens(filename: str) -> list[str]:
    """Uppercase alphabetic tokens from a filename, longest-lived order kept."""
    return [t.upper() for t in re.split(r"[^A-Za-z]+", filename) if t]


def _has_research_signal(text: str) -> bool:
    low = (text or "").lower()
    if any(h in low for h in RESEARCH_HINTS):
        return True
    return bool(QUARTER_RE.search(text or ""))


def detect_ticker(filename: str, universe: set[str], pdf_text: str = "") -> tuple[str | None, str, str]:
    """Decide where a file belongs.

    Returns (ticker, confidence, reason) where confidence is 'high' | 'low' |
    'none'. Only 'high' should ever cause a file to move.
    """
    universe = universe or set()
    name_tokens = _tokens(filename)

    # 1. Exchange tag in the document body — strongest signal available.
    for m in EXCHANGE_RE.finditer(pdf_text or ""):
        cand = m.group(1).upper()
        if not universe or cand in universe:
            return cand, "high", f"exchange tag '{m.group(0).strip()}' in document text"

    # 2. Ticker token in the filename, corroborated by a research signal.
    hit = next((t for t in name_tokens if t in universe), None)
    if hit:
        if _has_research_signal(filename) or _has_research_signal(pdf_text[:4000]):
            return hit, "high", f"'{hit}' in filename with a research signal"
        # A ticker with nothing else behind it: could be "CVS receipt.pdf".
        return hit, "low", f"'{hit}' in filename but no research signal — left in place"

    return None, "none", "no known ticker found in filename or document text"


def read_pdf_text(path: Path, max_pages: int = 2) -> str:
    """First pages of a PDF as text. Returns '' if nothing can read it."""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages[:max_pages])
    except Exception:
        pass
    try:
        import PyPDF2
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            return "\n".join((pg.extract_text() or "") for pg in reader.pages[:max_pages])
    except Exception:
        return ""


def target_path(stocks_dir: Path, ticker: str, filename: str) -> Path:
    """Destination for a filed document, never overwriting an existing file."""
    dest_dir = stocks_dir / ticker.upper()
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 2
    while (dest_dir / f"{stem} ({n}){suffix}").exists():
        n += 1
    return dest_dir / f"{stem} ({n}){suffix}"


def file_document(path: Path, stocks_dir: Path, *, dry_run: bool = False,
                  universe: set[str] | None = None):
    """Classify one downloaded file and move it if confidence is high.

    Returns (moved: bool, ticker, confidence, reason, destination|None).
    """
    if path.suffix.lower() != ".pdf" or path.name.startswith("."):
        return False, None, "none", "not a pdf", None

    # Never re-file something already inside the library.
    try:
        path.relative_to(stocks_dir)
        return False, None, "none", "already inside STOCKS", None
    except ValueError:
        pass

    uni = universe if universe is not None else known_tickers(stocks_dir)
    text = read_pdf_text(path)
    ticker, confidence, reason = detect_ticker(path.name, uni, text)

    if confidence != "high" or not ticker:
        return False, ticker, confidence, reason, None

    dest = target_path(stocks_dir, ticker, path.name)
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
    return True, ticker, confidence, reason, dest
