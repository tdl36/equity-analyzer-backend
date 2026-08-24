"""Tests for the Downloads filer.

The property that matters is restraint: this moves files out of the user's
Downloads folder, so a wrong guess makes a document appear to vanish. These
tests pin down when it acts and — more importantly — when it refuses to.
"""

from pathlib import Path

import charlie_filer as f


UNIVERSE = {"CVS", "DE", "NVDA", "CI", "AAPL"}


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_exchange_tag_in_body_is_enough_on_its_own():
    t, conf, why = f.detect_ticker("scan001.pdf", UNIVERSE, "CVS Health Corporation (NYSE: CVS)")
    assert (t, conf) == ("CVS", "high")
    assert "exchange tag" in why


def test_filename_ticker_plus_research_signal_files_it():
    t, conf, _ = f.detect_ticker("CVS_Q3_2026_earnings_transcript.pdf", UNIVERSE)
    assert (t, conf) == ("CVS", "high")


def test_bare_ticker_in_filename_is_not_enough():
    """'CVS receipt.pdf' contains a real ticker and is obviously not research."""
    t, conf, why = f.detect_ticker("CVS receipt.pdf", UNIVERSE)
    assert t == "CVS"
    assert conf == "low"
    assert "left in place" in why


def test_unknown_ticker_is_left_alone():
    t, conf, _ = f.detect_ticker("ZZZZ_earnings_transcript.pdf", UNIVERSE)
    assert t is None and conf == "none"


def test_unrelated_pdf_is_left_alone():
    t, conf, _ = f.detect_ticker("boarding pass.pdf", UNIVERSE, "Gate B12 Seat 14A")
    assert t is None and conf == "none"


def test_quarter_pattern_counts_as_a_research_signal():
    t, conf, _ = f.detect_ticker("DE 3Q26.pdf", UNIVERSE)
    assert (t, conf) == ("DE", "high")


def test_exchange_tag_outside_the_covered_universe_is_ignored():
    """Do not invent a folder for a name Charlie does not cover."""
    t, conf, _ = f.detect_ticker("note.pdf", UNIVERSE, "Foo Corp (NASDAQ: FOO)")
    assert t is None and conf == "none"


# --------------------------------------------------------------------------
# universe discovery
# --------------------------------------------------------------------------

def test_universe_comes_from_existing_folders(tmp_path):
    stocks = tmp_path / "STOCKS"
    for name in ("CVS", "DE", "Processed", ".DS_Store_dir", "TOOLONGNAME"):
        (stocks / name).mkdir(parents=True)
    (stocks / "loose.pdf").write_bytes(b"x")
    assert f.known_tickers(stocks) == {"CVS", "DE"}


def test_missing_stocks_dir_is_not_fatal(tmp_path):
    assert f.known_tickers(tmp_path / "nope") == set()


# --------------------------------------------------------------------------
# filing behaviour
# --------------------------------------------------------------------------

def _pdf(path: Path, name: str) -> Path:
    p = path / name
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def test_high_confidence_file_is_moved(tmp_path):
    stocks = tmp_path / "STOCKS"; (stocks / "CVS").mkdir(parents=True)
    src = _pdf(tmp_path, "CVS_Q3_earnings_transcript.pdf")
    moved, ticker, conf, _, dest = f.file_document(src, stocks, universe=UNIVERSE)
    assert moved and ticker == "CVS" and conf == "high"
    assert dest.exists() and not src.exists()


def test_low_confidence_file_is_never_moved(tmp_path):
    stocks = tmp_path / "STOCKS"; (stocks / "CVS").mkdir(parents=True)
    src = _pdf(tmp_path, "CVS receipt.pdf")
    moved, _, conf, _, dest = f.file_document(src, stocks, universe=UNIVERSE)
    assert moved is False and conf == "low" and dest is None
    assert src.exists(), "a low-confidence file must stay where the user left it"


def test_dry_run_moves_nothing(tmp_path):
    stocks = tmp_path / "STOCKS"; (stocks / "DE").mkdir(parents=True)
    src = _pdf(tmp_path, "DE_Q3_earnings_transcript.pdf")
    moved, _, _, _, dest = f.file_document(src, stocks, dry_run=True, universe=UNIVERSE)
    assert moved is True                    # it reports what it *would* do
    assert src.exists() and not dest.exists()


def test_existing_destination_is_never_overwritten(tmp_path):
    stocks = tmp_path / "STOCKS"; (stocks / "DE").mkdir(parents=True)
    existing = stocks / "DE" / "DE_Q3_earnings_transcript.pdf"
    existing.write_bytes(b"ORIGINAL")
    src = _pdf(tmp_path, "DE_Q3_earnings_transcript.pdf")

    moved, _, _, _, dest = f.file_document(src, stocks, universe=UNIVERSE)
    assert moved and dest != existing
    assert existing.read_bytes() == b"ORIGINAL"
    assert dest.exists()


def test_non_pdf_is_ignored(tmp_path):
    stocks = tmp_path / "STOCKS"; stocks.mkdir()
    src = tmp_path / "CVS_earnings_transcript.docx"; src.write_bytes(b"x")
    moved, _, _, why, _ = f.file_document(src, stocks, universe=UNIVERSE)
    assert moved is False and "not a pdf" in why


def test_file_already_inside_stocks_is_not_refiled(tmp_path):
    stocks = tmp_path / "STOCKS"; (stocks / "CVS").mkdir(parents=True)
    src = _pdf(stocks / "CVS", "CVS_Q3_earnings_transcript.pdf")
    moved, _, _, why, _ = f.file_document(src, stocks, universe=UNIVERSE)
    assert moved is False and "already inside STOCKS" in why


def test_target_path_suffixes_rather_than_clobbering(tmp_path):
    stocks = tmp_path / "STOCKS"; (stocks / "CI").mkdir(parents=True)
    (stocks / "CI" / "note.pdf").write_bytes(b"a")
    (stocks / "CI" / "note (2).pdf").write_bytes(b"b")
    assert f.target_path(stocks, "CI", "note.pdf").name == "note (3).pdf"
