"""SEC EDGAR scraper — ticker -> CIK + recent 8-K fetcher with Exhibit 99.1 / 99.2
auto-download. Used as the primary data source for auto-fetching earnings
press release + earnings presentation on earnings day.

SEC requires a User-Agent header identifying the requester. No API key needed.
Rate limit: 10 requests/sec (EDGAR fair-use).

Exposed helpers:
  ticker_to_cik(ticker)              -> str | None (10-digit zero-padded)
  recent_8k_filings(cik, days=7)     -> list of {accessionNumber, filingDate, primaryDoc, primaryDocDescription}
  download_earnings_exhibits(ticker, dest_dir, since_days=3) -> {downloaded: [{name, path}], notes: [str]}

A filing is treated as an earnings release when Item 2.02 appears in the
primary doc's description or the filing includes an Exhibit 99.x named
"Press Release" / "Earnings" / "Q\\d YYYY".
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

USER_AGENT = "Charlie Research charlie@tonydlee.com"

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

# Module-level cache of ticker->CIK, refreshed once per process start.
_TICKER_CACHE: dict[str, str] | None = None
_TICKER_CACHE_AT: float = 0.0
_TICKER_CACHE_TTL_SEC: int = 60 * 60 * 12  # 12h


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _load_ticker_map() -> dict[str, str]:
    """Fetch the SEC ticker file and return {TICKER: CIK10} mapping."""
    raw = _http_get(SEC_TICKER_MAP_URL)
    data = json.loads(raw.decode("utf-8"))
    out: dict[str, str] = {}
    # Response shape: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
    for _, row in data.items():
        t = (row.get("ticker") or "").upper()
        if not t:
            continue
        out[t] = f"{int(row['cik_str']):010d}"
    return out


def ticker_to_cik(ticker: str) -> str | None:
    """Return the 10-digit zero-padded CIK for a ticker, or None. Cached 12h."""
    global _TICKER_CACHE, _TICKER_CACHE_AT
    now = time.time()
    if _TICKER_CACHE is None or (now - _TICKER_CACHE_AT) > _TICKER_CACHE_TTL_SEC:
        try:
            _TICKER_CACHE = _load_ticker_map()
            _TICKER_CACHE_AT = now
        except Exception as e:
            print(f"sec_edgar: ticker map load failed: {e}")
            return None
    return _TICKER_CACHE.get(ticker.upper()) if _TICKER_CACHE else None


def recent_8k_filings(cik: str, days: int = 7) -> list[dict]:
    """Return 8-K filings from the last N days for this CIK, newest first."""
    url = f"{SEC_SUBMISSIONS_BASE}/CIK{cik}.json"
    try:
        raw = _http_get(url)
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"sec_edgar: submissions fetch failed for CIK {cik}: {e}")
        return []
    recent = data.get("filings", {}).get("recent", {}) or {}
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    primary_docs = recent.get("primaryDocument") or []
    primary_desc = recent.get("primaryDocDescription") or []
    items_list = recent.get("items") or []

    cutoff = datetime.utcnow().date() - timedelta(days=days)
    out = []
    for i in range(min(len(forms), len(accessions), len(filing_dates))):
        if forms[i] != "8-K":
            continue
        try:
            fdate = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
        except Exception:
            continue
        if fdate < cutoff:
            continue
        out.append({
            "cik": cik,
            "accessionNumber": accessions[i],
            "filingDate": fdate.isoformat(),
            "primaryDoc": primary_docs[i] if i < len(primary_docs) else None,
            "primaryDocDescription": primary_desc[i] if i < len(primary_desc) else "",
            "items": items_list[i] if i < len(items_list) else "",
        })
    # newest first
    out.sort(key=lambda r: r["filingDate"], reverse=True)
    return out


def _filing_archive_dir(cik: str, accession_no: str) -> str:
    acc_no_nodash = accession_no.replace("-", "")
    return f"{SEC_ARCHIVE_BASE}/{int(cik)}/{acc_no_nodash}"


def list_filing_exhibits(cik: str, accession_no: str) -> list[dict]:
    """Return [{name, url, type}] for documents in the filing. Uses the
    filing-index JSON which SEC exposes at /{archive}/{accno}/index.json."""
    base = _filing_archive_dir(cik, accession_no)
    try:
        raw = _http_get(f"{base}/index.json")
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"sec_edgar: index load failed for {accession_no}: {e}")
        return []
    items = (data.get("directory") or {}).get("item") or []
    out = []
    for it in items:
        name = it.get("name") or ""
        typ = (it.get("type") or "").upper()
        if not name or name.endswith("-index.htm") or name.endswith("-index.html"):
            continue
        out.append({
            "name": name,
            "type": typ,
            "url": f"{base}/{name}",
        })
    return out


def _is_earnings_release(filing: dict) -> bool:
    """Heuristic: Item 2.02 in the items string OR primary doc description
    mentions 'Earnings' / 'Results'."""
    items = (filing.get("items") or "").lower()
    desc = (filing.get("primaryDocDescription") or "").lower()
    if "2.02" in items:
        return True
    if any(kw in desc for kw in ("earnings", "results", "quarterly")):
        return True
    return False


def _exhibit_keep(name: str) -> bool:
    """Keep Exhibit 99.1 (PR) and 99.2 (presentation) by filename heuristics.
    SEC names them variably: ex991.htm, ex99_1.htm, a8kex991.htm, etc."""
    lc = name.lower()
    if lc.endswith((".htm", ".html", ".pdf")):
        if "ex99" in lc or "ex-99" in lc:
            return True
    return False


def download_earnings_exhibits(ticker: str, dest_dir: Path, since_days: int = 3) -> dict:
    """Find the most recent earnings 8-K for this ticker and download its
    99.x exhibits into dest_dir. Returns {downloaded, notes, accessionNumber}.

    Does NOT overwrite existing files (idempotent per filename)."""
    notes: list[str] = []
    downloaded: list[dict] = []
    cik = ticker_to_cik(ticker)
    if not cik:
        return {"downloaded": [], "notes": [f"no CIK for {ticker}"], "accessionNumber": None}

    filings = recent_8k_filings(cik, days=since_days)
    if not filings:
        return {"downloaded": [], "notes": ["no recent 8-K"], "accessionNumber": None}

    target = None
    for f in filings:
        if _is_earnings_release(f):
            target = f
            break
    if not target:
        return {"downloaded": [], "notes": ["no earnings 8-K in window"], "accessionNumber": None}

    acc = target["accessionNumber"]
    exhibits = list_filing_exhibits(cik, acc)
    kept = [ex for ex in exhibits if _exhibit_keep(ex["name"])]
    if not kept:
        notes.append(f"no 99.x exhibits in filing {acc}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for ex in kept:
        name = ex["name"]
        out_path = dest_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            notes.append(f"{name}: already present ({out_path.stat().st_size:,} bytes)")
            downloaded.append({"name": name, "path": str(out_path), "alreadyPresent": True})
            continue
        try:
            data = _http_get(ex["url"], timeout=60)
            out_path.write_bytes(data)
            downloaded.append({"name": name, "path": str(out_path), "alreadyPresent": False})
            notes.append(f"{name}: downloaded {len(data):,} bytes")
            time.sleep(0.15)  # polite pacing to stay under 10 rps
        except Exception as e:
            notes.append(f"{name}: download failed — {e}")

    return {
        "downloaded": downloaded,
        "notes": notes,
        "accessionNumber": acc,
        "filingDate": target.get("filingDate"),
        "items": target.get("items"),
    }
