#!/usr/bin/env python3
"""
batch_charlie_update.py — Batch-process stock thesis updates via Charlie API.

Scans iCloud STOCKS folders for new documents, uploads them to Charlie,
and triggers parallel background analysis jobs.

Usage:
    python3 batch_charlie_update.py                  # process all tickers with new files
    python3 batch_charlie_update.py ABT BSX ABBV     # process specific tickers
    python3 batch_charlie_update.py --dry-run        # show what would be processed
    python3 batch_charlie_update.py --reset ABT      # clear manifest for ABT and reprocess
    python3 batch_charlie_update.py --status         # check status of running jobs
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import zipfile
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STOCKS_ROOT = Path(os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/STOCKS"
))
API_URL = "https://equity-analyzer-backend.onrender.com"
MANIFEST_NAME = ".charlie_uploaded.json"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
POLL_INTERVAL = 5  # seconds between status polls

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_path(ticker_dir: Path) -> Path:
    return ticker_dir / MANIFEST_NAME


def load_manifest(ticker_dir: Path) -> dict:
    """Load {filename: upload_timestamp} manifest."""
    mp = _manifest_path(ticker_dir)
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(ticker_dir: Path, manifest: dict):
    mp = _manifest_path(ticker_dir)
    mp.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def _is_deliverable(filename: str) -> bool:
    """Check if a file is a deliverable (output) rather than a source doc."""
    fn = filename.lower()
    # Deliverables: notes, sources docs, changelogs, charts, manifest
    if fn.startswith("."):
        return True
    for pattern in ["_note_", "_sources_", "_changelog", "_revenue_", "_profit_",
                     "_breakdown", "_chart", "_infographic"]:
        if pattern in fn:
            return True
    if fn.endswith(".docx") or fn.endswith(".md"):
        return True
    return False


def scan_ticker_files(ticker_dir: Path) -> list[Path]:
    """Scan main folder and Processed/ subfolder for source documents.

    Deduplicates by filename — if the same file exists in both main and
    Processed/, prefer the main folder copy.
    """
    seen_names: dict[str, Path] = {}
    scan_dirs = [ticker_dir]

    processed_dir = ticker_dir / "Processed"
    if processed_dir.exists():
        scan_dirs.append(processed_dir)

    for d in scan_dirs:
        for f in d.iterdir():
            if f.is_file() and not _is_deliverable(f.name):
                ext = f.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    # First seen wins (main folder before Processed/)
                    if f.name not in seen_names:
                        seen_names[f.name] = f
                elif ext == ".zip":
                    for extracted in _extract_zip(f):
                        if extracted.name not in seen_names:
                            seen_names[extracted.name] = extracted

    return list(seen_names.values())


def _extract_zip(zip_path: Path) -> list[Path]:
    """Extract supported files from a zip into a temp directory, return paths."""
    extracted = []
    try:
        extract_dir = Path(tempfile.mkdtemp(prefix="charlie_zip_"))
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.startswith("__MACOSX") or member.startswith("."):
                    continue
                ext = Path(member).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    zf.extract(member, extract_dir)
                    extracted.append(extract_dir / member)
    except Exception as e:
        print(f"  Warning: Could not extract {zip_path.name}: {e}")
    return extracted


def find_new_files(ticker_dir: Path, manifest: dict) -> list[Path]:
    """Find files not yet in the manifest."""
    all_files = scan_ticker_files(ticker_dir)
    new_files = []
    for f in all_files:
        # Use filename as the manifest key (not full path) to handle moves between main/Processed
        if f.name not in manifest:
            new_files.append(f)
    return new_files


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    return key


def encode_file(path: Path) -> dict:
    """Read and base64-encode a file for the Charlie API."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    ext = path.suffix.lower()
    if ext == ".pdf":
        file_type, mime_type = "pdf", "application/pdf"
    elif ext in (".png",):
        file_type, mime_type = "image", "image/png"
    elif ext in (".jpg", ".jpeg"):
        file_type, mime_type = "image", "image/jpeg"
    else:
        file_type, mime_type = "document", "application/octet-stream"

    return {
        "filename": path.name,
        "fileData": b64,
        "fileType": file_type,
        "mimeType": mime_type,
        "metadata": {},
    }


def get_existing_analysis(ticker: str) -> dict | None:
    """Fetch existing analysis for a ticker from Charlie API."""
    import requests
    try:
        resp = requests.get(f"{API_URL}/api/analyses", timeout=15)
        if resp.ok:
            data = resp.json()
            analyses = data if isinstance(data, list) else data.get("analyses", [])
            for a in analyses:
                if a.get("ticker", "").upper() == ticker.upper():
                    analysis = a.get("analysis", {})
                    if isinstance(analysis, str):
                        analysis = json.loads(analysis)
                    return analysis
    except Exception as e:
        print(f"  Warning: Could not fetch existing analysis for {ticker}: {e}")
    return None


def submit_batch_jobs(ticker_jobs: dict[str, dict], api_key: str, existing_weight: int = 60) -> dict[str, str]:
    """Submit batch analysis jobs. Returns {ticker: job_id}."""
    import requests

    new_weight = 100 - existing_weight
    jobs = []
    for ticker, job_data in ticker_jobs.items():
        has_existing = bool(job_data.get("existingAnalysis"))
        job = {
            "ticker": ticker,
            "newDocuments": job_data["documents"],
            "documentFilenames": [d["filename"] for d in job_data["documents"]],
            "documentDetails": [
                {"filename": d["filename"], "weight": 1, "isNew": True}
                for d in job_data["documents"]
            ],
            "weightingConfig": {
                "mode": "simple",
                "existingAnalysisWeight": existing_weight,
                "newDocsWeight": new_weight,
            } if has_existing else {"mode": "simple"},
        }
        if has_existing:
            job["existingAnalysis"] = job_data["existingAnalysis"]

        jobs.append(job)

    resp = requests.post(
        f"{API_URL}/api/analysis-jobs-batch",
        json={"apiKey": api_key, "jobs": jobs},
        timeout=60,
    )

    if not resp.ok:
        print(f"Error submitting batch jobs: {resp.status_code} {resp.text}")
        sys.exit(1)

    return resp.json().get("jobIds", {})


def poll_jobs(job_ids: dict[str, str]) -> dict[str, dict]:
    """Poll until all jobs complete or fail. Returns {ticker: result_or_error}."""
    import requests

    id_list = list(job_ids.values())
    ticker_by_id = {v: k for k, v in job_ids.items()}
    results = {}
    start = time.time()

    while id_list:
        time.sleep(POLL_INTERVAL)
        elapsed = int(time.time() - start)

        try:
            resp = requests.post(
                f"{API_URL}/api/analysis-jobs-batch-status",
                json={"jobIds": id_list},
                timeout=30,
            )
            if not resp.ok:
                print(f"  Poll error: {resp.status_code}")
                continue

            jobs = resp.json().get("jobs", {})
        except Exception as e:
            print(f"  Poll error: {e}")
            continue

        still_running = []
        status_parts = []
        for jid in id_list:
            job = jobs.get(jid, {})
            ticker = ticker_by_id.get(jid, "???")
            status = job.get("status", "unknown")

            if status == "complete":
                results[ticker] = {"success": True, "result": job.get("result", {})}
                status_parts.append(f"{ticker}: done")
            elif status == "failed":
                results[ticker] = {"success": False, "error": job.get("error", "Unknown")}
                status_parts.append(f"{ticker}: FAILED")
            else:
                still_running.append(jid)
                progress = job.get("progress", status)
                status_parts.append(f"{ticker}: {progress}")

        print(f"  [{elapsed}s] {' | '.join(status_parts)}")
        id_list = still_running

    return results


def save_analysis_result(ticker: str, job_result: dict, existing_analysis: dict | None, new_filenames: list[str]):
    """Save completed analysis result to Charlie DB, preserving version history.

    Replicates the frontend merge logic:
    1. Snapshot the old thesis/signposts/threats into history[]
    2. Merge documentHistory (old + new metadata from LLM)
    3. Save the combined analysis
    """
    import requests
    from datetime import datetime

    new_analysis = job_result.get("analysis", {})
    doc_metadata = job_result.get("documentMetadata", [])

    # --- 1. Preserve version history ---
    existing_history = []
    if existing_analysis:
        existing_history = list(existing_analysis.get("history", []))
        # Snapshot the old version before overwriting
        if existing_analysis.get("thesis") or existing_analysis.get("signposts") or existing_analysis.get("threats"):
            existing_history.append({
                "timestamp": existing_analysis.get("updatedAt", datetime.utcnow().isoformat()),
                "thesis": existing_analysis.get("thesis"),
                "signposts": existing_analysis.get("signposts"),
                "threats": existing_analysis.get("threats"),
            })
            # Keep only last 20 versions
            existing_history = existing_history[-20:]

    new_analysis["history"] = existing_history

    # --- 2. Merge documentHistory ---
    existing_docs = list((existing_analysis or {}).get("documentHistory", []))

    # Update existing docs with any newly extracted metadata
    for ed in existing_docs:
        meta = next((m for m in doc_metadata if m.get("filename") == ed.get("filename")), None)
        if meta:
            for field in ("docType", "source", "publishDate", "title", "quarter", "filingType"):
                if meta.get(field):
                    ed[field] = meta[field]
            if meta.get("authors"):
                ed["authors"] = meta["authors"]

    # Add new docs not already in history
    now = datetime.utcnow().isoformat()
    for fname in new_filenames:
        if any(d.get("filename") == fname for d in existing_docs):
            continue
        meta = next((m for m in doc_metadata if m.get("filename") == fname), {})
        existing_docs.append({
            "filename": fname,
            "type": "broker_report",
            "date": now[:10],
            "weight": 1,
            "stored": True,
            "addedAt": now,
            "docType": meta.get("docType"),
            "source": meta.get("source"),
            "publishDate": meta.get("publishDate"),
            "authors": meta.get("authors", []),
            "title": meta.get("title"),
        })

    new_analysis["documentHistory"] = existing_docs

    # --- 3. Save ---
    resp = requests.post(
        f"{API_URL}/api/save-analysis",
        json={
            "ticker": new_analysis.get("ticker", ticker),
            "companyName": new_analysis.get("company", ticker),
            "analysis": new_analysis,
        },
        timeout=30,
    )
    if resp.ok:
        print(f"  {ticker}: Saved to Charlie DB (history preserved: {len(existing_history)} version(s))")
    else:
        print(f"  {ticker}: Failed to save — {resp.status_code} {resp.text[:200]}")


# ---------------------------------------------------------------------------
# Status check (for previously submitted jobs)
# ---------------------------------------------------------------------------

def check_running_jobs():
    """Check if there are running jobs from a previous invocation."""
    status_file = STOCKS_ROOT / "_CONFIG" / ".charlie_batch_jobs.json"
    if not status_file.exists():
        print("No running batch jobs found.")
        return

    job_ids = json.loads(status_file.read_text())
    print(f"Checking {len(job_ids)} job(s)...")

    results = poll_jobs(job_ids)
    for ticker, result in results.items():
        if result["success"]:
            # Fetch current analysis to preserve history (best-effort)
            existing = get_existing_analysis(ticker)
            save_analysis_result(ticker, result["result"], existing, [])
            print(f"  {ticker}: Complete and saved")
        else:
            print(f"  {ticker}: Failed — {result.get('error', 'Unknown')}")

    status_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_all_tickers() -> list[str]:
    """Get all ticker folders from STOCKS_ROOT."""
    tickers = []
    for d in sorted(STOCKS_ROOT.iterdir()):
        if d.is_dir() and d.name.isalpha() and d.name.isupper() and d.name != "_CONFIG":
            tickers.append(d.name)
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Batch-update Charlie thesis analyses")
    parser.add_argument("tickers", nargs="*", help="Specific tickers to process (default: all with new files)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without doing it")
    parser.add_argument("--reset", nargs="*", metavar="TICKER", help="Clear manifest for specified tickers (or all if none given)")
    parser.add_argument("--weight", type=int, default=60, metavar="N",
                        help="Existing analysis weight %% (default: 60, new docs get 100-N)")
    parser.add_argument("--status", action="store_true", help="Check status of running jobs")
    args = parser.parse_args()

    if args.status:
        check_running_jobs()
        return

    # Handle --reset
    if args.reset is not None:
        reset_tickers = [t.upper() for t in args.reset] if args.reset else get_all_tickers()
        for ticker in reset_tickers:
            ticker_dir = STOCKS_ROOT / ticker
            mp = _manifest_path(ticker_dir)
            if mp.exists():
                mp.unlink()
                print(f"  {ticker}: manifest cleared")
            else:
                print(f"  {ticker}: no manifest to clear")
        return

    # Determine tickers to scan
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = get_all_tickers()

    print(f"Scanning {len(tickers)} ticker(s) for new documents...\n")

    # Phase 1: Scan and identify new files
    ticker_work = {}
    for ticker in tickers:
        ticker_dir = STOCKS_ROOT / ticker
        if not ticker_dir.exists():
            continue

        manifest = load_manifest(ticker_dir)
        new_files = find_new_files(ticker_dir, manifest)

        if not new_files:
            continue

        total_size = sum(f.stat().st_size for f in new_files)
        size_mb = total_size / (1024 * 1024)
        print(f"  {ticker}: {len(new_files)} new file(s) ({size_mb:.1f} MB)")
        for f in new_files:
            print(f"    - {f.name} ({f.stat().st_size / 1024:.0f} KB)")

        ticker_work[ticker] = {
            "dir": ticker_dir,
            "files": new_files,
            "manifest": manifest,
        }

    if not ticker_work:
        print("No tickers with new documents found.")
        return

    print(f"\n{'='*50}")
    print(f"Ready to process {len(ticker_work)} ticker(s)")
    print(f"Weighting: {args.weight}% existing / {100 - args.weight}% new docs")
    print(f"{'='*50}\n")

    if args.dry_run:
        print("(Dry run — no changes made)")
        return

    # Phase 2: Encode files and build job payloads
    api_key = get_api_key()
    ticker_jobs = {}

    for ticker, work in ticker_work.items():
        print(f"  Encoding {ticker}...")
        documents = []
        for f in work["files"]:
            try:
                doc = encode_file(f)
                documents.append(doc)
            except Exception as e:
                print(f"    Warning: Could not encode {f.name}: {e}")

        if not documents:
            print(f"    {ticker}: No valid documents to upload, skipping")
            continue

        existing = get_existing_analysis(ticker)
        ticker_jobs[ticker] = {
            "documents": documents,
            "existingAnalysis": existing,
        }
        if existing:
            print(f"    {ticker}: {len(documents)} doc(s) + existing analysis (update mode)")
        else:
            print(f"    {ticker}: {len(documents)} doc(s) (new analysis)")

    if not ticker_jobs:
        print("No valid jobs to submit.")
        return

    # Phase 3: Submit batch jobs
    print(f"\nSubmitting {len(ticker_jobs)} analysis job(s) to Charlie...")
    job_ids = submit_batch_jobs(ticker_jobs, api_key, existing_weight=args.weight)
    print(f"Jobs started: {json.dumps(job_ids, indent=2)}\n")

    # Save job IDs for --status recovery
    config_dir = STOCKS_ROOT / "_CONFIG"
    config_dir.mkdir(exist_ok=True)
    (config_dir / ".charlie_batch_jobs.json").write_text(json.dumps(job_ids))

    # Phase 4: Poll for completion
    print("Polling for results...")
    results = poll_jobs(job_ids)

    # Phase 5: Save results and update manifests
    print(f"\n{'='*50}")
    print("Results:")
    print(f"{'='*50}")

    succeeded = 0
    failed = 0
    for ticker, result in results.items():
        work = ticker_work[ticker]
        if result["success"]:
            existing = ticker_jobs[ticker].get("existingAnalysis")
            new_fnames = [d["filename"] for d in ticker_jobs[ticker]["documents"]]
            save_analysis_result(ticker, result["result"], existing, new_fnames)

            # Update manifest
            manifest = work["manifest"]
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            for f in work["files"]:
                manifest[f.name] = ts
            save_manifest(work["dir"], manifest)
            succeeded += 1
            print(f"  {ticker}: OK — analysis saved, manifest updated")
        else:
            failed += 1
            print(f"  {ticker}: FAILED — {result.get('error', 'Unknown error')}")

    # Clean up job tracker
    (config_dir / ".charlie_batch_jobs.json").unlink(missing_ok=True)

    print(f"\nDone: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
