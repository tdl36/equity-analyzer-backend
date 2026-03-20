#!/usr/bin/env python3
"""Charlie iCloud Watcher -- auto-uploads new PDFs to the Charlie backend.

Monitors iCloud STOCKS folders for new PDF files and uploads them via the
/api/documents/save endpoint. Maintains persistent state to avoid duplicate
uploads across restarts.

Usage:
    python3 charlie_watcher.py                    # Watch mode only
    python3 charlie_watcher.py --initial-scan     # Scan existing + watch
    python3 charlie_watcher.py --dry-run          # Show what would upload
    python3 charlie_watcher.py --initial-scan --dry-run  # Preview initial scan
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

# Third-party -- install with: pip install watchdog requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WATCH_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/STOCKS"
API_URL = "https://equity-analyzer-backend.onrender.com"
UPLOAD_ENDPOINT = f"{API_URL}/api/documents/save"
STATE_FILE = Path.home() / ".charlie_watcher_state.json"

IGNORE_DIRS: set[str] = {"Processed", "Prior Versions", ".icloud"}
IGNORE_PREFIXES: tuple[str, ...] = ("~$", ".")
ALLOWED_EXTENSIONS: set[str] = {".pdf"}

# iCloud sometimes writes files in stages; wait briefly before reading.
SETTLE_DELAY_SECONDS = 2.0
# Max file size to upload (50 MB).
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
# HTTP timeout for uploads (Render cold-starts can be slow).
UPLOAD_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("charlie_watcher")

# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


class UploadState:
    """Track which files have already been uploaded, persisted to JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._uploaded: dict[str, str] = {}  # absolute path -> ISO timestamp
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._uploaded = data.get("uploaded", {})
                log.info("Loaded state: %d previously uploaded files", len(self._uploaded))
            except (json.JSONDecodeError, KeyError):
                log.warning("Corrupt state file; starting fresh")
                self._uploaded = {}
        else:
            log.info("No state file found; starting fresh")

    def save(self) -> None:
        payload = {
            "uploaded": self._uploaded,
            "last_saved": datetime.now(timezone.utc).isoformat(),
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_uploaded(self, filepath: str) -> bool:
        return filepath in self._uploaded

    def mark_uploaded(self, filepath: str) -> None:
        self._uploaded[filepath] = datetime.now(timezone.utc).isoformat()
        self.save()

    @property
    def count(self) -> int:
        return len(self._uploaded)


# ---------------------------------------------------------------------------
# File validation helpers
# ---------------------------------------------------------------------------


def _extract_ticker(filepath: Path) -> str | None:
    """Return the ticker if the file is a direct child of a ticker folder.

    Expected structure: WATCH_DIR / TICKER / file.pdf
    Returns None if the file is nested deeper (e.g., in Processed/).
    """
    try:
        rel = filepath.relative_to(WATCH_DIR)
    except ValueError:
        return None

    parts = rel.parts
    # Must be exactly TICKER/filename (depth == 2)
    if len(parts) != 2:
        return None

    ticker = parts[0]

    # Ignore reserved directory names at the ticker level
    if ticker in IGNORE_DIRS:
        return None

    return ticker.upper()


def _should_ignore(filepath: Path) -> bool:
    """Return True if the file should be skipped."""
    name = filepath.name

    # Ignore by prefix
    for prefix in IGNORE_PREFIXES:
        if name.startswith(prefix):
            return True

    # Ignore by extension
    if filepath.suffix.lower() not in ALLOWED_EXTENSIONS:
        return True

    # Ignore if inside a reserved subdirectory
    for part in filepath.parts:
        if part in IGNORE_DIRS:
            return True

    return False


def _wait_for_file_ready(filepath: Path, timeout: float = 30.0) -> bool:
    """Wait until a file stops growing (iCloud download settling)."""
    deadline = time.monotonic() + timeout
    prev_size = -1
    while time.monotonic() < deadline:
        if not filepath.exists():
            return False
        current_size = filepath.stat().st_size
        if current_size == prev_size and current_size > 0:
            return True
        prev_size = current_size
        time.sleep(SETTLE_DELAY_SECONDS)
    return filepath.exists() and filepath.stat().st_size > 0


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------


def upload_pdf(filepath: Path, ticker: str, *, dry_run: bool = False) -> bool:
    """Upload a single PDF to the Charlie backend. Returns True on success."""
    file_size = filepath.stat().st_size

    if file_size > MAX_FILE_SIZE_BYTES:
        log.warning("SKIP %s -- file too large (%d MB)", filepath.name, file_size // (1024 * 1024))
        return False

    if file_size == 0:
        log.warning("SKIP %s -- empty file", filepath.name)
        return False

    if dry_run:
        log.info("DRY RUN: would upload %s (%s) [%.1f KB]", filepath.name, ticker, file_size / 1024)
        return True

    log.info("Uploading %s (%s) [%.1f KB] ...", filepath.name, ticker, file_size / 1024)

    try:
        file_data = base64.b64encode(filepath.read_bytes()).decode("ascii")
    except OSError as exc:
        log.error("Failed to read %s: %s", filepath.name, exc)
        return False

    payload = {
        "ticker": ticker,
        "documents": [
            {
                "filename": filepath.name,
                "fileData": file_data,
                "fileType": "pdf",
                "mimeType": "application/pdf",
            }
        ],
    }

    try:
        resp = requests.post(
            UPLOAD_ENDPOINT,
            json=payload,
            timeout=UPLOAD_TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 200:
            log.info("OK: %s uploaded to %s", filepath.name, ticker)
            return True
        else:
            log.error(
                "FAIL: %s -- HTTP %d: %s",
                filepath.name,
                resp.status_code,
                resp.text[:300],
            )
            return False
    except requests.RequestException as exc:
        log.error("FAIL: %s -- %s", filepath.name, exc)
        return False


# ---------------------------------------------------------------------------
# Filesystem event handler
# ---------------------------------------------------------------------------


class PDFHandler(FileSystemEventHandler):
    """React to new PDF files appearing in ticker folders."""

    def __init__(self, state: UploadState, *, dry_run: bool = False) -> None:
        super().__init__()
        self.state = state
        self.dry_run = dry_run

    def _process(self, filepath: Path) -> None:
        if _should_ignore(filepath):
            return

        ticker = _extract_ticker(filepath)
        if ticker is None:
            return

        abs_key = str(filepath.resolve())
        if self.state.is_uploaded(abs_key):
            log.debug("Already uploaded, skipping: %s", filepath.name)
            return

        # Wait for iCloud to finish writing the file
        if not _wait_for_file_ready(filepath):
            log.warning("File never became ready: %s", filepath.name)
            return

        success = upload_pdf(filepath, ticker, dry_run=self.dry_run)
        if success and not self.dry_run:
            self.state.mark_uploaded(abs_key)

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        self._process(Path(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        # A file moved *into* a watched folder triggers dest_path
        self._process(Path(event.dest_path))


# ---------------------------------------------------------------------------
# Initial scan
# ---------------------------------------------------------------------------


def initial_scan(state: UploadState, *, dry_run: bool = False) -> tuple[int, int]:
    """Scan all ticker folders for un-uploaded PDFs. Returns (found, uploaded)."""
    log.info("Starting initial scan of %s", WATCH_DIR)
    found = 0
    uploaded = 0

    if not WATCH_DIR.exists():
        log.error("Watch directory does not exist: %s", WATCH_DIR)
        return 0, 0

    for ticker_dir in sorted(WATCH_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        if ticker_dir.name in IGNORE_DIRS or ticker_dir.name.startswith("."):
            continue

        ticker = ticker_dir.name.upper()

        for pdf in sorted(ticker_dir.glob("*.pdf")):
            if _should_ignore(pdf):
                continue

            # Only direct children (glob("*.pdf") already ensures this)
            found += 1
            abs_key = str(pdf.resolve())

            if state.is_uploaded(abs_key):
                log.debug("Already uploaded: %s", pdf.name)
                continue

            success = upload_pdf(pdf, ticker, dry_run=dry_run)
            if success:
                uploaded += 1
                if not dry_run:
                    state.mark_uploaded(abs_key)

    log.info(
        "Initial scan complete: %d PDFs found, %d newly uploaded",
        found,
        uploaded,
    )
    return found, uploaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def show_status(state: UploadState) -> None:
    """Show a table comparing local iCloud files vs Charlie uploads."""
    if not WATCH_DIR.exists():
        log.error("Watch directory does not exist: %s", WATCH_DIR)
        return

    # Scan local files
    local: dict[str, list[dict]] = {}  # ticker -> [{name, size, uploaded}]
    for ticker_dir in sorted(WATCH_DIR.iterdir()):
        if not ticker_dir.is_dir() or ticker_dir.name in IGNORE_DIRS or ticker_dir.name.startswith("."):
            continue
        ticker = ticker_dir.name.upper()
        files = []
        for pdf in sorted(ticker_dir.glob("*.pdf")):
            if _should_ignore(pdf):
                continue
            abs_key = str(pdf.resolve())
            files.append({
                "name": pdf.name,
                "size_kb": pdf.stat().st_size / 1024,
                "uploaded": state.is_uploaded(abs_key),
                "upload_time": state._uploaded.get(abs_key, ""),
            })
        if files:
            local[ticker] = files

    # Fetch Charlie document counts
    charlie_counts: dict[str, int] = {}
    try:
        resp = requests.get(f"{API_URL}/api/pipeline/universe", timeout=30)
        if resp.status_code == 200:
            for stock in resp.json().get("universe", []):
                charlie_counts[stock["ticker"]] = 0
            # Fetch doc counts per ticker with docs
            for ticker in local:
                try:
                    r = requests.get(f"{API_URL}/api/pipeline/documents/{ticker}", timeout=15)
                    if r.status_code == 200:
                        charlie_counts[ticker] = r.json().get("total", 0)
                except Exception:
                    pass
    except Exception as e:
        log.warning("Could not fetch Charlie data: %s", e)

    # Print table
    print()
    print(f"{'TICKER':<8} {'LOCAL':>6} {'UPLOADED':>9} {'IN CHARLIE':>11} {'STATUS':<12}")
    print("-" * 52)

    total_local = 0
    total_uploaded = 0
    total_pending = 0

    for ticker in sorted(local.keys()):
        files = local[ticker]
        n_local = len(files)
        n_uploaded = sum(1 for f in files if f["uploaded"])
        n_pending = n_local - n_uploaded
        n_charlie = charlie_counts.get(ticker, "?")
        total_local += n_local
        total_uploaded += n_uploaded
        total_pending += n_pending

        if n_pending == 0:
            status = "OK"
            color = "\033[32m"  # green
        else:
            status = f"{n_pending} PENDING"
            color = "\033[33m"  # yellow

        print(f"{color}{ticker:<8} {n_local:>6} {n_uploaded:>9} {str(n_charlie):>11} {status:<12}\033[0m")

    print("-" * 52)
    print(f"{'TOTAL':<8} {total_local:>6} {total_uploaded:>9} {'':>11} ", end="")
    if total_pending > 0:
        print(f"\033[33m{total_pending} PENDING\033[0m")
    else:
        print("\033[32mALL SYNCED\033[0m")
    print()

    # Show pending files detail
    if total_pending > 0:
        print("PENDING FILES:")
        for ticker in sorted(local.keys()):
            for f in local[ticker]:
                if not f["uploaded"]:
                    print(f"  {ticker}/{f['name']} ({f['size_kb']:.0f} KB)")
        print()
        print(f"Run with --initial-scan to upload all pending files.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Charlie iCloud Watcher -- auto-upload PDFs to Charlie backend",
    )
    parser.add_argument(
        "--initial-scan",
        action="store_true",
        help="Scan all existing PDFs and upload missing ones before watching",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show sync status table (local files vs Charlie uploads) and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    if not WATCH_DIR.exists():
        log.error("Watch directory does not exist: %s", WATCH_DIR)
        sys.exit(1)

    state = UploadState(STATE_FILE)

    # -- Status mode ----------------------------------------------------------
    if args.status:
        show_status(state)
        return

    if args.dry_run:
        log.info("*** DRY RUN MODE -- no files will be uploaded ***")

    # -- Initial scan ---------------------------------------------------------
    if args.initial_scan:
        initial_scan(state, dry_run=args.dry_run)

    # -- Set up filesystem watcher --------------------------------------------
    handler = PDFHandler(state, dry_run=args.dry_run)
    observer = Observer()
    observer.schedule(handler, str(WATCH_DIR), recursive=True)

    shutdown_event = Event()

    def _signal_handler(signum: int, frame: object) -> None:
        signame = signal.Signals(signum).name
        log.info("Received %s, shutting down...", signame)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    observer.start()
    log.info("Watching %s for new PDFs (state: %d files tracked)", WATCH_DIR, state.count)
    log.info("Press Ctrl+C to stop")

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1.0)
    finally:
        observer.stop()
        observer.join(timeout=5.0)
        log.info("Watcher stopped. %d total files in state.", state.count)


if __name__ == "__main__":
    main()
