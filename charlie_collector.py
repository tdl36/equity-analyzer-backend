#!/usr/bin/env python3
"""Supervised AlphaSense collection ledger and verified iCloud handoff.

Browser interaction is performed by the supervising operator/Codex session.
This module never accepts credentials or drives an authenticated browser.
"""
import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
from urllib.parse import urlsplit, parse_qsl
import uuid
import zipfile

DEFAULT_STATE = Path.home() / "Library/Application Support/Charlie/AlphaSense"
DEFAULT_STOCKS = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/STOCKS"
KINDS = ("transcript", "broker-report")
MAX_FILE = 100 * 1024 * 1024
MAX_ARCHIVE = 500 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def ticker_name(value):
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", value):
        raise ValueError("Invalid ticker")
    return value


def source_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.hostname != "research.alpha-sense.com"
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or re.search(r"login|authenticate|authorize", parsed.path, re.I)
            or any(re.search(r"token|password|secret|code|state", k, re.I)
                   for k, _ in parse_qsl(parsed.query))):
        raise ValueError("Use an AlphaSense research/search permalink, not a sign-in URL")
    return value


def pdf_pages(data):
    from PyPDF2 import PdfReader
    if len(data) > MAX_FILE or not data.startswith(b"%PDF-"):
        raise ValueError("Download is not a supported PDF or exceeds 100 MB")
    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise ValueError("Encrypted originals require manual review")
        count = len(reader.pages)
        if not count:
            raise ValueError("PDF has no pages")
        return count
    except Exception as exc:
        raise ValueError("PDF validation failed; original remains untouched") from exc


def document_key(url):
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query))
    return query.get('docid') or (parsed.path.split('/doc-viewer/', 1)[1]
                                if '/doc-viewer/' in parsed.path else None)


def same_watermarked_export(left, right):
    """Compare exports of one verified document, ignoring only download timestamps.

    Never use this to infer identity: callers must first match AlphaSense doc IDs.
    Blank/image-only PDFs and exports without a recognized watermark fail closed.
    """
    from PyPDF2 import PdfReader
    def content(data):
        reader = PdfReader(io.BytesIO(data), strict=True)
        pages = []
        stamps = 0
        for page in reader.pages:
            text = page.extract_text() or ''
            text, count = re.subn(
                r'([\w.+-]+@[\w.-]+ - [^\n]*? - )\d{2}/\d{2}/\d{4} \d{2}:\d{2} (?:AM|PM) UTC',
                r'\1[download timestamp]', text)
            stamps += count
            pages.append(text)
        return pages if stamps and all(len(p.strip()) > 100 for p in pages) else None
    try:
        a, b = content(left), content(right)
        return a is not None and a == b
    except Exception:
        return False


def originals(path):
    """Validate the whole download before returning any files; never extract paths."""
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("Download exceeds 500 MB")
    if path.suffix.lower() == ".pdf":
        data = path.read_bytes()
        return [(path.name, data, pdf_pages(data))]
    if path.suffix.lower() != ".zip":
        raise ValueError("Select a completed PDF or ZIP download")
    files = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 500 or sum(m.file_size for m in members) > MAX_ARCHIVE:
            raise ValueError("Archive exceeds pilot limits")
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive contains a symbolic link")
            if member.is_dir() or "__MACOSX" in name.parts or name.name.startswith("."):
                continue
            if name.suffix.lower() != ".pdf":
                raise ValueError("Archive contains a non-PDF original; review it separately")
            if member.file_size > MAX_FILE:
                raise ValueError("PDF exceeds 100 MB")
            with archive.open(member) as stream:
                data = stream.read(MAX_FILE + 1)
            files.append((name.name, data, pdf_pages(data)))
    if not files:
        raise ValueError("No PDF originals in download")
    return files


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def handoff_fingerprint(documents):
    """Bind a verification to the exact eligible ledger state that was checked."""
    rows = sorted((d['id'], d['sha256'], d['status'], d['destination'])
                  for d in documents if d['usage'] == 'research')
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def publish(path, data):
    """Publish a complete file atomically without overwriting an existing original."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".charlie-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if file_hash(path) != hashlib.sha256(data).hexdigest():
                raise ValueError("Destination collision; existing file preserved")
    finally:
        os.unlink(temp)


class Collector:
    def __init__(self, state=DEFAULT_STATE, stocks=DEFAULT_STOCKS):
        self.state, self.stocks = Path(state), Path(stocks)
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state, 0o700)
        self.db = sqlite3.connect(self.state / "ledger.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, created TEXT, since TEXT, until_date TEXT);
            CREATE TABLE IF NOT EXISTS tasks (
                run TEXT, ticker TEXT, kind TEXT, status TEXT DEFAULT 'pending',
                expected INTEGER, PRIMARY KEY(run,ticker,kind));
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, run TEXT, ticker TEXT, kind TEXT,
                filename TEXT, sha256 TEXT, pages INTEGER, source_url TEXT,
                published TEXT, publisher TEXT, staged TEXT, destination TEXT,
                status TEXT DEFAULT 'staged', created TEXT,
                UNIQUE(run,ticker,kind,sha256));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, run TEXT, at TEXT, event TEXT, details TEXT);
            CREATE TABLE IF NOT EXISTS observations (
                run TEXT, ticker TEXT, kind TEXT, source_url TEXT, result_count INTEGER,
                note TEXT, checked TEXT, PRIMARY KEY(run,ticker,kind));
            CREATE TABLE IF NOT EXISTS verifications (
                run TEXT, ticker TEXT, checked TEXT, fingerprint TEXT,
                expected INTEGER, visible INTEGER, missing TEXT,
                PRIMARY KEY(run,ticker));
        """)
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(documents)')}
        if 'usage' not in columns:
            self.db.execute("ALTER TABLE documents ADD COLUMN usage TEXT DEFAULT 'research'")
            self.db.commit()
        task_columns = {r[1] for r in self.db.execute('PRAGMA table_info(tasks)')}
        for column in ('paused_status', 'evidence_after'):
            if column not in task_columns:
                self.db.execute(f'ALTER TABLE tasks ADD COLUMN {column} TEXT')
        self.db.commit()

    @contextmanager
    def lock(self):
        with (self.state / "collector.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another collection operation is active")
            try:
                with self.db:
                    yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def event(self, run, event, **details):
        self.db.execute("INSERT INTO events(run,at,event,details) VALUES(?,?,?,?)",
                        (run, now(), event, json.dumps(details)))

    def create(self, tickers, since, until):
        if date.fromisoformat(since) > date.fromisoformat(until):
            raise ValueError("Start date must be before end date")
        tickers = list(dict.fromkeys(ticker_name(t) for t in tickers))
        if not 1 <= len(tickers) <= 12:
            raise ValueError("Choose 1–12 tickers")
        run = uuid.uuid4().hex[:12]
        with self.lock():
            self.db.execute("INSERT INTO runs VALUES(?,?,?,?)", (run, now(), since, until))
            self.db.executemany("INSERT INTO tasks(run,ticker,kind) VALUES(?,?,?)",
                                [(run, t, k) for t in tickers for k in KINDS])
            self.event(run, "created", tickers=tickers, since=since, until=until)
        return self.status(run)

    def status(self, run):
        result = self.db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
        if result is None:
            raise ValueError("Unknown collection run")
        return {**dict(result), "tasks": [dict(r) for r in self.db.execute(
            "SELECT * FROM tasks WHERE run=? ORDER BY ticker,kind", (run,))],
            "documents": [dict(r) for r in self.db.execute(
                "SELECT * FROM documents WHERE run=? ORDER BY created", (run,))],
            "observations": [dict(r) for r in self.db.execute(
                "SELECT * FROM observations WHERE run=? ORDER BY ticker,kind", (run,))]}

    def auth(self, run, needed):
        self.status(run)
        with self.lock():
            if needed:
                self.db.execute("""UPDATE tasks SET paused_status=status,status='needs_auth',evidence_after=?
                    WHERE run=? AND status IN ('pending','review')""", (now(), run))
            else:
                self.db.execute("""UPDATE tasks SET status=COALESCE(paused_status,'pending'),paused_status=NULL
                    WHERE run=? AND status='needs_auth'""", (run,))
            self.event(run, "needs_auth" if needed else "auth_resumed")
        return self.status(run)

    def stage(self, run, ticker, kind, path, url, published=None, publisher=None, usage='research'):
        ticker = ticker_name(ticker)
        source_url(url)
        if usage not in ('research', 'reference_only'):
            raise ValueError('Unknown document usage')
        if published:
            date.fromisoformat(published)
            window = self.status(run)
            if not window['since'] <= published <= window['until_date']:
                raise ValueError('Publication date is outside the collection window')
        files = originals(path)
        with self.lock():
            task = self.db.execute("SELECT * FROM tasks WHERE run=? AND ticker=? AND kind=?",
                                   (run, ticker, kind)).fetchone()
            if not task or task["status"] in ("complete", "complete_with_exceptions", "no_results", "needs_auth"):
                raise ValueError("Task is missing, complete, or paused for sign-in")
            for name, data, pages in files:
                digest = hashlib.sha256(data).hexdigest()
                previous_usage = self.db.execute("""SELECT status,usage FROM documents
                    WHERE run=? AND ticker=? AND kind=? AND sha256=?""", (run, ticker, kind, digest)).fetchone()
                if (usage == 'reference_only' and previous_usage and
                        previous_usage['usage'] == 'research' and previous_usage['status'] in ('handed_off','duplicate')):
                    raise ValueError('New usage restriction on an existing handoff requires manual isolation review')
                staged = self.state / "originals" / (digest + ".pdf")
                publish(staged, data)
                key = document_key(url)
                if key:
                    candidates = self.db.execute(
                        "SELECT * FROM documents WHERE run=? AND ticker=? AND kind=?",
                        (run, ticker, kind)).fetchall()
                    previous = next((r for r in candidates if document_key(r['source_url']) == key), None)
                    if previous and previous['sha256'] != digest:
                        if usage == 'reference_only' or previous['usage'] == 'reference_only':
                            raise ValueError('Changed restricted export requires manual review')
                        if not same_watermarked_export(Path(previous['staged']).read_bytes(), data):
                            raise ValueError('Existing document has different content; review the new export')
                        self.event(run, 'alternate_export', document=previous['id'], sha256=digest,
                                   source_url=url, reason='Same document ID and content; download timestamp differs')
                        continue
                self.db.execute("""INSERT OR IGNORE INTO documents
                    (id,run,ticker,kind,filename,sha256,pages,source_url,published,publisher,staged,created)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (uuid.uuid4().hex[:16], run, ticker, kind, name, digest, pages,
                     url, published, publisher, str(staged), now()))
                # A restrictive observation can tighten a previous ingestion, never relax it.
                if usage == 'reference_only':
                    self.db.execute("UPDATE documents SET usage='reference_only' WHERE run=? AND ticker=? AND kind=? AND sha256=?",
                                    (run, ticker, kind, digest))
                self.event(run, "original_validated", ticker=ticker, kind=kind, sha256=digest)
            self.db.execute("UPDATE tasks SET status='review' WHERE run=? AND ticker=? AND kind=?",
                            (run, ticker, kind))
        return self.status(run)

    def handoff(self, document_id):
        with self.lock():
            row = self.db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown document")
            if row['usage'] == 'reference_only':
                raise ValueError('Reference-only original remains in staging; AI pipeline handoff is blocked')
            if row["status"] in ("handed_off", "duplicate"):
                return dict(row)
            staged = Path(row["staged"])
            if file_hash(staged) != row["sha256"]:
                raise ValueError("Staged original changed; handoff stopped")
            # Only existing ticker folders: never silently invent a company mapping.
            folder = self.stocks / row["ticker"]
            if not folder.is_dir() or folder.is_symlink():
                raise ValueError("Existing iCloud ticker folder is required")
            existing = None
            for candidate in folder.rglob("*"):
                if candidate.suffix.lower() == ".pdf" and not candidate.is_symlink():
                    if file_hash(candidate) == row["sha256"]:
                        existing = candidate
                        break
            filename = re.sub(r"[^\w .()-]", "_", Path(row["filename"]).stem)[:100].strip(" .") or "document"
            target = folder / f"{filename}--{row['sha256'][:12]}.pdf"
            if target.parent.is_symlink():
                raise ValueError("Handoff folder cannot be a symbolic link")
            if existing:
                target, status = existing, "duplicate"
            else:
                publish(target, staged.read_bytes())
                status = "handed_off"
            self.db.execute("UPDATE documents SET destination=?,status=? WHERE id=?",
                            (str(target), status, document_id))
            self.event(row["run"], status, document=document_id, destination=str(target))
        return dict(self.db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone())

    def finish(self, run, ticker, kind, expected):
        ticker = ticker_name(ticker)
        with self.lock():
            task = self.db.execute("SELECT * FROM tasks WHERE run=? AND ticker=? AND kind=?",
                                   (run, ticker, kind)).fetchone()
            if task is None or task["status"] == "needs_auth":
                raise ValueError("Task is missing or requires sign-in")
            if task['status'] in ('complete', 'complete_with_exceptions', 'no_results'):
                if task['expected'] != expected:
                    raise ValueError('Completed search count cannot be changed')
                return self.status(run)
            observation = self.db.execute('SELECT * FROM observations WHERE run=? AND ticker=? AND kind=?',
                                          (run, ticker, kind)).fetchone()
            if (observation is None or observation['result_count'] != expected or
                    (task['evidence_after'] and observation['checked'] <= task['evidence_after'])):
                raise ValueError('Record a current reviewed original count before completing the search')
            rows = self.db.execute("SELECT status,usage FROM documents WHERE run=? AND ticker=? AND kind=?",
                                   (run, ticker, kind)).fetchall()
            if expected < 0 or len(rows) != expected or any(r['status'] == "staged" and r['usage'] != 'reference_only' for r in rows):
                raise ValueError("Expected unique originals must match verified handoffs/duplicates")
            finished = 'complete_with_exceptions' if any(r['usage']=='reference_only' for r in rows) else ('complete' if expected else 'no_results')
            self.db.execute("UPDATE tasks SET status=?,expected=? WHERE run=? AND ticker=? AND kind=?",
                            (finished, expected, run, ticker, kind))
            self.event(run, "search_reviewed", ticker=ticker, kind=kind, expected=expected)
        return self.status(run)

    def observe(self, run, ticker, kind, url, count, note):
        """Record actual browser search evidence without marking unfinished work complete."""
        ticker = ticker_name(ticker)
        source_url(url)
        if type(count) is not int or count < 0 or not note.strip() or len(note) > 2000:
            raise ValueError('Invalid search observation')
        with self.lock():
            task = self.db.execute('SELECT * FROM tasks WHERE run=? AND ticker=? AND kind=?',
                                   (run, ticker, kind)).fetchone()
            if not task:
                raise ValueError('Unknown task')
            if task['status'] == 'needs_auth':
                raise ValueError('Resume after sign-in before recording browser evidence')
            if task['expected'] is not None and task['expected'] != count:
                raise ValueError('Completed search evidence count cannot be changed')
            self.db.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?)',
                            (run, ticker, kind, url, count, note, now()))
            self.event(run, 'search_observed', ticker=ticker, kind=kind, result_count=count)
        return self.status(run)

    def verify(self, run, fetcher=None):
        """Read production manifests and verify local bytes; never upload or generate."""
        documents = self.status(run)['documents']
        if fetcher is None:
            import requests
            from charlie_local_agent import CHARLIE_API, _agent_headers
            def fetcher(ticker):
                response = requests.get(CHARLIE_API + '/api/agent/local-files/' + ticker,
                                        headers=_agent_headers(), timeout=30)
                response.raise_for_status()
                return response.json()
        results = []
        for ticker in sorted({t['ticker'] for t in self.status(run)['tasks']}):
            eligible = [d for d in documents if d['ticker'] == ticker and d['usage'] == 'research']
            try:
                manifest = fetcher(ticker)
                files = manifest['files']
                if not isinstance(files, list) or any(not isinstance(f, dict) for f in files):
                    raise ValueError('Invalid manifest')
                missing = []
                for doc in eligible:
                    destination = Path(doc['destination']) if doc['destination'] else None
                    valid = False
                    if destination and doc['status'] in ('handed_off', 'duplicate'):
                        try:
                            relative = destination.relative_to(self.stocks / ticker)
                            folder = 'main' if relative.parent == Path('.') else relative.parent.as_posix()
                            valid = (file_hash(destination) == doc['sha256'] and
                                     any(f.get('filename') == relative.name and f.get('folder') == folder for f in files))
                        except (OSError, ValueError):
                            pass
                    if not valid:
                        missing.append(doc['id'])
                result = dict(ticker=ticker, checked=now(), expected=len(eligible),
                              visible=len(eligible)-len(missing), missing=missing)
                with self.lock():
                    current = [d for d in self.status(run)['documents'] if d['ticker'] == ticker]
                    if handoff_fingerprint(current) != handoff_fingerprint(eligible):
                        raise ValueError('Ledger changed during verification')
                    self.db.execute('INSERT OR REPLACE INTO verifications VALUES(?,?,?,?,?,?,?)',
                                    (run, ticker, result['checked'], handoff_fingerprint(eligible),
                                     len(eligible), result['visible'], json.dumps(missing)))
                    self.event(run, 'manifest_verified', **result)
                results.append(result)
            except Exception:
                # Requests errors can include headers/URLs. Keep previous successful
                # evidence and return a bounded, credential-free error.
                results.append(dict(ticker=ticker, error='Verification unavailable; previous evidence retained'))
        return {'run': run, 'verifications': results}

    def notify(self, run, event, sender=None):
        """One notification per run/event. Never send authentication secrets."""
        result = self.status(run)
        if event not in ("auth", "digest"):
            raise ValueError("Unknown notification event")
        if event == "auth":
            message = ("Charlie: AlphaSense collection is waiting for sign-in on your Mac. "
                       "Please complete password and verification directly in the AlphaSense browser tab. "
                       "Do not reply with a password or passcode. Run: " + run)
        else:
            if any(t["status"] not in ("complete", "complete_with_exceptions", "no_results") for t in result["tasks"]):
                raise ValueError("Collection is still incomplete; completion digest was not sent")
            docs = result["documents"]
            message = (f"Charlie: AlphaSense collection {run} completed. "
                       f"{sum(d['status'] == 'handed_off' for d in docs)} originals handed to iCloud; "
                       f"{sum(d['status'] == 'duplicate' for d in docs)} duplicates skipped. "
                       f"{sum(d['usage'] == 'reference_only' for d in docs)} reference-only originals held locally. "
                       "This confirms file handoff, not completion of research notes or theses.")
        with self.lock():
            previous = self.db.execute(
                "SELECT 1 FROM events WHERE run=? AND event=?", (run, "notification_" + event)).fetchone()
            if previous:
                return {"notification": "already_sent", "run": run}
            (sender or send_telegram)(message)
            self.event(run, "notification_" + event)
        return {"notification": "sent", "run": run}


def send_telegram(message):
    # Reuse the existing Mac agent's secret resolver; do not copy credentials.
    from charlie_local_agent import get_secret
    import requests
    token, chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise ValueError("Telegram is not configured in the existing Charlie agent")
    try:
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                 json={"chat_id": chat, "text": message}, timeout=15)
        if response.status_code != 200 or not response.json().get("ok"):
            raise ValueError("Telegram did not confirm delivery")
    except Exception:
        # Requests exceptions may embed a bot-token URL. Do not expose them.
        raise ValueError("Telegram delivery was not confirmed; check Telegram before retrying") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--stocks", type=Path, default=DEFAULT_STOCKS)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--tickers", nargs="+", required=True)
    create.add_argument("--since", required=True)
    create.add_argument("--until", default=date.today().isoformat())
    for command in ("status", "verify", "auth-needed", "resume", "stage", "finish", "observe"):
        p = commands.add_parser(command)
        p.add_argument("run")
        if command in ("stage", "finish", "observe"):
            p.add_argument("--ticker", required=True)
            p.add_argument("--kind", choices=KINDS, required=True)
        if command == "stage":
            p.add_argument("--file", type=Path, required=True)
            p.add_argument("--url", required=True)
            p.add_argument("--published")
            p.add_argument("--publisher")
            p.add_argument('--usage', choices=('research','reference_only'), default='research')
        if command == 'observe':
            p.add_argument('--url', required=True)
            p.add_argument('--count', type=int, required=True)
            p.add_argument('--note', required=True)
        if command == "finish":
            p.add_argument("--expected", type=int, required=True)
    handoff = commands.add_parser("handoff")
    handoff.add_argument("document")
    notification = commands.add_parser("notify")
    notification.add_argument("run")
    notification.add_argument("--event", choices=("auth", "digest"), required=True)
    args = parser.parse_args()
    collector = Collector(args.state, args.stocks)
    try:
        if args.command == "create":
            result = collector.create(args.tickers, args.since, args.until)
        elif args.command == "status":
            result = collector.status(args.run)
        elif args.command == "verify":
            result = collector.verify(args.run)
        elif args.command in ("auth-needed", "resume"):
            result = collector.auth(args.run, args.command == "auth-needed")
        elif args.command == "stage":
            result = collector.stage(args.run, args.ticker, args.kind, args.file, args.url,
                                     args.published, args.publisher, args.usage)
        elif args.command == 'observe':
            result = collector.observe(args.run,args.ticker,args.kind,args.url,args.count,args.note)
        elif args.command == "handoff":
            result = collector.handoff(args.document)
        elif args.command == "notify":
            result = collector.notify(args.run, args.event)
        else:
            result = collector.finish(args.run, args.ticker, args.kind, args.expected)
        print(json.dumps(result, indent=2))
        if args.command == 'verify' and any(v.get('error') or v.get('missing') for v in result['verifications']):
            parser.exit(1, 'Manifest verification needs attention.\n')
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Collection stopped: {exc}\n")
    finally:
        collector.db.close()


if __name__ == "__main__":
    main()
