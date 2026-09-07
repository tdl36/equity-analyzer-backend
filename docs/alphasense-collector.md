# AlphaSense collection pilot

Charlie now has a local collection ledger, PDF/ZIP validation, duplicate detection,
an atomic iCloud handoff, and optional Telegram notifications. Browser collection
is currently supervised through Codex computer use. This is **not yet an unattended
browser worker**. No scheduler, AlphaSense API integration, WhatsApp integration,
or automatic thesis edits are enabled.

## Authentication

Complete sign-in and MFA directly in AlphaSense. The collector never asks for or
stores a password, MFA code, browser cookie, or authentication token. If the
embedded browser cannot accept typing, use a supported connected browser or the
native computer-use surface after macOS permissions are enabled. Never work around
that problem by putting credentials into the collection ledger.

## Storage and workflow

- Private local state: `~/Library/Application Support/Charlie/AlphaSense/`.
- SQLite ledger: runs, separate transcript/broker-report searches per ticker,
  original SHA-256 hashes, source links, optional publication date/publisher,
  local destinations, and event history.
- Originals stage locally. Ingestion does not move or delete the downloaded file.
- Handoff requires an existing ticker folder and writes complete PDFs into
  `~/Library/Mobile Documents/com~apple~CloudDocs/STOCKS/<ticker>/`.
- Existing ticker PDFs are compared by content. Identical originals are skipped;
  different documents with the same filename get different hash suffixes.
- The existing Mac agent scans this ticker root for new-file alerts and includes it in its manifest. A successful
  handoff does not prove backend upload, note generation, or thesis revision.
  Verify those separately in Charlie's source picker before starting research.

Each search task ends only after the supervising browser operator checks all
results/pages for the ticker, date interval, and document type, and records the
number of unique downloadable originals. Downloads unavailable under the account's
entitlements must be left for review, not counted as a successful full collection.
Never infer a company identity from a filename alone. Confirm the AlphaSense company
filter, especially for ambiguous tickers. No search completion is inferred from an
empty page while unauthenticated.

## Pilot commands

Run from the repository using `.venv/bin/python charlie_collector.py`:

```sh
.venv/bin/python charlie_collector.py create --tickers DE ABT AMT --since 2026-08-07 --until 2026-09-06
.venv/bin/python charlie_collector.py auth-needed RUN_ID
.venv/bin/python charlie_collector.py status RUN_ID
```

After the user completes sign-in, `resume RUN_ID` clears the authentication pause.
Search one ticker and document type at a time in AlphaSense, with the requested
date range. Download permitted original PDFs, or a ZIP of originals through the
visible UI. Use the browser's supported download facility; do not extract session
cookies or call undocumented authenticated endpoints.

```sh
.venv/bin/python charlie_collector.py stage RUN_ID --ticker DE --kind transcript --file /absolute/download.pdf --url 'https://research.alpha-sense.com/OBSERVED_RESEARCH_LINK'
.venv/bin/python charlie_collector.py handoff DOCUMENT_ID
.venv/bin/python charlie_collector.py finish RUN_ID --ticker DE --kind transcript --expected 1
```

The source URL must be the actual observed document/search permalink. The sample
above is a placeholder. `--published YYYY-MM-DD` and `--publisher NAME` are optional
when verified in the UI. A ZIP must contain originals from one confirmed ticker and
document type; use individual files when publication dates or publishers differ.
Unknown metadata stays unknown. Encrypted, malformed, non-PDF, oversized, or unsafe
archives stop for review. Repeating a stage/handoff is idempotent for identical
content. The expected count is the number of unique originals, not search hits
when a search contains duplicate versions.

After reviewing an actual empty search, use `finish ... --expected 0`. Do not use
this merely because downloads failed. A task cannot finish with unhanded staged
files or a count mismatch. A single local operation lock prevents overlapping
ledger mutations; browser navigation remains one supervised worker.

## Notifications

```sh
.venv/bin/python charlie_collector.py notify RUN_ID --event auth
.venv/bin/python charlie_collector.py notify RUN_ID --event digest
```

These commands explicitly send to the Telegram destination configured for the
existing local agent, resolving secrets through its environment/Keychain/config
helper. Auth messages ask the user to sign in on their Mac, never to send a code.
Digest sends only when every search is complete or reviewed with no results.
Confirmed sends are deduplicated per run/event. Delivery errors are reported with
secrets removed; an ambiguous network failure may still have delivered, so check
Telegram before retrying. No messages are sent by creating or inspecting a run.

## Validation and next implementation boundary

Run `.venv/bin/python tests/unit/test_charlie_collector.py` directly. It uses
temporary folders and an isolated SQLite ledger; it does not touch Charlie's
PostgreSQL database, iCloud data, or Telegram.

The first live pilot must still validate login/session persistence, AlphaSense
company/type/date filters, pagination, original-download behavior, iCloud
availability, and backend source visibility. Then add a durable browser worker
with explicit authentication checkpoints and bounded retries, followed by the
Charlie collection-status UI and a schedule. Record inaccessible documents and
per-document metadata from actual results before enabling unattended collection.
Do not promise full automation based only on passing local ingestion tests.

Known pilot limits: 12 tickers per run, 100 MB per PDF, 500 MB/500 entries per ZIP;
PDF originals only. Hashing existing iCloud originals can require their local
materialization. State and provenance are local to this Mac, and are not a cloud
backup or a multi-machine queue. No cross-run date watermark advances automatically.
