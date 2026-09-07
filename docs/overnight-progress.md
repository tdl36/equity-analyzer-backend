# Charlie overnight work — September 6, 2026

User authorized continued implementation, commits, and production deployment.

## Verified
- Eight eligible pilot PDFs relocated from STOCKS/<ticker>/AlphaSense/ into STOCKS/<ticker>/; hashes verified, ledger updated. Production agent manifest confirms all eight with folder `main`. New handoffs now use the ticker root so existing new-file alerts also see them.
- Pilot f287ec99ad89 dates 2026-08-07 to 2026-09-06. DE transcript 1, ABT transcript 0, AMT transcript 1, AMT brokers 8 (5 handed off, 3 reference-only held). DE broker search has 45 hits; one already handed off. ABT broker search outstanding.
- DE first 20 selected; exported `DE_2026-08-07_to_09-06_broker_batch1.zip` to private state downloads. Pending validation/staging. Native Chrome currently retains first 20 selected. Must unselect these and select next 20, then final 5; AlphaSense export max20.
- Private ledger/downloads: ~/Library/Application Support/Charlie/AlphaSense/. Reference-only originals must stay outside STOCKS/CATALYSTS because these are AI ingestion folders.
- Local read-only monitor http://127.0.0.1:8766/ running via collector_status.py.
- Frontend9 + standalone Python16 tests pass, build succeeds. Do not run pytest: global conftest truncates test DB.

## Next
- Finish production commit/push and Wrangler deployment, verify /version and /api/buildinfo. Version worker/frontend 2026-09-06T02, service worker20260906-02.
- Finish DE remaining batches and ABT brokers. Record actual observed source URLs, counts, publication dates/publishers/usage flags, validate ZIPs before any handoff. No paid research generation.
- Verify production manifests after new handoffs, inspect mobile layout, make monitor startup durable.
- Record completion evidence here. Finite overnight Codex heartbeat `continue-charlie-overnight` hourly4runs ends no later than 2026-09-07 05:45 UTC.
