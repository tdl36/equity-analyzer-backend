# Command Charlie build continuation

User authorized implementation, safe testing, commits and deployment. Temporary build follow-up ends September 8, 2026 at 05:40 UTC; existing collection heartbeat then returns to collection only. Preserve saved research and unrelated dirty files. Never run pytest: repository fixtures truncate the database. No paid research runs solely for QA.

## Released / current increment

T20 command/favorites was committed as `e8d2ebc` and deployed (Cloudflare `0b5e99ae-6683-472d-979b-fdd69515e231`). Backend health and command/favorite routes returned 200, and command cards were inspected in a fresh browser tab. SEC lookup succeeded, zero matching UNH filings for Sep 1–7. No paid research launched.

T21 adds the previously missing recap delivery endpoint, atomic receipts and bounded lease recovery for new Mac recap jobs. 208 backend/20 frontend tests and build pass. T21 backend health revision `0ee7b69` and lease/recovery endpoints were verified 200 (no jobs recovered/exhausted); Cloudflare `018e2c7f-f9de-4801-9f23-41f70f399591`. Mac helpers must not import Flask at module scope: system Python 3.9 lacks Flask. Command validation and result hashing now defer Flask imports to blueprint creation.

Current automated public-source adapter is SEC EDGAR only; wider internet retrieval and automatic structured thesis proposal chaining remain open.

## Next increments

1. Extend interrupted-job recovery beyond new Mac recap jobs to server thesis/edit/chat workers, using durable checkpoints and bounded retries.
2. Local OCR is implemented in T22 with 217 tests plus a real OCR fixture. Extend cloud OCR availability/coverage later; do not claim comprehensive image/table extraction.
3. T23 implements note/review restoration with preview, immutable clones and source/latest hash checks. Confirm deployed preview UI. General thesis/event draft restoration remain open. T23 has 226 backend/20 frontend tests.
4. T22 adds two historical FDA excerpt regression packs; expand to financial multi-source packs and analyst/expert grading. `scripts/evaluate-research-quality.py --suite` runs the current three control packs.
5. NEXT: bounded coordinated Charlie analyst assignments: lead/challenge/editor roles with checkpoints, exact source provenance, visible role outputs, user-selectable extra passes and final evidence audit. App functionality, not permission to spawn Codex development agents. Current recap audit already independently checks selected claims; do not mislabel ungrounded critique as source verification.
6. Expand trusted public-source collection and command-to-thesis proposal linkage, then remaining gaps in `docs/charlie-roadmap.md`.

## Validation/release

Run `.venv/bin/python -m unittest discover -s tests/unit`, `npm run test:frontend`, `npm run build`; compile Mac paths under Python 3.9. Use CUA for UI inspection. Update release identifiers in `src/app.jsx`, `worker.js`, `service-worker.js`; copy `dist/tailwind.css` to ignored `dev/tailwind.css`. Commit/push only explicit task files; deploy using existing Wrangler configuration. Verify Render `/health` revision and frontend assets. Record what actually passed; source/paid-generation stubs are not live end-to-end validation.

Baseline T19: commit `698cd54`, frontend `2026-09-08T19`, 181 backend / 20 frontend tests. Existing weekly 70-ticker policies and prospective catalyst watch are enabled; managed browser work still requires Mac, Chrome authentication and worker availability. User originals remain in iCloud STOCKS/CATALYSTS ticker folders.

T22 deployed backend `00e9a22`, Cloudflare `db33daf2-4424-4ed8-9260-ea0146a05e74`; agent heartbeat verified current (~27 seconds). T23 release verification pending.
