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
5. T24 implements opt-in lead/challenger/editor passes and saved-thesis context, with visible role outputs and checkpoints. 235 backend/20 frontend tests. This is draft reasoning review, not independently sourced sector-agent research. Live paid end-to-end validation remains for a real user task; never incur model charges solely for QA.
6. NEXT: expand trusted public-source collection and command-to-thesis proposal linkage. SEC is currently the only automatic public adapter. Consider verified investor-relations/FDA/clinical registry sources, bounded source registration and explicit provenance; avoid arbitrary URLs/SSRF. Command inputs currently use explicit ticker/date fields, not a general chat tool planner.
7. Improve automatic event follow-up windows, general server-worker recovery, full causal history and source/portfolio quality monitoring. Preserve the original weekly policy and never silently modify an approved thesis.

## Validation/release

Run `.venv/bin/python -m unittest discover -s tests/unit`, `npm run test:frontend`, `npm run build`; compile Mac paths under Python 3.9. Use CUA for UI inspection. Update release identifiers in `src/app.jsx`, `worker.js`, `service-worker.js`; copy `dist/tailwind.css` to ignored `dev/tailwind.css`. Commit/push only explicit task files; deploy using existing Wrangler configuration. Verify Render `/health` revision and frontend assets. Record what actually passed; source/paid-generation stubs are not live end-to-end validation.

Baseline T19: commit `698cd54`, frontend `2026-09-08T19`, 181 backend / 20 frontend tests. Existing weekly 70-ticker policies and prospective catalyst watch are enabled; managed browser work still requires Mac, Chrome authentication and worker availability. User originals remain in iCloud STOCKS/CATALYSTS ticker folders.

T22 deployed backend `00e9a22`, Cloudflare `db33daf2-4424-4ed8-9260-ea0146a05e74`; agent heartbeat verified current (~27 seconds). T23 release verification pending.

T23 deployed backend `01b1a95`, Cloudflare `42da20b3-59f9-486c-8f51-cba3709603b0`; restoration preview API verified 200 for DE. T24 deployed and verified: backend `4ebc895`, Cloudflare `5acf5820-1443-4941-8281-96bcdd9e6f9c`; 235 backend/20 frontend tests, live API and asset hashes checked. Continue work through the existing heartbeat build extension until 05:40 UTC.

## Trusted-source supplement increment

The local managed worker now supports FDA HTML and ClinicalTrials.gov study JSON supplements through `research_task_sources.py --supplement RECORD.json`. Original bytes, source date, observed URL, company-relevance assessment and hashes join the existing production manifest gate. SEC remains the deterministic lookup; FDA/registry discovery and publication/relevance verification require the scheduled worker. No general internet/IR adapter or unattended clinical-event validation is claimed. Next: structured command-to-thesis proposal chaining and broader recovery.

Supplement validation: all 241 safe backend tests passed, Mac Python 3.9 compile/import passed, and task diff checks passed. Local-worker-only change; frontend remains verified T24.

## T25 Command-to-thesis comparison

Completed Command cards now open a command-specific thesis comparison. The exact latest topic/ticker recap source register is matched to cloud-imported original bytes; missing, ambiguous, transformed or changed inputs are blocked. Assignment instructions carry forward, selected source hashes and command/activity provenance persist on the proposal, and hashes are checked again before model work. User selects up to 10 verified documents and explicitly requests comparison; existing selective approval remains. This is a reviewable handoff, not automatic paid proposal chaining or automatic local-source import. Next: import eligible event originals safely for comparison, then opt-in durable proposal orchestration; broader server worker recovery remains open.

T25 validation: 250 backend tests, 20 frontend tests and production build pass. Live completed-command comparison and paid generation are not exercised solely for QA; deployment verification follows.

T26 follow-up explicitly labels an existing same-ticker proposal from a different comparison, preventing it being mistaken for the current command output. Final validation remains 250 backend/20 frontend tests and build. T25 was briefly deployed; T26 supersedes its frontend cache identifiers.
