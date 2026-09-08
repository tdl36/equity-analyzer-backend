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

T26 verified: backend `e00a3fd`, Cloudflare `ad7fae34-175c-472f-b6a6-530dd20d67a6`, live command/amendment routes and frontend hashes passed. The comparison panel was inspected using a styled synthetic browser fixture; source selection enabled submission without executing research.

## Automatic command original import

Before managed command recap dispatch, collection completion now imports only registered handed-off research-eligible AlphaSense originals and verified public originals into document_files. Exact SHA-256 receipts are required; same-name different-content originals are never overwritten. Imports are idempotent and retryable without starting research; the collection lease fences each import. Restriction-held and unregistered folder files are excluded. Limit 60 originals and 60 MB per file. New recap snapshots record original-file hashes separately from extracted-input hashes, so transformed sources can link back to imported originals. Older text snapshots without original hashes remain blocked if bytes differ. General unregistered iCloud auto-import and automatic paid thesis proposal execution remain open.

Validation: 258 backend tests plus a targeted restricted-file regression pass; Mac Python 3.9 imports pass. Backend/local-worker-only release, compatible with T26 frontend; no live import or paid research was executed solely for QA. Next: durable proposal checkpoint/recovery and explicit opt-in post-recap proposal orchestration.

## T27 Durable thesis-proposal checkpoints

Proposal draft and independent-review outputs are now persisted as stages, keyed to the complete source prompt, original hashes and configured model. A later failure preserves completed stages. Failed proposals with checkpoints offer explicit Resume saved proposal work, capped at two attempts under the same job ID. Resume rejects a changed saved thesis, another active proposal, changed source/prompt/model identity, and cancelled late writes. This does not yet recover abandoned running proposals automatically; safe ownership/lease recovery is the next increment. No paid research was launched for testing. 265 backend/20 frontend tests and build pass.

T27 verified: backend `c39cf88`, Cloudflare `08f4211f-59d0-47d3-a42b-3f5091e48b4b`; live health/history, unknown-job resume rejection and frontend hash checks passed.

## T28 Abandoned running proposal recovery

New thesis proposals hold a PostgreSQL session advisory lock during execution. The Mac heartbeat checks running recoverable proposals older than two minutes, skips live owners, and requeues abandoned work at most twice under the same ID. Worker tokens fence checkpoint writes and final results after ownership changes. Recovery requires a server research key and rejects changed saved theses. SQL connection loss releases execution ownership; a lost in-flight provider response may still need recomputation. One pooled DB connection is held for each executing proposal. Queued jobs, old proposals without recoverable=true, ordinary provider failures and automatic paid command-to-proposal chaining remain open. UI discloses recovery count/dependencies. 272 full backend tests, 20 frontend tests, build and Mac compile pass, plus ownership regressions after final cleanup hardening.

T28 verified: backend `f098df4`, Cloudflare `d7d0473f-ba51-4f58-bc37-f937f607edbf`; recovery returned 200 with no abandoned jobs, agent heartbeat was ~6 seconds old, and the frontend hash matched. Synthetic browser fixture showed recovery count/dependencies.

## SEC financial control pack

Added a frozen MSFT FY2025 10-K XBRL pack (four FY2024/FY2025 revenue/operating-income facts, selected by start/end period and accession). Four structured Decimal checks cover revenue, growth, operating margin and percentage-point change, with nine adversarial controls plus source-hash drift checks. This is offline annotation/fixture scoring, not automatic prose extraction or a live/expert model benchmark. 275 full backend tests and four evaluation packs pass, with targeted numeric-bound/source regression checks after final hardening. No model calls. Frontend remains verified T28; evaluator-only change needs no UI release.

Next remaining build priorities: queued proposal recovery without misclassifying live waiting work; opt-in automatic command-to-proposal dispatch after verified recap; broader expert/real-output benchmarks and comprehensive claim/financial reconciliation. Continue scheduled collection as configured after 05:40 UTC.

## Final recovery hardening and handoff

The recovery scan now considers up to 20 candidates while changing at most three jobs, avoiding starvation behind live owners. Manual failed-job resume takes the same execution lock and rejects a still-releasing worker before requeueing. 276 backend tests pass, including live-owner skipping, bounded dispatch and resume-race coverage. This is backend-only and remains compatible with T28. See `docs/charlie-overnight-status.md` for completed features and remaining gaps. At this check, 70 saved policies remain and no request is due. Final endpoint verification follows deployment.

## T29 Opt-in automatic command-to-proposal handoff

The user resumed development after the overnight window. Commands and editable favorites now support autoProposal (off by default). The Mac heartbeat advances opted-in commands after a completed linked recap, verifies every source, requires 1–10 originals, and submits the existing source-checked amendment workflow using deterministic IDs. Missing inputs, over-limit sets, missing server key or competing proposals remain visible as blocked dispatch records. Candidate rotation avoids starving older blocked commands; authoritative linked proposal status appears in task history. Existing/manual proposals for the command prevent another automatic proposal. No thesis changes are applied automatically. 282 backend tests, 20 frontend tests, build and Mac compile pass. Live paid end-to-end validation remains for a real assignment.
