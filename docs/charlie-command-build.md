# Command Charlie build continuation

User authorized implementation, safe testing, commits and deployment. Temporary build follow-up ends September 8, 2026 at 05:40 UTC; existing collection heartbeat then returns to collection only. Preserve saved research and unrelated dirty files. Never run pytest: repository fixtures truncate the database. No paid research runs solely for QA.

## Current increment (not yet released)

`research_commands.py`, `research_task_sources.py`, `src/command-charlie.jsx`, collection refresh/source gating and Research desk navigation implement explicit ticker/date research assignments and cloud favorites. Commands are durable collection-control jobs; browser downloads and SEC sources must verify before covering-analyst recap dispatch. Original weekly policy stays unchanged. Thesis application remains a review action.

Before releasing: finish strict SEC collector/checkpoints and worker runbook integration; test malformed requests, idempotency, lease revocation, date/source coverage, source gates and manifest checks; inspect UI; bump all three frontend release identifiers; build and deploy explicit task files. Do not describe generic internet expansion as implemented: current automated public source adapter is SEC EDGAR only.

## Next increments

1. Interrupted-job recovery with durable checkpoints, bounded retries and protection against duplicate uncertain paid dispatches.
2. OCR for scanned source documents, preserving originals/page provenance and reporting extraction coverage.
3. Broader immutable version restoration with preview and conflict checks.
4. Real public-source annotated quality benchmarks; disclose coverage and failure thresholds.
5. Bounded coordinated Charlie analyst assignments, evidence/challenge/synthesis roles and visible subtasks. This means app functionality, not permission to spawn Codex development agents.
6. Expand trusted public-source collection and command-to-thesis proposal linkage, then remaining gaps in `docs/charlie-roadmap.md`.

## Validation/release

Run `.venv/bin/python -m unittest discover -s tests/unit`, `npm run test:frontend`, `npm run build`; compile Mac paths under Python 3.9. Use CUA for UI inspection. Update release identifiers in `src/app.jsx`, `worker.js`, `service-worker.js`; copy `dist/tailwind.css` to ignored `dev/tailwind.css`. Commit/push only explicit task files; deploy using existing Wrangler configuration. Verify Render `/health` revision and frontend assets. Record what actually passed; source/paid-generation stubs are not live end-to-end validation.

Baseline T19: commit `698cd54`, frontend `2026-09-08T19`, 181 backend / 20 frontend tests. Existing weekly 70-ticker policies and prospective catalyst watch are enabled; managed browser work still requires Mac, Chrome authentication and worker availability. User originals remain in iCloud STOCKS/CATALYSTS ticker folders.
