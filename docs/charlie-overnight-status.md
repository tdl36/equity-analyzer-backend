# Charlie overnight build status — September 8, 2026

The frontend is release **T28**. Backend and local-worker changes are committed on `main`; exact release checks are recorded in the private local release receipts. The overnight feature-development window ends at 05:40 UTC. The existing managed collection schedule continues afterward.

## Built and checked

- **Command Charlie and favorites:** explicit ticker/date/window assignments, four editable reusable templates, cloud persistence, task history and managed collection handoff.
- **Document collection:** SEC filing lookup, worker-reviewed FDA/clinical-registry supplements, verified iCloud event folders, source restrictions and original hashes. Newly collected managed command originals are imported into Charlie before recap dispatch, with conflict protection.
- **Research output:** frozen input/source records, selected-claim source checking, saved-thesis context and opt-in lead/challenger/editor passes with checkpoints.
- **OCR:** bounded local PDF OCR, private page caching and visible extraction limitations. Originals are preserved.
- **Recap delivery/recovery:** durable outbox receipts, matching Inbox delivery and bounded recovery of new Mac synthesis jobs.
- **Thesis comparison:** completed commands link to exact recap source inputs. Verified originals and assignment instructions carry into reviewable proposals. Changed or missing inputs are explicit.
- **Proposal recovery:** saved draft/review checkpoints, explicit failed-job resume, and automatic recovery of new abandoned running proposals. Live database execution locks and ownership tokens protect concurrent work. Recovery is bounded and requires the Mac heartbeat and a server key.
- **Version restoration:** note/review preview and immutable restoration copies, plus the existing guarded amendment rollback. Historical review dates remain explicit.
- **Quality controls:** four offline packs, including historical FDA excerpts and SEC financial facts with arithmetic, unit and fiscal-period checks. These are developer-authored controls, not live model or expert-quality certification.

## Validation

276 safe backend unit tests pass. The latest frontend changes passed 20 frontend tests and the production build. New controls were checked with synthetic browser fixtures. Production endpoints, frontend hashes and Mac health were checked during each deployed application increment. No paid research was started or interrupted solely for QA, and no thesis edits, notes or recaps were approved automatically.

The final recovery hardening scans up to 20 candidates so live jobs do not hide abandoned work, but changes at most three jobs per check. Manual resume also waits for the previous worker to release execution ownership.

## Still needed

1. A real user assignment to validate the complete AlphaSense-to-report-to-thesis-proposal path. The full 70-ticker unattended rollout has not been demonstrated; saved policies are not evidence of completed downloads.
2. Opt-in automatic paid proposal dispatch after a completed recap. Current commands offer a reviewable comparison step.
3. Recovery for queued proposals, older jobs, and broader server thesis/chat/note workers. Existing automatic proposal recovery covers new running proposals only.
4. Broader trusted investor-relations/web adapters, event follow-up windows, and general unregistered iCloud source import.
5. Cloud OCR and more comprehensive image/table extraction.
6. Full thesis/event restoration, broader real-output benchmarks, independent analyst grading and comprehensive financial/claim reconciliation.

Collection still depends on an available Mac, signed-in Chrome and the managed browser worker. Authentication is completed directly in AlphaSense. Evidence gaps, restricted files and retrieval failures remain visible; unavailable evidence is not treated as a successful empty search.

## Subsequent morning build: T29

Following the user’s request to keep building, Command Charlie and favorites gained an optional automatic thesis-proposal handoff. This implements the opt-in dispatch item above, subject to source verification, the 10-document comparison limit, available server credentials and no competing proposal. The live end-to-end validation gap remains. Latest checks: 282 backend tests, 20 frontend tests and build.
