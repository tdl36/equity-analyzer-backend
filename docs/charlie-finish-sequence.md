# Charlie agreed finish sequence — October 15, 2026

The user authorized continued implementation of this sequence and requested a
progress update every 15 minutes. Existing continue-charlie-improvements heartbeat
was updated to 15 minutes; report shipped work, current work, blockers and next
step, including honestly when nothing new shipped. This is a cadence request,
not permission to generate research or accept drafts during backfill.

## 1. Unified processing control — active engineering task

Extend Ready to process beyond event-folder proposals to expose Lab stock-source
generation and analyst tasks. Show sources, intended output, readiness blockers,
credit use and exact action. Keep processing approval separate from acceptance of
finished research. Preserve the current event checklist and its atomic dispatch.

Starting checkpoint: T306 is verified in production. ReadyProcessing lives in
src/ready-processing.jsx and loads independently of the desk overview. Lab saved
selections are in app_settings under pipeline_lab_v1:<ticker>, exposed individually
by pipeline_lab.py. Inspect existing config/check/generation contracts before
adding an aggregate readiness inventory; inventory alone is not byte verification.

Analyst activity /run dispatches earnings_recap or takeaway generation, whereas
/approve accepts finished research and can file/sync it. Never wire the latter
into bulk processing. Inspect _dispatch_activity_run identity/concurrency handling
before exposing run actions. Treat uncertain outcomes as unresolved, not retryable.
Do not label a saved selection or a missing output as ready without validation.

Acceptance: a single visible entry distinguishes event proposals, saved Lab packs,
analyst work and completed drafts; failures stay explicit; actions preserve exact
source/config identity and cannot duplicate paid work or accept outputs. Test with
synthetic fixtures and existing read-only records; no paid QA.

## 2. Source-to-output journey

Trace existing authorized collection requests through preservation, verified cloud
imports, organized inventory, date coverage, frozen selections and saved output
links. Operations owns browser collection. Correct defects found without creating
another collector. Source-only requests remain generation-free.

## 3. Historical valuation

Resume docs/historical-security-mapping.md. Obtain a permitted retained reference
schema and explicit currency/security semantics, implement normalization and
assignment-bound storage, then connect valuation and one-pagers. Do not infer
historical currency/share continuity from current profiles. Missing external
evidence does not block independent work in this sequence.

## 4. Home and company reconciliation

Compare real saved jobs, failures, reviews and outputs with Home and company views.
Exercise exact-object links, back/forward, mobile, long content, reloads, unsaved
edits and service failures. T304-T306 are bounded fixes, not complete acceptance.

## 5. Genuine assignment proof

Observe a genuinely authorized ticker/window/output assignment from discovery
through downloads, exact-byte imports, approved processing and saved drafts.
Record receipts for each stage and recoverable interruption states. During the
backfill pause, stop before generation unless the specific request authorizes it.
Investor review remains separate. Do not manufacture a paid acceptance test.

## 6. Close evidence ledger

Update docs/eight-priority-acceptance.md with concrete evidence and remaining
issues. Distinguish implementation, synthetic validation, real-run proof and
investor acceptance. Resolve the legacy WELL outcome only from actual evidence.
Preserve the original Pipeline, originals, active work and unrelated dirty files.
Use the full AGENTS.md release procedure for implemented changes.
