# Professional research operations

## Delivered

Research Desk (Automations → Research desk) combines live agent runs, the analyst review inbox, sector coverage, and a thesis maintenance queue. It reads production data through the existing authenticated API connection. The queue uses the actual `updated` field returned by `/api/analyses`, with configurable 30/60/90/180-day thresholds; age is a maintenance rule, not an investment ranking.

The batch planner deduplicates tickers, validates symbols, limits batches to 12 companies, obtains model choices from the backend, and shows a concrete review before submission. Browser-local reusable plans retain company groups and model preferences. Saved plans never run automatically. Runs are inspected with their decision, reports, analysis date and execution log; records marked active for more than two hours are flagged for a status check; analyst outputs can be read before opening the existing analyst controls.

Four workflow playbooks make existing capabilities easier to use: coverage refresh, thesis challenge, management meeting preparation, and thesis-to-deliverables. The analyst roster continues to use existing coverage, playbook and auto-mode configuration. No new analyst identities or recurring schedules were created automatically.

Harbor, Graphite and Parchment join the four existing themes. All seven use the existing color-token and persistence system. Appearance settings include named preview cards.

## Parallel execution

`agent_batch.py` validates and executes independent company teams with 1–3 workers. A shared semaphore caps batch work at three concurrent jobs **per Python process**, including overlapping batches. Batches are limited to 12 unique symbols. A failing company does not stop siblings. Default concurrency remains one for existing callers.

The runner no longer replaces `builtins.print`, which previously risked mixing logs between simultaneous teams. Explicit lifecycle events remain scoped to each run; verbose framework output remains in server logs.

`GET /api/agents/capabilities` advertises limits. The frontend offers parallel execution only when that backend confirms support. Until the changed backend is deployed, the production-backed preview can still prepare and submit ordinary sequential batches.

This is bounded concurrency for the existing runner, not a durable queue or a universal cost governor. Single-run endpoints and other research engines retain their existing execution behavior. Multi-process deployments multiply the per-process cap. Process restarts can interrupt queued/running work. No backend deployment or paid research job was performed during development.

## Validation

- Nine frontend test groups cover routing, production timestamp normalization, coverage merging, batch symbols, maintenance queues, run counts, and slide selection.
- Four database-independent Python tests exercise batch validation, concurrent execution, overlapping-batch limits, failure isolation and sequential defaults.
- Python compile check and production frontend build.
- Local API checks confirm capability limits and rejection of empty batches, invalid dates and excess concurrency before inserting runs.
- Browser checks use saved production data; batch preparation stops before paid submission.

## Next engineering priorities

1. **Durable research queue:** Postgres job leases, worker heartbeats, resumable stages, idempotency keys and cancellation. Enforce organization-wide concurrency and spend budgets across workers.
2. **Evidence ledger:** Bind claims and scenario inputs to document IDs, page/paragraph anchors, publication dates and extraction versions. Add a contradiction-review agent that reports competing claims rather than silently choosing one.
3. **Earnings change agent:** Compare the new release, transcript and guidance with prior periods and the saved thesis. Separate reported numbers from interpretation and queue material changes for review.
4. **Portfolio decision journal:** Record rationale, expected catalyst, risk, confidence, horizon and review date. Evaluate decisions against subsequent evidence and attribution, without rewriting the original rationale.
5. **Research evaluations:** A fixed set of historical cases for citation validity, numeric accuracy, omitted risks and contradiction recall; compare model/agent configurations before release.
6. **Shared investment-committee work:** Assign reviewers, persist comments, resolve disagreements and snapshot the evidence available at approval. Add permissions and auditable transitions before broader team rollout.
7. **Event-driven automation:** After durable execution and cost controls, route new filings, earnings events and material signals into research queues. Use explicit freshness/materiality rules and deduplication to avoid repeated work.
