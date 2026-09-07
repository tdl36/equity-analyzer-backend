# Charlie implementation sequence

This is an implementation ledger, not a claim that the whole roadmap is complete.

1. Claim validation: selected recap claims now receive exact source/page passage matching and a separate model support check. Extraction gaps and review limits are disclosed. Full claim coverage, mathematical reconciliation and independently benchmarked entailment remain open.
2. Structured investment changes: source-linked change records are generated with the audit; absent thesis baselines remain explicit. Deterministic before/after financial-period comparisons remain open.
3. Integrated event workflow: event workspace and collection ledger exist separately. Connect live collection and filesystem status into a single event timeline; add source freshness.
4. Thesis amendment review: existing source-backed proposals support individual decisions. Extend version restoration and event-to-thesis review linking.
5. Resilient processing: recap delivery outbox and preflight exist. Automatic partitioning, OCR/hydration and resumable generation checkpoints remain open.
6. AlphaSense operations: scheduled worker and local ticker policies exist. Mobile controls, full coverage collection, unattended download validation and automatic event-folder suggestions remain open.
7. Portfolio-aware prioritization: connect verified holdings/exposure and user materiality preferences, with stale position handling.
8. Evaluation: deterministic unit regressions exist. Add representative source packs, scoring and release thresholds for research quality.

Agent interaction: event-scoped revision conversations now submit instructions and the prior draft to the covering analyst through existing regeneration. Prior drafts and their evidence are retained (last five), instruction history last 30. Replies show queue/processing state, not invented agent speech. Saved research approval is manual. General conversational Q&A, thesis/note revision chat, cross-agent assignment and broader durable chat history remain open. Generation is asynchronous, not instant editing. No paid production regeneration is part of automated QA.

## T12 implementation update

- Event folder inventories are now fetched from the existing authenticated agent manifest and compared by relative filename against the draft. Timestamp/unknown states remain visible. This is not a content-hash freshness check.
- Thesis proposal instructions are persisted in job input and participate in request-id conflict detection. The analyst can request targeted changes before reviewing source-supported field edits.
- Applied thesis proposals have a conflict-protected restore action: restore only if the complete current thesis equals the recorded applied snapshot. Later analyst changes block restoration; repeat restore requests are idempotent.
- Local text-only recap providers now partition oversized extracted text into bounded batches, retaining every source character. OCR, native-PDF splitting and durable generation checkpoints remain open.
- Regression suite includes forged quotes, incorrect source/page references, fallible reviewer verdicts, missing baselines, revision context, restoration conflicts and batch preservation.

The eight-workstream roadmap is not complete. Mobile collection control, full unattended AlphaSense coverage, portfolio-aware priorities, broad agent conversation, OCR/hydration and resumable generation still need implementation and validation.
