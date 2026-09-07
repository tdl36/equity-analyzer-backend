# Charlie implementation ledger — September 7, 2026

Releases T12–T14 implement the first working increments across the roadmap. This is not a certification of error-free research or complete unattended coverage.

1. **Claim validation:** new recaps receive selected-claim exact source/page matching and a separate model support check. Source extraction gaps and excerpt limits are disclosed. Full claim coverage, arithmetic reconciliation and independently benchmarked entailment remain open.
2. **Investment changes:** new recaps include structured, source-linked investment changes, with missing baselines identified. Deterministic financial-period bridges remain open.
3. **Event workspace:** live authenticated iCloud filename inventory is compared with the draft source register. Unknown/stale states remain visible. This comparison does not detect changed contents within an existing filename.
4. **Revisions and versions:** event instructions rerun the covering analyst against sources and the prior draft. The last five drafts/evidence and 30 instructions are retained. Thesis proposals accept targeted instructions and support selected edits plus restoration when the current thesis exactly matches the applied snapshot. Later edits block restoration.
5. **Processing:** text-only recap inputs are partitioned without dropping characters. Merges receive the full preceding draft. Partial batch checkpoints are atomic/private and keyed to source hashes, prompt, model and batch count. They are removed only after the final result is in the durable delivery outbox. Explicit output-token truncation is rejected. OCR and native-PDF splitting remain open; existing iCloud hydration retries remain in place. Checkpoints resume matching retries; they do not automatically restart failed jobs.
6. **AlphaSense control:** authenticated cloud commands and Mac receipts enable mobile policy changes and refresh triggers. Local policy mutation/refresh creation and receipt commit together, preventing replay after an earlier refresh completes. A cloud command marked applied means the local manager accepted it, not that browser downloads completed. Browser requests and Mac report age are shown separately. Codex/Mac/Chrome availability and sign-in remain necessary. Legacy 49-ticker coverage has not been silently enabled or completed. Mobile cancel/retry controls and unattended validation across full coverage remain open.
7. **Portfolio priorities:** user-reported signed weights, as-of dates, thesis age and pending/failed activities drive an explicit research ordering heuristic. Weights older than 30 days are excluded. No holdings are fabricated or independently verified; broker import and richer risk/materiality models remain open. Ordering never triggers research or trades automatically.
8. **Evaluation:** unit regressions cover forged citations, wrong pages, failed reviewers, missing baselines, revision context, rollback conflicts, recovery, cloud replay, conversation idempotency and stale holdings. The offline annotated synthetic earnings pack tests mechanical scoring with positive and negative controls. Representative real-source packs and expert/model benchmark thresholds remain open.

## Analyst conversations

The thesis/note discussion drawer and Evidence & changes offer analyst selection and saved conversations. Requests return a job immediately; each request ID is idempotent and one reply may run per conversation. Reopen history after closing the panel. Current research is supplied in full up to an explicit 120,000-character limit; the last 20 messages are model context and a conversation supports up to 100 saved messages. Updated research requires a new conversation.

Conversations propose replacement wording, with no tool execution or automatic document mutation. Event regeneration and thesis amendment proposals are the executable edit paths. Direct chat-driven note/review application and arbitrary cross-agent tool delegation remain open. Stopping reply delivery prevents a late result being saved but cannot cancel an already billed provider call. Interrupted replies are visible and can be stopped; they are not silently replayed.

## Evaluation commands

Never use pytest in this repository: its global fixtures truncate the local database.

- `.venv/bin/python -m unittest discover -s tests/unit`
- `npm run test:frontend`
- `.venv/bin/python scripts/evaluate-research-quality.py --self-test`
- For a saved recap: `.venv/bin/python scripts/evaluate-research-quality.py --pack ANNOTATED_PACK.json --candidate RECAP_RESULT.json`

The mechanical evaluation checks annotated text patterns and source quotations. Passing is not proof of correct investment judgment or complete claim entailment. No paid production regeneration is part of automated QA.
