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

## Live verification

Backend T14 and the cloud/agent endpoints returned successfully after deployment. An unchanged MDT manual transcript policy was submitted through the browser, acknowledged by the local collector, and reported as applied through the cloud bridge. Its manual frequency, seven-day lookback and transcript-only scope remained unchanged. No new collection or paid research run was launched for this check. Analyst selection and conversation controls were inspected in the browser; automated reply tests use a stub provider.

## T16 actionable research revisions

The chat instruction can now prepare source-backed narrative edits to a selected saved note or investment review. Select 1–10 imported documents, review before/after text and source checks, then apply selected edits. Requests are idempotent; changed document content/status and newer review versions block stale proposals.

Notes create new drafts and retain the accepted note, original version and source register. Review inherited charts before publication; DOCX is not reused from the old note. Investment reviews create new immutable versions and render markdown, HTML and PDF from the updated structured state. Existing full-review evidence/readiness is invalidated rather than incorrectly carried forward. Numeric investment-model inputs, personal ratings and convictions are not model-editable in this narrative revision path.

This closes the initial chat-to-note/review narrative edit gap. Numeric model updates, broader restoration UI, multi-agent delegation and the other remaining workstreams still need implementation.

## T17 coverage controls and numerical reconciliation

Collection now supports an explicit batch of ticker policies using the selected cadence, source types and destination. All local destinations validate before the group commits; duplicate commands return their original receipt. Mobile retry/cancel controls target an exact ticker/request. Cancellation prevents further managed export/handoff steps; an already-running browser action may finish. Coverage gaps are shown against saved company coverage and the Mac's available ticker folders. No new schedules have been enabled without the user's ticker/cadence choices.

New recap audits can include up to eight source-linked numerical comparisons. Deterministic checks validate exact numeric tokens, declared currency scales, fiscal periods, measurement bases, consensus labels and arithmetic, including percentage points versus basis points. Unsupported comparisons remain flagged without a calculated result. These selected checks do not establish comprehensive numerical accuracy or prove the model's semantic interpretation.

Validation: 162 safe backend unit tests and 20 frontend tests passed; production build passed. Tests use synthetic/stubbed inputs and do not incur paid research generation.

## T18 large native PDFs and research history

Native-PDF recap synthesis now splits long originals into bounded page segments and combines them into conservative page/byte/text batches. Every page is preserved in order; prompts map segment pages back to the original filename and page numbers. A single page exceeding the byte budget fails before synthesis with an actionable error. The original evidence snapshot remains based on unchanged original bytes. Checkpoint algorithm version changes prevent reuse across incompatible batch layouts. The UI shows matching batches recovered on a completed recap. OCR is still required for scanned pages sent to text-only models; no OCR engine is installed on this Mac.

Research desk now has Research history: a read-only chronology of up to 100 latest imports, notes, reviews, analyst activities (including approved records), research/control jobs and Mac-reported browser requests. Filter by ticker and record type. Exact revision parent/proposal IDs are shown when stored. Creation timestamps and current statuses are distinct from a full transition audit; shared tickers never imply causal links. Source dates are ingestion dates, and collection snapshot age is explicit.

Remaining: approved coverage schedules/full-universe operational validation, OCR and automatic interrupted-job recovery, comprehensive event linkage and version restoration, numerical model edits, representative real-source/expert benchmarks, and bounded cross-agent task delegation. New controls and selected numerical checks do not establish unattended reliability or institutional-grade accuracy by themselves.
