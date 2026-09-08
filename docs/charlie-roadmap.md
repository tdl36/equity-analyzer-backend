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

## T19 user-directed coverage and prospective catalyst watch

Per explicit user direction, weekly default policies have been saved for the 70 tickers in the union of saved analyses and analyst coverage. Missing STOCKS/CATALYSTS ticker directories were created; originals were not moved. A private SQLite backup precedes this change. Frequencies support manual, daily, biweekly (14 days), monthly (explicitly 30 days), and custom integer intervals of 1–8760 hours. Policies can be paused, edited or added; explicit manual refreshes can run while a schedule is paused and do not shift its next due time. Save & refresh is one durable local command. An explicit create-folder control allows new ticker destinations.

Prospective catalyst watch screens Finnhub company-news headlines for potential clinical results, regulatory decisions, guidance and corporate transactions. The existing Mac bridge requests one lookup per minute, cycling watched tickers (~70 minutes per full 70-ticker sweep); detection requires valid Finnhub news access. A headline rule is not a verified materiality decision and company-news coverage is incomplete. Identical normalized ticker/headline/publication-day signals deduplicate across URLs; different headlines about the same event can still generate separate signals. Future previews and old/pre-enable news are excluded.

With automatic mode enabled, up to the configured daily limit (default 10, editable 1–30) produce durable event-specific collection commands, provided an enabled ticker policy and covering analyst exist. Those commands request press releases, broker reports and transcripts into a named CATALYSTS event folder, then use existing verified handoff/analyst recap dispatch. Source restrictions remain enforced. Event runs do not advance the normal STOCKS refresh cursor or rewrite its policy. Excess/unrouted signals remain visible for manual research, without automatic backfill. Successful scan/queue status does not prove browser export, recap completion or a thesis edit. Numeric thesis amendments and final approval remain separate review steps.

The new press-release browser filter and event-to-paid-recap path need validation against the next qualifying live event; regression tests stub news and dispatch and do not claim such a live event occurred. Real-source quality benchmarks, OCR, general job resumption, full causal event linking, broader restoration and bounded cross-agent delegation remain open.

## T20 Command Charlie and reusable favorites

Research desk → Command Charlie supports a saved instruction, explicit ticker, New York event date and 1–90 day collection window. Four editable cloud favorites cover filing/reaction, earnings, clinical catalysts and weekly refresh; `{ticker}` and `{date}` substitute per assignment. Optimistic revisions prevent silent multi-device favorite overwrite. Idempotent task IDs preserve uncertain submissions, and task history separates Mac acknowledgement, browser collection and exact event-folder analyst recaps.

Managed commands retain weekly policies, collect AlphaSense press releases/broker reports/transcripts and SEC recent 8-K/8-K/A primary documents plus filename-identified Exhibit 99 attachments into a unique CATALYSTS event folder. A pending marker holds synthesis until verified collection. SEC source hashes, URLs and download checkpoints survive interruption; errors and unqueried historical archive windows are explicit failures. Public-source completion and production manifest visibility are required before handoff. Assignment instructions persist on the analyst activity even in automatic mode. Empty verified searches do not generate a recap. Proposed investment implications are in the recap; structured automatic thesis amendment chaining and wider internet adapters remain open.

Validation: 193 safe backend tests, 20 frontend tests and production build passed. A live SEC UNH lookup for September 1–7 completed with no matching filing documents; no AlphaSense export or paid synthesis was launched for QA. This does not validate a live command end to end. Full job recovery, OCR, broader restoration, representative source benchmarks and bounded analyst delegation remain next.

## T21 Durable recap receipts and bounded worker recovery

The recap outbox now has a matching backend receipt route. A complete result and its still-current Inbox activity update commit together; a different result, superseded claim or cancelled job is rejected while the local copy remains. A matching job ID and content hash are required before deleting the outbox file. Old timed-out activity deliveries can recover; approved or superseded Inbox records remain unchanged.

Token-aware local synthesis workers claim atomically, renew ten-minute leases and verify ownership before each new provider call. On the returning Mac heartbeat, expired claims are requeued at most twice under the same job ID; available exact-input batch checkpoints resume. Exhaustion becomes an explicit failure. In-flight provider requests cannot be cancelled retroactively and a lost, uncheckpointed response may require repeated computation. Normal source/model errors remain failures rather than blind retries. This covers new Mac recap jobs, not general server-side thesis/chat/note worker resumption. Old queued jobs are no longer failed merely for waiting thirty minutes.

Validation: 208 backend unit tests, 20 frontend tests, build and Python 3.9 import/command tests passed. Tests cover actual outbox-to-blueprint routing, receipt conflicts, activity preservation, competing claims, owner fencing and bounded recovery. No live paid process was deliberately interrupted for QA. Command favorites loaded from the deployed cloud API and were inspected in the browser; a fresh style load corrected the stale development tab's unstyled cards.

## T22 Local OCR and initial real-source controls

Tesseract 5.5.3 and Poppler are available on this Mac. Text extraction now attempts OCR on empty/text-poor image pages, preserving original page numbers and leaving original PDFs unchanged. Private page caches are keyed by original bytes and engine configuration; completed pages survive interruption. Text-only recap inputs stop on empty/low-confidence OCR, and native-PDF evidence audits disclose extraction limitations. Local large-PDF note text extraction uses the same adapter. Source records expose OCR page numbers where text extraction was used. Cloud-only OCR is not installed; unreadable cloud sources still require the Mac or a native-PDF path.

English OCR is bounded to 80 candidate pages and 60 MB per source, with per-process serialization and renderer/OCR timeouts. Engine confidence is not factual accuracy. Text-rich pages with embedded image regions are not exhaustively OCRed; charts, tables, symbols and material figures still require original-page review. Long scans need splitting or native-PDF synthesis.

Validation: 217 safe backend tests; an actual image-only fixture recovered all seven financial numeric tokens with engine confidence 95 and unchanged original bytes, under both development and Mac Python. The three-pack quality suite includes two historical FDA excerpt packs plus the synthetic earnings pack. All positive/negative controls passed; excerpt hashes detect fixture drift. These are authored, selected mechanical controls, not expert-graded or live-model benchmark results. Broader financial packs and blind analyst scoring remain open.

## T23 Historical version restoration

Research history offers a side-by-side restoration preview for saved notes and investment reviews. Restoration creates a new version; notes return as drafts. A historical review regenerates markdown, HTML and PDF with an explicit historical-date banner, preserves its assumptions and source lineage, and requires a fresh quality review. Existing versions, charts and accepted notes remain intact. Source/current content hashes and latest-version identity block stale previews; a durable restoration request receipt prevents duplicate versions on retry. Narrative previews are bounded and disclose truncation.

Validation: 226 safe backend tests and 20 frontend tests; production build passed. Tests cover originals/charts preservation, same-request replay, current/source conflicts, stale identity, historical readiness and historical date/warning across all three export formats. No user research was actually restored for QA. General thesis history beyond the existing amendment rollback, event draft restoration controls and expert quality scoring remain separate gaps.

## T24 Coordinated draft review and saved-thesis context

Command Charlie and favorites can request independent challenge and editorial revision, with up to two extra model passes clearly labeled in the UI. A lead recap is challenged for material reasoning gaps; the editor must address every finding, preserving the complete draft and source/number boundaries. Exact draft-quote matching, conservative new-number checks and excessive-deletion checks reject invalid revisions. Role outputs, editorial decisions, the lead draft and reused stages are retained and shown in Earnings & evidence. Each role has a source/input/prompt-keyed checkpoint; completed roles resume without another model call. A final selected-claim source audit follows editing. This is bounded role delegation through the configured recap model, not unrestricted autonomous agents or independent new-source verification; active recap jobs retain the existing five-job concurrency cap.

Recaps now receive selected saved thesis narrative fields, where available, in addition to any applicable analyst playbook. Oversized/missing baselines are disclosed explicitly. Non-earnings event recaps now include this baseline in their prompt, improving comparison against the prior investment view. Structured thesis application remains a separate decision.

Validation: 235 backend unit tests, 20 frontend tests, production build and Mac Python coordination tests passed. No paid production research or collaboration run was launched solely for QA. T23 restoration preview returned live saved DE note data without restoring anything; backend health was verified at its release revision.

## Trusted public-source supplements (local worker)

Managed clinical/regulatory commands can supplement SEC and AlphaSense with verified FDA HTML pages and ClinicalTrials.gov registry originals. The worker searches bounded primary domains, checks issuer/sponsor relevance and dates, then registers the original through an allowlisted, size-limited adapter. Registry identity and last posted date must match the official API. Originals share the existing CATALYSTS folder, immutable hash receipt, lease fence and production manifest verification before synthesis. Redirects and arbitrary URLs are rejected. Registry changes are explicitly not represented as clinical readouts.

This extends the local collection runbook; no frontend release is required. Discovery/relevance remains worker-reviewed, not a general internet search engine or independently validated clinical analysis. Safe regression coverage checks invalid URLs, dates, study identity, cancellation during retrieval, changed originals and manifest visibility. No paid research or live catalyst job was launched for QA.

## T25 Reviewable command-to-thesis handoff

A completed Command Charlie card opens its own source comparison panel. The backend resolves the exact event topic and ticker, requires the latest linked recap, and verifies its recorded input hashes against imported originals. Missing and changed sources are listed, including transformed text that cannot be byte-matched. The user selects up to 10 verified sources, carries the original assignment into editable instructions, and requests the existing source-checked thesis proposal workflow. Command/activity provenance and selected hashes are saved; a changed recap blocks stale submissions and changed document bytes block model execution. Saved thesis edits still require selective approval.

This does not auto-import iCloud files or automatically launch a paid comparison after each recap. Those remain subsequent orchestration work. Validation includes exact-source matching, wrong ticker and incomplete recap rejection, stale-revision route rejection and persisted source hashes/parent command before worker dispatch.

## Automatic import of managed command originals

Managed command completion imports registered eligible originals into the cloud document store before recap dispatch, with source URL, command/topic provenance and SHA-256 receipts. This removes the manual upload step for newly collected command documents. Restricted originals and unregistered folder files are excluded. Same-name different-content collisions fail visibly and preserve both local originals and existing cloud documents; retries verify existing bytes. Limits are 60 files per command and 60 MB per file. The worker must still be available and collection verified. New snapshots retain original hashes as well as extracted-input hashes. Older transformed snapshots and general iCloud-only documents still need separate verification/import. 258 backend regressions pass; live paid workflow validation remains for a real assignment.

## T27 Proposal stage checkpoints and bounded resume

Thesis proposals retain completed draft/review stage outputs even when later work fails. Resume saved proposal work reuses matching stages under the same proposal ID, with at most two retries. Saved thesis, source bytes, full prompt and configured model identities protect against stale reuse. Dismissed proposals reject late stage writes. This is explicit failed-job resumption; interrupted jobs still marked running require the next ownership-recovery increment. In-flight responses lost before persistence may need another billed call. Validation: 265 backend tests, 20 frontend tests and build; no live paid job interrupted for QA.

## T28 Automatic recovery of abandoned running thesis proposals

The Mac heartbeat can recover new running proposals after their PostgreSQL execution connection is gone. A live session lock blocks recovery regardless of runtime; ownership tokens prevent superseded workers from saving late outputs. Recovery is capped at two attempts, reuses matching stage checkpoints, requires a server research key and fails explicitly if the saved thesis changed. The UI displays recovery availability/count. Queued jobs and older proposals are excluded to avoid mistaking waiting work for abandonment. Ordinary failures retain explicit resume. One pooled connection is occupied per running proposal; provider responses lost before checkpointing may be billed again. 272 backend/20 frontend tests and build pass; no paid live job interrupted solely for QA.

## Real SEC financial regression controls

The offline evaluator now includes frozen Microsoft FY2024/FY2025 annual revenue and operating income from one identified SEC 10-K accession. Structured assertions are checked using decimal arithmetic, unit scaling, exact fiscal periods and reported/calculated labels. Nine negative controls cover wrong units, periods, arithmetic, percentage points, nonfinite values, unsupported guidance, unsupported price targets and invented consensus. Frozen multi-source hashes catch content/identity drift. Four control packs and 275 backend tests pass. These are developer-authored annotations and fixture results, not a live model quality score, general prose verifier or expert investment benchmark.

## T29 Automatic proposal preparation after Command research

An optional task/favorite preference now continues completed Command research into a thesis amendment proposal. The Mac heartbeat verifies the exact completed recap and all imported source hashes, then uses the original assignment and stable proposal ID for an idempotent comparison. Automatic sets must contain 1–10 verified originals with no blocked sources; existing proposals and missing server credentials block visibly. Proposal generation consumes model credits; application remains an explicit user decision. Task history exposes actual proposal state and dispatch blockers. Candidate checks rotate with a maximum of three per heartbeat. Validation: 282 backend tests, 20 frontend tests, build and Mac compile; no paid research run solely for QA.

## T30 Guided meeting / conference workflow

Implemented the guided 1–10-company selector, auto-written assignments, exact command-source-to-Meeting-Prep bridge, per-company progress, stage resumption, protected final-save receipt and registered question-source checks. See `docs/charlie-command-build.md` for bounds. Next: validate an actual authorized meeting assignment, then immediate worker dispatch, cache-first preparation, verified presentation/IR retrieval, per-company conference scheduling and measured quality/latency benchmarks. UI selection alone is not proof of collection or output quality.
