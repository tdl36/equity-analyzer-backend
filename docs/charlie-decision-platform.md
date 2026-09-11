# Decision-platform implementation ledger

## T59: connected research workbench and due-work queue

- Company investment cases now include versioned model sensitivities, recorded-answer
  follow-ups and benchmark/nonownership reviews. Every record identifies a saved
  assumption, owner, due date, interpretation and next action. Reviewed/closed states
  require an explicit outcome. Records appear in fresh company memory, with changed
  case baselines flagged. They do not overwrite cases or execute portfolio changes.
- Single-driver margin bridge computes operating-profit, after-tax income, EPS and
  P/E sensitivity using Decimal. It preserves period/currency/basis, baseline inputs,
  original passage and extraction hash. Negative EPS remains visible while unsuitable
  P/E valuation is unavailable. It is not a full model, cash-flow forecast or spreadsheet
  integration. Passage matching does not certify numerical interpretation.
- Meeting follow-ups preserve the exact selected answer snapshot separately from
  the analyst's resolution assessment and next action. Only dated actual answer
  records qualify. Changed answers and source text fail stale-preview acceptance.
- Underweight reviews retain user-reported mandate/benchmark/weights, as-of date,
  reason, valuation assessment and reconsideration action. Active weight is calculated;
  old holdings inputs are visibly flagged. No holdings or benchmark feed is inferred.
- My work includes a live date-based queue for open work and unsuperseded issue-review
  dates, with direct company-case navigation. It polls while open; it does not send
  external reminders or semantically monitor catalyst conditions.
- Immutable work versions support history inspection, historical drafts and copying
  a PM/research brief. Request replay, stale revision, company identity and current
  case checks protect writes. Model/source/answer snapshots remain separate from
  subsequent evidence. Older decision matching now uses English stemming rather
  than exact word equality; this is not embedding-based semantic retrieval.
- Validation: disposable PostgreSQL covers three work types, hand-calculated bridge,
  source provenance, actual-answer snapshots, replay/conflicts, history, stale cases,
  queue behavior and fresh-memory inclusion. Safe unit tests and frontend checks
  plus desktop/mobile fixture exercise follow-up save and underweight form. No paid
  research, production test writes or outbound messages were used.
- Still open: automatic numeric model propagation after review, full financial-model
  mapping, contextual semantic retrieval, event-condition monitoring/notifications,
  multi-user specialist routing/entitlements, cross-company assumption links and
  systematic outcome benchmarking. This release advances a connected analyst loop;
  it does not certify full unattended automation or investment accuracy.

## T58: recorded answers, source passages and scoped review recognition

- Shared context now retrieves up to eight dated answered/resolved meeting records
  with nonempty response notes, company isolation and exclusion of future meetings.
  These are analyst-recorded notes, not verified verbatim management statements.
- Research chat prioritizes the current question's literal terms before saved case
  terms and freezes the resulting context. It can include up to six partial cached
  meeting-document excerpts with source IDs, extraction SHA-256, exact offsets and
  passage hashes. It does not fetch fresh originals or claim complete coverage.
  Other meeting-generation context excludes these additional source excerpts so
  manually selected source boundaries are retained.
- Recall preview shows answers and passages. Users can attach up to six reviewed
  documents to a named issue. Server checks company ownership and the previewed
  extraction hash before saving. Immutable review records retain source identities.
- Later matching excerpts carry up to five unsuperseded issue-review records for
  that exact saved extraction. Changed text is not considered already reviewed;
  stale-preview acceptance fails. Prompts preserve the prior rationale and scope,
  requiring reassessment for new evidence or changed conditions. No alerts are
  suppressed automatically and no general preferences are learned from dismissal.
- Validation: isolated PostgreSQL verifies actual/planned/future answer handling,
  ticker isolation, exact passages, frozen jobs, issue replay, source-change conflict
  and changed-text review recognition. Focused backend tests cover question-driven
  context routing. Desktop/mobile fixture checks source selection and recall.
  Existing research-history routes are retained. No paid generation was performed.
- Remaining: semantic ranking, full source retrieval/verification beyond cached
  meeting text, source entitlements for multi-user distribution, scheduled review
  reminders and operational alert adjudication. This is a recall integration, not
  full unattended monitoring or a claim of model factual accuracy.

## T57: bounded decision recall and issue reviews

- Shared memory supplements the latest 20 decisions with up to 12 older literal
  matches to the current saved case. SQL ranks across ticker history; returned
  bodies are bounded. Retrieval terms, selected IDs and omissions are retained in
  the snapshot. Existing captured jobs remain unchanged. This is not semantic
  retrieval, full predecessor-chain retrieval or original/meeting-answer retrieval.
- Decision log supports optional named issues, explicit dispositions and review
  dates. Server-assigned issue identity persists through superseding reviews;
  immutable earlier decisions, conflicts and replay checks remain intact. Dates
  are recorded, not scheduled. Dispositions do not alter cases or suppress alerts.
- A read-only recall preview in the decision log exposes the current selected
  decisions and their historical status. New issue controls use theme-aware fields.
- Validation: 41 focused backend tests, 29 frontend tests, production build and
  isolated PostgreSQL checks covering old relevant history, issue identity/replay,
  ticker separation and frozen jobs. Browser fixture at 1200 and 390px verified
  issue controls and recall preview without overflow; mobile screenshot inspected.
  No paid research or production research writes were made for validation.
- Next: query-specific semantic/history retrieval, full issue/evidence identities,
  actual meeting-answer and original-document retrieval, scoped repeat-alert
  adjudication, then one source-to-model bridge. See the connected-phase plan.

User authorized the full proposal on September 10, 2026. This is a phased build;
no entry below certifies complete unattended operation or investment accuracy.

## T48 implemented

- Clinical headlines require medical context and reject mining/drilling matches.
- Named-asset/trial clinical headlines with the same issuer, UTC publication day,
  phase and outcome are held as possible duplicates before another collection is
  queued. Each headline remains a distinct signal with a related signal/command ID.
  This is conservative review triage, not automatic semantic event merging. Unknown
  drug names, cross-day events, regulatory/corporate duplicates and resolution UI
  remain open. Existing requests are not mutated.
- Research desk > Investment cases & portfolio contains user-authored investment
  cases: thesis, variant view, market baseline, change conditions, stable-ID
  assumptions, evidence attribution, supporting/contrary evidence and next tests.
  Source references are manual and unverified. Every save creates an immutable
  database revision, with cross-device optimistic conflicts and idempotent receipts.
  This does not overwrite or automatically import an existing thesis.
- Optional bear/base/bull EPS x P/E calculations use Decimal and explicitly entered
  price date, currency and forecast basis. They calculate price return, not total
  return, and reject nonpositive EPS/multiples and invalid numeric inputs. This is
  a sensitivity worksheet, not financial-model integration or a recommendation.
- Newly created improved-note comparisons can recover stale queued/running work
  after restart with a server research key. A 90-second startup grace and 30-second
  sweep, three-minute stale threshold, per-process slots and duplicate exclusion,
  database session ownership and worker tokens protect execution. Recovery stops
  after two attempts; completed, failed and historical non-opted-in jobs are skipped.
  Existing checkpoints are reused. Lost uncheckpointed provider responses can still
  require repeated computation. Historical interrupted comparisons need explicit retry.
- App, worker and service-worker release identifiers are synchronized at T48.

## Verification

384 safe backend tests, 29 frontend tests, build and Python compilation passed.
After recovery changes, targeted summary tests and disposable LOCAL PostgreSQL
checks verify checkpoint recovery, live-owner exclusion, limits and legacy/failed
skip behavior. Investment-case SQL checks cover replay, concurrent edits and history.
Browser fixture verified loading a company, editing an assumption and saving a
revision. Fixture styles are simplified; full mobile/integrated visual acceptance
remains open. No provider calls, test research edits or emails were sent.

## Remaining sequence

1. Common assignment status/delivery contracts across all job families; global
   cancellation and safe retry semantics; cross-day canonical event resolution,
   source aliases and configured event follow-up windows.
2. Connect investment-case revisions to source-verified proposals with immutable
   assumption/evidence links and human-reviewed before/after changes. Preserve the
   management record separately from interpretations. Add case restoration/export.
3. Versioned financial-driver model and spreadsheet mappings: fiscal periods,
   reported/adjusted bases, units, assumption bridges and deterministic scenarios.
   Do not label the initial EPS worksheet a complete model bridge.
4. Portfolio exposure and sensitivity-aware priorities using user-verified holdings;
   explicit staleness, benchmarks, catalyst calendars and reasons for ranking.
5. Complete meeting-to-answer-to-follow-up-to-case/model proposal linkage, with
   saved-source quick preparation and refreshed versions when sourcing completes.
6. Bounded specialists (sector/accounting/valuation/clinical/challenger/coordinator)
   with source contracts, dependency tracking, budgets, stop conditions and distinct
   evidence. Existing editorial passes are not independent-source verification.
7. Cross-company idea discovery and supplier/customer implications linked to
   explicit evidence and dated expectations; watch candidates rather than trades.
8. Management commitments and decision journals; outcome reviews with attribution,
   contextual caveats and visible/editable learned preferences.
9. Consolidate navigation around Today, Companies, Portfolio, Assignments, Library
   with contextual Ask Charlie and guided actions; preserve all legacy deep links.
10. Real-source, expert-graded acceptance suite and operational soak: factual/number
    accuracy, attribution, missing evidence, latency, cost, recovery, duplicate
    prevention and verified delivery. Passing synthetic tests is not certification.

No automatic thesis acceptance, portfolio changes or external communications are
introduced. Current collection continues to depend on Mac/Codex/Chrome availability
and direct AlphaSense authentication.

## T49: reviewed evidence links and delivery clarity

- Investment cases can accept an existing source-supported amendment into a chosen
  assumption's support, contrary evidence or next-test field. The UI previews the
  old/new wording and original excerpt. The server checks ticker identity, proposal
  availability, passage-match and model-review flags, plus the current case revision.
  Source identity/extraction hash, quotation, rationale, original and replacement
  wording are retained in that immutable case revision. Later manual edits mark the
  displayed link historical. This records provenance at proposal generation; it does
  not re-fetch sources or certify interpretation/currentness. Case-specific automatic
  proposal generation and numeric model propagation remain open.
- Restoring a historical case creates a new revision, preserving later history.
  Download exports the saved case, scenarios and evidence metadata as JSON. Cross-device
  conflicts and request replay protection apply to restoration and evidence acceptance.
- Inspect assignment now distinguishes source collection, saved analyst recap, saved
  meeting questions and requested thesis proposals. Dispatch alone is never a completed
  output. Existing reports remain accessible when a regeneration fails, with attention
  surfaced. Results use exact command/topic links; no same-ticker inference.
- Verification: 389 safe backend tests passed before a further delivery regression
  test was added; targeted tests and the production build are rerun for release.
  Disposable local PostgreSQL verifies restore/replay, immutable evidence, rejection
  of forged metadata, dismissed proposals and stale revisions. Browser fixture checks
  source selection, before/after acceptance, restoration and assignment status.
  These checks use synthetic data, not paid generation or live research writes.

The remaining sequence above still applies. This release advances parts of steps
1 and 2; it does not provide global cancellation, automatic assumption matching,
full model integration or unattended end-to-end certification.

## T50: assumption-specific proposal generation

- Investment cases now offer a saved-document selector and optional instructions.
  Generation freezes the case revision and selected original hashes, then runs the
  existing background amendment engine with a case-specific prompt. Up to ten
  originals are supported; oversized inputs fail explicitly, never silently clip.
- Stable assumption IDs constrain proposals to support, contrary evidence and
  next-test fields. Claims, attribution basis, model inputs and the legacy thesis
  are not automatically edited. Passage matching and a model reviewer gate acceptance;
  failed checks remain visible. Empty change lists are valid outcomes.
- Acceptance verifies unchanged core case context and unchanged target wording.
  Disjoint accepted fields can be reviewed separately. Restored, edited or retargeted
  assumptions cannot silently receive stale changes. Provenance remains in revisions.
- The UI polls saved jobs, exposes checkpoint resume and closes proposals without
  removing accepted case revisions. Case proposals are separated from legacy thesis
  queues and cannot use the legacy apply/revert endpoint. Existing running-job
  recovery now checks the correct case baseline; queued-before-worker-start recovery
  remains a broader operational gap. Recovery still depends on the existing worker
  invoking the recovery endpoint.
- Validation: 394 safe backend tests, existing frontend tests, build, browser fixture
  and disposable local PostgreSQL. The SQL fixture exercises a reviewer failure,
  draft reuse on resume, original-source provenance, legacy isolation, no automatic
  application, and idempotent acceptance. Canonical case-context serialization fixes
  checkpoint drift after a JSONB round trip. No paid model acceptance run was made.

Remaining: event-triggered case proposals, real-source quality acceptance, numeric
financial-model/portfolio propagation, and the broader sequence above.

## T51: shared company-memory reader, first integration

- A read-only company-memory service assembles the latest saved investment-case
  revision, its evidence attribution, and separately labelled legacy thesis fields.
  Repeatable-read database snapshots and content hashes make the context auditable.
  An absent case is explicit. No research records are migrated or overwritten.
- Newly dispatched analyst recaps and research-chat replies consume this same service.
  Their jobs retain the snapshot. Chat replies show a case-revision/legacy-context
  receipt. Retrieval failures stop submission instead of silently losing context.
  Existing queued jobs and completed replies keep their original context.
- Prompts distinguish saved beliefs from verified facts, surface conflicts, retain
  provenance, and disclose that original sources have not been reverified. Full case
  content is preserved; the former 60,000-character recap baseline omission is removed.
  Model context windows still apply; this is not unlimited model capacity.
- Validation: safe unittest suite (418 executions, including inherited chat regression
  cases), 29 frontend tests, production build, and disposable local PostgreSQL checks
  for latest revision, source attribution, ticker isolation, stable hashes, missing
  case-table fallback and no-store responses. No live model quality claim is made.
- Scope remaining: meeting-prep integration, historical decisions/meeting retrieval,
  source-document retrieval, editable versioned investor framework, belief adjudication
  and repeat-alert suppression, followed by the benchmark-underweight replay. This is
  a shared current-context reader, not the complete canonical company-memory system.

## T52: meeting-prep company context (backend release)

- New manual and Command meeting packs freeze shared company memory before source
  analysis. The job retains the exact snapshot; retries reuse it even if the saved
  investment case changes. Worker ownership and ticker identity are checked before
  retrieval. Older jobs keep their existing context contract.
- Question generation receives the snapshot separately from selected documents.
  Instructions use beliefs as questions to test, prohibit treating them as original
  evidence, preserve source checks and attribution, and retain the chosen meeting
  length and breadth. Document analysis/synthesis remains source-focused.
- Removed the Command preparation path's 40,000-character legacy-thesis clipping;
  both entry paths now use the shared reader. Model context-window limits still
  apply. Completed job results retain the context receipt as well as job input.
- Validation: 426 safe unit-test executions plus a subsequent targeted 20-test
  pipeline run; isolated PostgreSQL verifies frozen context across later revisions
  and ownership rejection. No paid real-source generation was run, so output-quality
  improvements remain to be evaluated with actual packs. No frontend changes.
- Next: explicit dated analyst decisions and historical retrieval, versioned investor
  framework, belief adjudication and alert suppression. A thesis revision is not
  automatically classified as an investment decision or management answer.

## T53: dated analyst decisions

- Investment Case contains a decision journal: decision date, reasoning and explicit
  revisit condition. New entries never change a portfolio or saved thesis. Superseding
  an entry creates a linked record; the original remains immutable. Server revision
  checks and request receipts prevent stale saves and duplicate request replay.
- Shared memory includes the latest 20 decision records with dates and supersession
  status. Older-record omission is explicit. The journal displays the latest 50.
  Unsuperseded records are not certified current; prompts distinguish decisions from
  management answers, verified facts and executed trades. Existing job snapshots
  remain unchanged. Revisit conditions are not yet automatically monitored.
- Validation: 430 safe unit-test executions, 29 frontend tests, production build,
  isolated PostgreSQL tests for journal creation, replay, ticker isolation, stale
  revisions, immutable supersession and memory retrieval. Browser fixture loaded;
  interactive browser acceptance was not completed. No live research writes or paid
  generations were used in validation.
- Next: complete historical pagination/search and contextual decision retrieval;
  editable versioned investor framework; belief adjudication and repeat-alert
  suppression; real-source quality evaluation. Decision records deliberately require
  explicit analyst entry rather than being inferred from thesis edits.

## T54: complete decision-history browsing

- The decision journal now searches decision wording, rationale and revisit conditions,
  filters by decision date, and pages through older matching records using a revision
  cursor. Literal search treats `%` and `_` as text. The latest write revision remains
  independent of filtered/paged results. Stale history responses cannot replace newer
  results in the UI, and draft text survives reloads.
- Superseding entries open their predecessor directly, including records outside the
  loaded page; the predecessor can in turn open its own predecessor. Lookups remain
  ticker scoped. Server supersession checks continue to prevent invalid replacements.
- This is user-facing history retrieval. Searching does not alter an agent's context:
  shared memory still discloses its latest-20-record limit. Contextual model retrieval
  and automatic monitoring remain separate follow-on work.
- Validation: 432 safe backend test executions, 29 frontend tests, production build,
  and isolated PostgreSQL checks with 64 records for no-overlap pagination, literal
  search, dates, empty-result write revision, cross-ticker isolation, predecessor
  lookup and memory omission disclosure. Interactive browser QA is not claimed.

## T55: editable investor framework

- Investment Case now contains a cross-company framework editor with six sections:
  philosophy, business quality, valuation, value realization, evidence discipline,
  and risk/disconfirmation. An optional value-investor starter fills an unsaved draft;
  no preferences are activated until the user saves. Clearing and saving removes
  active preferences. History retains immutable versions; restoration creates another
  version. Revision checks and request receipts protect against stale/duplicate saves.
- Company-memory schema 3 carries the saved framework separately from company
  evidence. Prompts apply methodology without overriding original-source checks,
  attribution, selected documents, current assignment instructions, or meeting breadth.
  Shared-context chat, analyst recap and meeting-question workflows receive it. This
  does not retrofit every legacy generator or authorize actions. Chat receipts record
  the framework revision; meeting recovery retains its captured version.
- Validation: 436 safe backend test executions, 29 frontend tests, build, disposable
  PostgreSQL save/replay/conflict/clear/restore tests and cross-company/frozen-job
  checks. Interactive browser fixture verified starter draft, save v1, clear/save v2,
  inspect history and restore v1 as v3. No production preferences or paid model runs
  were created by validation. This proves workflow mechanics, not research quality.
- Next: real-source framework comparison, broader generator integration, contextual
  historical retrieval, belief adjudication and repeat-alert suppression. Revisit
  conditions remain recorded instructions, not monitored event triggers.

## T56: framework production styling correction

The initial framework browser fixture supplied generic label/input CSS absent from
Charlie, masking unstyled production form controls. Added scoped framework styles
and explicit action/field layout: two columns on desktop, one below 800px,
full-width theme-aware textareas/selects, labelled spacing, focus outlines and
44px buttons. No framework content or backend semantics changed.

Validation now uses the actual built production stylesheet and workspace panel.
Headless Chrome checks and screenshots cover Dusk at 1200/390px and Ink at 1200px;
six readable themed fields, correct mobile column count and no horizontal overflow.
Desktop screenshot inspected visually. This corrects the earlier isolated-fixture
validation gap. Production build completed.

## T60 — Summary bulk export and pooled email (2026-09-11)

Library → Summary → Select now exposes Brief and Meeting Summary alongside existing sections for Word/PDF, with All sections, Korean takeaways and audio transcript options. PDF generation uses the same ordered section registry as pooled email. Email selected sends one message containing all saved sections of each selected entry to the Settings recipient; export toggles only affect exports. Selection order is preserved. Missing IDs or invalid section filters fail explicitly rather than creating partial exports. Email has an in-flight click guard and no automatic retry after ambiguous delivery. Existing saved content is unchanged; no generation or SMTP delivery occurs during tests.

Validation: five isolated unit tests cover real DOCX/PDF extraction, ordered ZIP entries, validation, long transcript preservation, escaping, and mocked pooled SMTP. Browser fixture exercised section toggles/email action at desktop and 390px without overflow. Frontend production build passes.

## T61 — Roundtable workflow foundation

- Investment case workspace now separates Current thesis, Evidence & proposals,
  Decisions & underweights, and Evolution while retaining unsaved editor state.
- Read-only lifecycle endpoint reconstructs case/work/decision snapshots by actual
  recorded timestamp and ticker. A timezone-aware cutoff excludes later saves.
  Each category exposes its 100-row bound. It does not backdate knowledge using a
  document's publication date or reconstruct historical market data.
- Evolution displays revision rail, stable-ID assumption matrix, field comparisons,
  accepted original excerpts, underweight snapshots and dated work/decision history.
  Revised wording is not labeled as stronger investment conviction. Independent
  decisions are not falsely attributed as the cause of a nearby case revision.
- Underweight reviews now preserve typed reconsideration conditions, user-assessed
  status and supporting interpretation/source reference, plus maintain/investigate/
  propose-change decisions. Assessed conditions require evidence explanations.
  Completed structured reviews require a decision; portfolio weights cannot exceed
  benchmark weight in this workflow. Old records remain readable/editable.
- A proposed evidence change or saved underweight can open durable research chat
  with its snapshot and a prefilled challenge. Opening does not submit paid work.
  Legacy edit controls are hidden for these discussions; accepting a source proposal
  remains in the source workflow with existing stale-case protection. Chat still
  receives current shared memory and its last 20 messages, so this is not a
  historically frozen model replay. Original sources are not fetched by chat.
- Validation: 31 frontend tests; 3 new lifecycle/condition tests; 10 conversation
  and 2 case-proposal unit tests; disposable local PostgreSQL lifecycle cutoff,
  isolation/bounds checks plus existing workbench and source-proposal acceptance.
  No production data edits, model calls, portfolio actions or outbound email in QA.

Remaining roundtable milestones: source-arrival dispatch into materiality review,
condition matching against new verified evidence, explicit pillar-strength analyst
assessments, persistent debate-to-decision links, proposal-level no-change/reject
rationale, portfolio-wide underweight monitor, and a timed real-source demo replay.
This increment is not full autonomous event monitoring or a merged lifecycle for
legacy thesis documents and the structured investment case.
- Browser fixture passed desktop/390px history selection, field comparisons,
  condition entry and no page overflow; Dusk rendering inspected. No live investment
  conclusions were generated solely for testing.

## T62 — Prospective evidence monitoring and review continuity

- Per-company opt-in monitor baselines existing uploaded originals, detects changed
  content (including replacements), deduplicates renamed identical payloads and
  submits up to ten new sources against the saved investment case. Existing agent
  heartbeats wake a backend scanner, throttled to once a minute per process and up
  to three enabled companies per scan. This is not instantaneous web/AlphaSense
  discovery: collection/import must occur first. Legacy automatic intake remains
  separate and targets legacy thesis proposals.
- Stable reservations survive lost receipts. Source hashes are checked before job
  submission and again before analysis. A source is consumed only after a confirmed
  proposal receipt. Existing unresolved proposals block later batches. Failed jobs
  use existing checkpoint recovery; unresolved reservations expose a manual recheck.
  Pause prevents new scans; already reserved/in-flight work may finish. Initial
  enrollment does not backfill; files arriving while paused remain eligible.
- Current underweight records/conditions are frozen in each case comparison. AI can
  propose condition assessments with exact source excerpts; both thesis changes
  and condition suggestions pass through model review. Unknown IDs, unsupported
  states, repeated IDs and failed excerpts remain rejected or visibly unverified.
  Suggestions do not mutate work records. Resumption checks the captured underweight
  baseline as well as the case. Reviewed source interpretation is still human work.
- Proposal closure can retain a no-change/rejected/changes-reviewed conclusion and
  rationale. First saved conclusion is immutable on replay. Evolution displays
  conclusions using their recorded timestamp, not the earlier proposal creation date.
- Portfolio underweight overview links latest review records to company workspaces.
  It discloses manually recorded weights, condition status and 200-record scope.
- New-case setup can copy saved thesis summary/pillars and textual risks/signposts
  into an unsaved review draft. It retains a source hash/reference, labels prior
  research as interpretation, and fails visibly rather than truncating oversized
  fields. Original thesis documents remain unchanged; valuation/variant view need
  separate review. Saving the draft establishes the structured baseline.
- Tests: isolated PostgreSQL covers enrollment, duplicate/replacement handling,
  receipt retry, pause/resume and stale settings; case proposal acceptance covers
  source mutation before submission, condition identity/review, checkpoint resume,
  explicit no-auto-apply and immutable closure. Baseline and condition unit tests
  pass, as do 31 frontend tests. Desktop/390px fixture verifies monitor settings and
  underweight navigation. No real model calls or production research writes in QA.

Remaining: timed real-source acceptance/demo, reviewer-approved transfer of AI
condition suggestions into work records, direct debate-to-decision identities,
publication/arrival latency receipts, unified legacy/structured-thesis lifecycle,
more complete queue ownership recovery, and source/condition-specific notifications.
Monitoring requires enabling it after a saved case exists; no companies were
silently enrolled or historical outputs regenerated by this release.

## T63 — Evidence workspace usability repair

Replaced the inline checkbox/filename wall and unstyled forms with three explicit
steps: choose documents, optional focus, review/decide. Sources have separate full
filename rows, search, selection count, selected-only filter, review/remove list,
and visible ten-source limit. Input, checkbox and action styling is scoped to this
workspace across themes. Monitoring moves to collapsed optional settings; completed
proposal reviews are collapsed and closure controls are separated from acceptance.
In-progress work has a route-safe scroll action rather than changing the app hash.
Loading/errors and missing selected sources prevent ambiguous submissions.

Browser fixture with 42 long filenames passed desktop and 390px: search, persistent
selection across filtering, ten-document limit, exact mocked submission, duplicate
submission guard and no page overflow. Dusk screenshot reviewed. No live paid
research was generated and no research content was changed by this UI repair.
