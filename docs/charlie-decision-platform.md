# Decision-platform implementation ledger

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
