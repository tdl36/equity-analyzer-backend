## T317 — Meeting-answer submission recovery across reloads

Critical-path recovery increment: before dispatch, meeting-answer proposals now
persist the exact request identity and saved follow-up revision in per-tab session
storage, scoped to API/company/follow-up identity. Reload performs read-only exact-ID
reconciliation. A matching saved receipt restores its exact review link and clears
the unresolved intent; a missing receipt requires explicit retry of the same frozen
request. Newer follow-up revisions never silently replace an unresolved request.
No answer bodies, credentials or automatic retries are stored or introduced.
Malformed/unavailable/nonpersistent storage and mismatched receipts block unsafe
submission. Status-read errors remain separate from submission messages.

Validation: 218 frontend tests, four existing backend meeting-proposal identity
checks, Python compilation and production build pass. Full-app synthetic checks
at 390/1280px cover lost-response/reload recovery without a second POST, unchanged
request identity across a changed follow-up revision, wrong-company receipt refusal,
unreadable/unavailable storage, exact review links and Back/Forward. Browser POSTs
are intercepted fixtures only; no real comparison or investor decision was made.
Production verification is recorded below after release.

Next: inspect the genuine CVS collection-only assignment's exact-byte receipts and
linked processing-proposal readiness once collection completes; processing remains
investor-authorized. Independently inspect recovery of definitively rejected/stale
follow-up submissions without clearing ambiguous paid outcomes. This increment
closes reload loss, not cross-device/session recovery or all eight priorities.
Shared Ready expansion remains after genuine connected-journey proof. The last
verified CVS checkpoint (T316, 18:41 UTC) was running, collecting/verifying originals,
zero verified sources and no saved error; no collection status was supplied.

## T316 — Follow-up proposal links and atomic history navigation

Meeting-answer follow-ups now retrieve their own proposal history independently
of the company's ten recent jobs, preserving each saved follow-up revision and an
exact review link. The read is bounded to 30 with explicit truncation. A separate
company-active-proposal lookup checks all dates and prevents an unrelated active
proposal from being misidentified as this follow-up's result. Empty, failed and
identity-mismatched reads stay distinct; failed refreshes retain saved links while
blocking new submission. Reads never authorize processing or accept research.

The full-app journey exposed a shared routing defect: native hash/popstate events
under the legacy React root committed view/section/work changes separately, adding
mixed-destination history entries. Those restore updates are now batched. A single
Back/Forward restores the originating company review or exact saved proposal,
including after reload. No execution worker or original Pipeline was changed.

Validation: 29 focused backend tests, 1,632 safe backend tests, 215 frontend tests,
Python compilation and production build pass. Full-app synthetic 390/1280px checks
cover follow-up revision/identity, older links, active-company separation, empty vs
error, reload and back/forward with zero API writes. The earlier Activity exact-link
and rejected-mock-decision regression also passes at both widths. Production
verification follows; no real proposal was submitted, dismissed or accepted.

Next critical path: recheck the existing genuine collection-only assignment's saved
stage once, without waking/restarting collection, then inspect exact-byte receipts
and processing-proposal readiness if it completed. Investor-specific processing
approval remains separate. Independent job-recovery gap: meeting-answer proposal
submission identity currently survives only in memory; persist the exact unresolved
intent across reload and reconcile it without dispatching another paid comparison.
Reuse the existing proposal endpoint, work revision and review route. Shared Ready
expansion still follows genuine connected-journey proof; all broader acceptance
gaps remain in the eight-priority ledger.

T316 production verified: Render 5bb7e5e15e67f0cc97e87b75d1b504f53e6d4b9e;
Cloudflare 81ee7b8a-e5d4-44d2-a3c2-383ec453dd8e. Exact hosted JS/service-worker
bytes and HTML marker match; Mac-agent health ok at 104.8 seconds. Three existing
ETN exact-proposal reads and wrong-company refusals passed. The new follow-up query
returned explicit empty history separately from active-company status; authentication,
invalid/ambiguous-filter, original/library refusal checks passed. All live checks
were read-only; the new real follow-up-to-proposal journey is not claimed accepted.

October 15 18:41 UTC focused CVS checkpoint: assignment
6cdf8f71-0e5c-49c4-838a-525c0731a105 remains running, collectionOnly true,
step Collecting and verifying originals, zero verified sources and no saved error.
Collection status was not supplied in this response; do not infer queued or failed.
No processing readiness is established and no collector/coordinator was restarted.
Next independent engineering is durable meeting-answer submission recovery: save
its exact request ID and follow-up ID/revision before POST, reconcile via exact-ID
GET after reload, and retain ambiguous intents without automatically submitting or
retrying. Storage failure must prevent a new submission; a changed follow-up must
not silently replace an unresolved intent. Keep generation/acceptance decisions
explicit and preserve Operations ownership of collection.

## T315 — Preserve rejected review-action messages

Follow-up to the T314 exact evidence-proposal navigation fix: focused read errors
and review-action errors now have separate state. A successful status poll cannot
clear a rejected decision message. Missing/error reads still withhold focused
proposal actions; retrying the read never substitutes another saved proposal.

T314 was deployed to the frontend but its final verification was superseded by this
release. T315 includes that complete navigation fix. Validation: 213 frontend tests,
production build, and full-app 390/1280px synthetic checks, including a deliberately
rejected mock decision followed by a successful status poll. Navigation performed
zero writes; only the isolated mock rejection was exercised. No production decision
or generation was sent. Backend is unchanged from T314's 26 focused tests and
1,628-test broader suite, plus final compact-ID regression. Exact production
verification follows. Next action remains the T314 saved-object acceptance audit;
genuine collection-to-approved-output and investor acceptance are still open.

T315 production verified: Render c8cfee06d6dfe4f895721f4ce456be8a2168c55c;
Cloudflare c2e8e66c-adad-4902-a4f5-5e8fee86e854. Exact hosted JS/service-worker
bytes and HTML marker match. Mac-agent health was ok at 14.3 seconds. Three
existing ETN proposals (one awaiting approval and two failed) each returned only
the requested identity; wrong-company requests returned no rows. Authentication,
malformed-ID and original/library refusal checks passed. All live checks were GETs.
The initial health check briefly returned 502 during deployment; final exact-revision
and endpoint verification succeeded. T314 is included in this verified release.

Concrete next action: MeetingAnswerProposal currently displays linked work-evidence
jobs but sends every review link to the general company evidence screen. Reuse the
new amendment work route for each linked saved proposal, preserving follow-up
revision and explicit no-match/error handling; test reload and back/forward without
creating or accepting research. Do not introduce another proposal/orchestration
layer. Genuine collection-to-approved-output proof, shared Ready expansion,
historical reference semantics and eight-priority acceptance remain outstanding.

## T314 — Exact saved evidence-proposal navigation

Read-only Activity inspection found eleven evidence-amendment rows, including
multiple ETN and DOV proposals. Their links previously discarded job identity;
the destination selected the newest active company proposal even when the clicked
row referred to an older failure. Activity now carries an amendment work identity.
Evidence review retrieves that exact company/stage/thesis-target record, including
records older than the ten-item recent list. Missing, wrong-company, wrong-target,
failed-read and mismatched-identity results never select another proposal.

Focused historical review hides new-comparison authoring and has an explicit
return to company proposals. Browser back/forward preserves the selected identity.
Opening, restoring or retrying a read cannot start research or accept edits. Existing
explicit proposal decisions remain in the existing workflow. No original Pipeline,
automatic-synthesis policy, research content or collection request was changed.

Validation: 26 focused backend tests (including compact saved IDs), 1,628 broader
safe backend tests before the final compact-ID regression, 213 frontend tests,
Python compilation and production build passed. Full-app synthetic 390/1280px
checks cover Activity navigation, older/newer proposals, back/forward, missing
records, outages and identity mismatch, with zero writes/page errors or horizontal
overflow. Production verification follows. This closes a demonstrated review-link
defect, not full Home/company or end-to-end acceptance.

Next: verify the deployed exact-ID read using existing saved records; retain the
genuine collection-to-explicit-processing-to-output gap. Continue saved-object
navigation and unavailable-service acceptance while Operations owns collection.
Shared Ready expansion still follows connected-journey proof. Historical reference
semantics, real recovery/provider evidence and investor acceptance remain open.

## T313 — Recover unsaved processing-proposal choices

The proposal form now retains authoring choices in session storage scoped to API
and collection identity: selected originals/supplements, purpose/event details,
outputs, focus, full-note/PDF choices and meeting settings. Reload/navigation in
the same tab restores them with an explicit notice. Successful save or deliberate
discard clears the local draft. Source and baseline checks still run at save and
approval; restoring does not authorize processing or accept research.

Local authoring is separate from the existing exact pending-submission record.
A lost save response reconciles against the saved proposal without submitting a
second request, then clears authoring. Pending submissions cannot be discarded via
the authoring reset. Malformed/storage-failed drafts are explicit; no credentials,
source bodies, request IDs, approvals or hashes are stored in the authoring record.
This is per-tab recovery, not server autosave or cross-device backup.

Validation: 211 frontend tests and production build pass. Synthetic 390/1280px
full-app checks cover reload restoration, scope, discard, successful-save cleanup,
lost-response reconciliation and explicit approval; all APIs are intercepted and
remote fonts blocked. Backend behavior is unchanged from T312's verified 1,625-test
suite. Production release verification follows; no paid work or live approvals.

Next critical path: inspect the existing source-only assignment's collection and
exact-byte receipt boundary read-only to establish why no completed source-only
parent is yet available. Diagnose a concrete coordinator/import defect if supported
by saved evidence, without duplicating Operations collection or submitting research.
Then observe the genuine connected journey when request-specific authorization is
available. Shared Ready expansion follows that proof. Historical schema and wider
live/investor acceptance remain open; do not certify all eight priorities complete.

T313 production verified: Render 3198f900b0c74bc3ba3b673545cf9ccdd8160977;
Cloudflare d4d9248c-1197-4b92-b899-df4c048dc01e. Exact hosted JS/service-worker
bytes and HTML marker match; Mac-agent health ok at 71.7 seconds. Original/library
authentication and missing-identity refusals pass. No live processing, source
mutation, paid research or investor decisions were performed.

Read-only delivery checkpoint (October 15, 18:09 UTC): CVS source-only assignment
6cdf8f71-0e5c-49c4-838a-525c0731a105 is running at collecting/verifying originals,
linked to collection 9f7ced1d-7b7a-42d1-9747-6e51f0cfff4d, queued with zero sources
and no recorded issue/error. Its four source-category tasks are pending. The fresh
collection snapshot reports 66 queued, one collecting and three attention; this
establishes queue state, not a cause or a stalled-worker diagnosis. No collection,
policy, approval or coordinator mutation was made. Operations owns this queue.

Next action: retain the genuine collection-to-output acceptance gap, and inspect
existing saved Home/company job and output links against read-only records while
collection proceeds; fix demonstrated delivery/navigation defects without starting
research. On genuine source-only completion, check exact-byte receipts and linked
proposal readiness, then await the investor's request-specific processing approval.
Do not repeatedly poll the unchanged queue or expand the shared dashboard before
connected-journey proof. Historical reference semantics and investor acceptance
remain open independently of these releases.

## T312 — Frozen evidence coverage in processing proposals

Saved processing proposals now expose a read-only summary derived solely from
frozen sourceContext and the saved discovery window. It separates selected
collection/supplemental/unknown roles, recorded publication range, unknown dates,
before/within/after-window counts, source types and exact-byte saved-thesis register
references. Missing legacy metadata stays unknown; current library metadata is
never substituted. Existing proposal digests, choices and authorization are unchanged.

The UI distinguishes a requested collection window from observed selected dates,
and explicitly avoids continuous/exhaustive coverage, accepted-thesis incorporation
or freshness claims. No-reference sources are not automatically classified as new
or unused. Older foundational sources remain valid selections for longitudinal
review. This is deterministic metadata accounting, not model-assisted screening.

Validation: 28 focused tests, 1,625 safe backend tests, 208 frontend tests, Python
compilation and production build pass. Synthetic 390/1280px browser checks (all
APIs intercepted; remote fonts blocked) cover saved coverage/reference limits and
explicit approval. Production verification follows. Original Pipeline, active
collections, automatic-synthesis pause and investor approval boundaries are preserved.

Next independent engineering: protect unsaved source/purpose/output choices when
navigating away from the processing-proposal form, with a scoped local draft and
explicit restored-choice notice. Restore only authoring choices, never pending
approval or an execution request; keep the existing exact submission/replay key.
Revalidate sources/baselines at save/approval. Genuine collection-to-approved-output
proof still awaits a completed source-only assignment and investor authorization;
shared Ready expansion remains after that proof. Historical schema and wider
live/investor acceptance gaps remain open; none of the eight priorities is closed.

T312 production verified: Render bff1b81648aed33a7bb117bac61330685631be8e;
Cloudflare 29f53b34-f643-4dd5-b75d-731133d57a4a. Exact hosted JS/service-worker
bytes and HTML marker match; Mac-agent health ok at 2.0 seconds. Original/library
authentication and missing-identity refusals pass. No live processing, source
mutation, paid research or investor decisions were performed.

## T311 — Full-note source readiness before processing approval

New full-note processing proposals now run local source planning at save. They
freeze source representation hashes, batch count and extraction warnings alongside
the T310 settings. Exact original IDs/bytes remain bound. Approval rechecks the
representation, the queued assignment checks it before starting shared research,
and the note adapter checks it again before enqueueing. Changed extraction cannot
silently substitute a different representation. Legacy T310 proposals retain their
frozen contract; no retrospective authorization or automatic reprocessing occurs.

The review displays planned originals/batches and per-source limitations. Unreadable
or omitted originals, damaged/encrypted PDFs, unapproved PDF-text conversion and
more than eight generation batches are blocked before saving. Oversized PDFs that
need OCR are explicitly blocked; interactive checks do not launch OCR, token-count
APIs or model calls. Native PDF page planning does not prove every figure is
readable. Provider context/budget validation still occurs at execution. Collection
breadth is unchanged and can exceed the selected generation pack.

Validation: 39 focused tests, 1,620 safe backend tests, 208 frontend tests, Python
compilation and production build pass. Synthetic 390/1280px review checks cover
readiness/warnings (all APIs intercepted, remote fonts blocked). Fixtures cover
representation drift, queue delay, PDF policy/OCR refusal, legacy replay and saved-
note integration. Production verification follows. No paid research or live approvals.

Next: genuine completed collection → explicitly approved processing → saved
outputs proof remains open. While waiting for that evidence, priority 3 independent
engineering is the saved proposal's coverage summary: selected publication range,
unknown dates, collection vs supplemental counts and exact-hash thesis references,
with clear limits (references do not establish incorporation). Reuse frozen source
metadata; no inferred accepted-thesis freshness, model screening or new coordinator.
Shared Ready expansion remains after the connected proof. Eight-priority live and
investor acceptance and historical security reference-schema gaps remain open.

T311 production verified: Render 5691d01d3323ad96dc40b2188b734a83b4056f34;
Cloudflare 56dd5159-eacd-4c11-8d6d-63dc650117e4. Exact hosted JS/service-worker
bytes and HTML marker match; Mac-agent health ok at 4.3 seconds. Original/library
authentication and missing-identity refusals pass. A read-only complete archive
inspection found four assignments, two completed, zero eligible completed source-
only assignments and no remaining history cursor. No live approval, paid research,
source mutation or Operations collection action occurred.

Next-action implementation map: processing_proposal_sources.context already
freezes role, ISO-normalized publishedDate, publisher, sourceKind and exact-hash
referencedBySavedThesis for each selected documentId. Derive a bounded coverage
summary from those frozen values and the saved discovery window, not current live
metadata or thesis.updated_at. Count known/unknown dates and types, collected vs
supplemental originals, historical/outside-window sources and hash references;
show observed date range without calling it continuous/exhaustive coverage. Keep
legacy proposals without sourceContext explicitly unknown. Tests must cover mixed
roles, missing dates, out-of-window history, legacy rows and no false thesis-
incorporation or freshness claims. Existing UI: src/processing-proposals.jsx.

## T310 full-note and meeting choices in processing approval

Completed collection-only assignments can now propose a full versioned Lab note
and/or meeting questions from the explicitly selected pack. Saving freezes the note
model, purpose-specific instructions, PDF-text fallback choice and configuration
hash. Approval uses that saved configuration even if central defaults change.
The event-note prompt cannot become a holistic thesis prompt. Meeting date/type
are frozen too; meeting preparation reuses the existing shared report.

The review shows the extra note budget (up to eight calls, 32,000 output tokens
per call), in addition to the shared research budget. No exact monetary estimate
is claimed. No generation occurs at save; processing and research acceptance stay
separate. Missing integration blocks dispatch and retains the pending proposal.
The original Pipeline, collection workers and paused auto-synthesis are unchanged.
Live collection-to-approved-output proof and shared Ready integration remain open.

T310 production verified: Render 8e19f199b3e003d05a46a4fcd3d16550f31cbff2;
Cloudflare 0145a47d-41c2-4194-9016-633061bf3e28. Hosted JS/service-worker bytes
and HTML marker match. Mac-agent health reports ok (71.7 seconds old). Original
and library authentication/missing-identity refusals pass. No production writes
or paid runs were used for verification.

Next concrete engineering increment: use the existing local-only
pipeline_lab_note_sources.prepare planner when saving a full-note processing
proposal. Freeze representation hash, batch count and warnings in the proposal;
show unreadable/unsupported originals, PDF-text requirements and the eight-batch
bound before approval. Recheck the frozen representation before dispatch and pass
it through assignment_note; never invoke token-count/model APIs for this check.
Currently T310 freezes the representation policy, but actual readable/batch
planning still happens in assignment_note after the shared report. Surface this
remaining readiness gap rather than calling the full end-to-end build complete.
Keep real connected proof pending a completed source-only collection and explicit
investor processing approval. Do not expand the shared dashboard ahead of that
proof, duplicate Operations collection, or run paid research solely for QA.

## T309 supplemental foundational evidence

The processing-approval form now includes a wider stored-original library. Browse
metadata, filter by type/search, load more entries and explicitly select older
foundational/contextual sources. The discovery date window does not exclude history.
Collection and supplemental roles are frozen separately; identical-byte duplicates,
changed originals and excluded sources cannot dispatch. Saved-thesis hash references
are shown as references only, never as proof of complete incorporation or freshness.
Filtering covers loaded entries; unknown dates and inferred types stay explicit.
Saved proposals are durable; collapsing the form also retains unsaved in-screen
choices, but navigation before saving does not create a durable proposal.

The original Pipeline and paused auto-synthesis controls are unchanged. Full-note and meeting choices were subsequently added in T310; live authorized
processing proof and shared Ready integration remain open. Supported stored originals appear here;
iCloud-only/research-storage evidence needs the existing import path first.

## T308 collection-to-processing handoff

After a source-only assignment completes, open its coordinated assignment controls
(or saved assignment) and “Choose sources & prepare processing approval.” Select
originals explicitly, choose longitudinal thesis review or a dated event note, and
save a proposal. This does not generate research. A separate approval starts the
existing coordinator from the exact checked sources and baseline; cancelled/stale
proposals cannot dispatch. Repeated approval returns the same job. Follow its saved
job/output link; research acceptance remains separate. No new browser collection is
created and automatic synthesis remains paused during backfill.

T308 initially used originals from the collection only; T309 adds explicit
supplemental history from the supported stored-original library.
Collection dates are not accepted-thesis coverage. T310 adds full-note/meeting configuration
in this approval form; integration into Ready to process remains open. Existing
Lab note/meeting paths remain available. Synthetic checks establish mechanics, not
live investment-quality acceptance.

## T304 source-selection recovery

Unsaved thesis and note selections survive reload/navigation in the same browser
tab. Newer central revisions require explicit recovery; recovered choices remain
unsaved until Save Lab settings. Storage failure is visible. This does not preserve
choices across devices or guarantee recovery after the browser session ends.

## T297 collection breadth and research selection

New collection controls default to 40 relevant originals and allow a chosen scope
of 1–1000. This is a collection ceiling, not exhaustive coverage or a quota. The
scope is frozen in the request and checked on replay and completion. Older frozen
requests without this field retain their original eight-document scope.
Source-only collection verifies a larger library without model calls. Existing
collector identity/hash reconciliation preserves originals and deduplicates exports;
the Lab retains type/date grouping and separate thesis/note selections.

Lab selected packs and imports no longer have an eight-document ceiling. Up to
1000 references can be selected, subject to existing 15 MB per-original and 60 MB
per-generation-pack limits. Full notes use the existing complete, size-aware batch
planner (up to eight **model calls**, not eight originals), which rejects skipped
sources and preserves per-call recovery. Stock Analysis keeps its 160,000-character
shared context with explicitly recorded excerpt omissions; it is not exhaustive
full-document review of a large library. Collection-only can span multiple worker
wakes and import batches. No broad historical backfill or paid QA run is triggered.

# Research Pipeline Lab

## Product contract — October 14, 2026

The original Research Pipeline remains available with its existing stock cards,
configuration and direct thesis-update workflow. Lab is a separate experimental
workspace. Do not replace or migrate original settings implicitly.

The owner has **no AlphaSense MCP or API access**. Use the existing scheduled
Operations worker and supported signed-in Chrome controls. Do not make API/MCP
entitlement a dependency or add a second collection worker here. Collection needs
the Mac awake/online, agent, Codex/browser control and signed-in AlphaSense; MFA
is performed directly in AlphaSense. An open settings screen is not a source
search, and a successful collection is not exhaustive coverage.

## Current implementation

- Separate centrally saved per-stock thesis and note selections, revision checks
  and save replay. Storage ID / folder identity keeps same-named files distinct.
- Search title/publisher/folder, suggested document-type filters, newest sorting
  and 30/90/365-day filters. Recorded publication dates are distinct from fallback
  added/modified dates. No originals are moved to organize the view.
- Stored browser-collected sources expose recorded import hashes, collection-run
  identity and safe observed AlphaSense links in an expandable receipt. This is
  historical metadata, not a fresh byte check or proof of exhaustive collection.
- Per-type thesis/note guidance explains how to review a source without claiming
  its contents were assessed. Selected-only review and hidden-selection counts
  make the chosen pack easier to inspect. Filters never change selections.
- Stock cards show last reported collection state and verified-through dates.
  Stale Mac snapshots and unavailable status are explicit. Existing collection
  schedules/history/recovery are linked per ticker.
- Check selected originals validates stored bytes, permissions, IDs, nonempty
  selections and pack bounds (up to 1000 originals, 15 MB each, 60 MB total).
- Import selected Research originals copies saved Research-storage selections with
  exact-byte receipts, atomic conflict handling and replay after lost responses.
  Originals remain in place; only that task’s selection changes. Import starts no
  research and does not establish source freshness or licensing permission.
- Explicit iCloud imports use exact folder identities, server-frozen hashes and
  atomic byte receipts. Missing/download-needed originals and conflicts require
  attention; retry/cancel and reload reconciliation preserve the request identity.
  The Mac agent must be online and files downloaded. Import starts no research.
- Generate from saved selection creates an existing coordinated research
  assignment. Its immutable manifest binds saved task/revision, original IDs,
  filenames, sizes and SHA-256 hashes. The originals remain in existing storage;
  this is not an additional byte archive. Changes before dispatch or between
  model stages block execution rather than silently switching the source pack.
- A stable client request identity, server replay checks and ticker concurrency
  prevent duplicate submission. Saved runs retain their original manifest after
  Lab settings change. Existing uncertain-call recovery remains explicit.
- Thesis selections prepare reviewable proposals. Note selections now have a full
  versioned-note path: frozen originals/context/playbook/model, central credentials,
  durable call journal, token preflight, worker leases and explicit recovery.
- Full notes have saved job status, draft revision history, prose editing,
  sources/quality caveats, Word and Markdown exports and exact-revision investor
  acceptance. Accepting a Lab note never changes the live thesis or legacy notes.
  Word text is Calibri/black; tables are actual Word tables. Chart data is exposed
  as reviewable segment tables with source/unit caveats, not certified figures.
- Lab's coordinated assignment controls offer source-only collection, with no
  provider key or research generation required, and optional full-note follow-through
  from the shared collected pack. Opt-in explicitly authorizes up to eight additional
  note calls at the existing 32K cap within the monthly budget. The assignment waits
  for the note; downstream failures remain visible alongside already saved outputs.
- Browser-collected type and evidenced author metadata are carried by future imports.
  Older originals are preserved; unknown metadata remains unknown. Types based on
  filename hints are labeled suggested. No source is moved to organize the display.

### Local note edit recovery

Unsubmitted title/prose changes are retained in the current browser tab's session
and restored only against their original revision. A newer saved revision displays
a separate recovery action; it is never silently overwritten or used as a new
baseline. Successful saves and explicit discard clear local recovery. Storage
failures are visible. This is not server autosave or cross-device backup; save a
draft revision in Charlie for durable retention. Pending submitted edits continue
to use the existing exact-request reconciliation path.

### Direct ticker entry

Use Open a ticker directly to reach a stock's Lab workspace even if it is absent
from the cards. The ticker is saved in the URL for reload/bookmark. Opening performs
only reads: it does not add a portfolio thesis, discover sources or authorize paid
research. Confirm company identity and source/output choices before explicitly
submitting the coordinated assignment. Ticker syntax alone is not identity proof.

### Connected meeting follow-through (T303)

A ticker-specific Lab link opens the existing company review workbench. Saved
meeting follow-ups can explicitly prepare thesis proposals from the exact recorded
answer and saved work revision. The question, meeting, answer hash and analyst
interpretation remain inspectable in proposal review. Recorded notes are not
verified transcripts; only answer text is treated as citable evidence. Changed
answers/case baselines require a fresh reviewed follow-up. No model work starts
from opening the link, and no thesis edits are accepted automatically.

## Remaining acceptance and limits

1. Verify a real user-authorized selection through collection, iCloud preservation,
   exact-byte import, frozen-pack generation and investor review. Synthetic checks
   establish mechanics only; they do not prove coverage, freshness or factual quality.
2. Source-only collection uses the existing Operations browser worker and a frozen
   date window, without changing the recurring policy. Reload inventory after completion
   to review new originals. Opening settings alone does not search AlphaSense.
3. Observed metadata remains incomplete on older sources; no retroactive guesses.
   Task-specific relevance guidance explains source uses but is not content analysis.
   Review contrary evidence, foundational sources and event relevance yourself.
4. Long provider calls renew a lease; interruption surfaces attention. Never replay
   uncertain paid calls. Source-only work authorizes no paid follow-up. Full-note
   generation and optional assignment follow-through require explicit submission.
5. The Mac must be awake/online with the agent, Codex and signed-in Chrome available.
   Authentication/MFA stays in AlphaSense. No API entitlement or duplicate worker.
6. Live acceptance and investment quality remain investor responsibilities. Do not
   describe synthetic fixtures as an accepted unattended real-world research run.

Historical valuation remains at its separate documented checkpoint. Do not
claim all eight priorities or ticker-only automation complete.
