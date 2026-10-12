# Versioned one-pager editorial handoff

Implementation contract and dated implementation history, October 12, 2026.
Through T216, existing scalar corrections, explicit pending editions, investor
accept/reject decisions, version-specific rendering/export and private resumable
handoff are implemented. New rows and multi-source corrections remain planned.
Earlier checkpoint sections describe their state at that checkpoint, not current
availability. The contract never authorizes the assistant to approve research;
matching passages and saving a proposal do not establish investment quality.

## Why a separate revision is required

The ABT and UNH audit found missing or unmatched claims and incomplete scenario
rows. Resizing the page, changing renderer eligibility or replaying a paid report
cannot establish evidence. The investor must be able to inspect a proposed
replacement against its originals and retain both the original and proposed
versions. A proposal must not update `stock_analysis_runs.state.report`, existing
citation reviews, the assignment's original artifact, or the accepted thesis.

Current integration points inspected:

- `company_research.public_run` returns report/citation state and source metadata,
  excluding extracted source text and private response bodies. The existing source
  download route verifies original bytes against the frozen report hash.
- `research_assignments.artifact` renders the completed report against the
  assignment's saved `visualFundamentals`; `render=original` retains its export.
- `research_visual_view.current_view` checks assignment/report/ticker identity.
  Both entry points must select an explicit editorial version in the new feature;
  neither may silently switch to a pending or most recently created proposal.
- `src/stock-analysis.jsx` already offers source inspection, current/original
  visual export, and a separate thesis handoff. Its current infographic attestation
  does not approve replacement wording or establish a source match.
- `research_meeting_handoff` demonstrates transactional idempotent handoff and
  stale-input fingerprints. Reuse that transaction pattern, not its data model.

## Bounded first deliverable

A user selects a completed report and its specific assignment, prepares a local
source-checked editorial proposal, uploads it to a pending review, and previews
it alongside the original. The investor can explicitly accept or reject the
one-pager revision. Accepting this editorial revision does not apply a thesis,
accept a note, approve a recap, refresh sources, or call a model. A subsequent
thesis proposal uses the existing separate approval flow.

The initial schema allows only replacement strings at existing scalar report
paths, plus explicit new entries in the thesis and scenario sections. Unknown
paths, object-wide patches, removals, duplicate paths and arbitrary JSON Patch
operations are rejected. New scenario rows require all core inputs from that
same case; no merging of fragments from different cases. Set concrete limits
before implementation: 20 changed claims, 2,000 characters per replacement,
5 evidence references per claim, 3 new scenario rows. Longer material belongs
in the full report rather than being silently truncated.

Each proposed claim includes its path, complete proposed wording, factual basis
(fact, guidance, estimate or interpretation), rationale, and evidence references.
References bind frozen source ID, filename, original SHA-256, page/locator and
an exact excerpt. A locally prepared excerpt is evidence to inspect, not proof
that the original was read or that its meaning supports the claim. Interpretation
must identify the source-supported premises and remain labeled interpretation.

Do not accept client-supplied `review=supported`, `passageMatched`, renderer flags
or ownership claims. Server-side source matching must use the same frozen source
content and hashes; original availability/eligibility must be rechecked. Missing
originals, unsupported extraction, changed hashes or restricted usage produce a
specific blocked review state. A public supplement cannot silently become part
of the frozen licensed source pack. Additional sources require a separate source
revision and renewed review.

## Identity and persistence

Introduce a dedicated `onepager_editorial_revisions` table; do not use the paid
job queue. Store UUID, parent revision, assignment/report/ticker identity,
original report-and-citation digest, frozen input/source-hash digest, saved
fundamentals digest, canonical proposal digest, proposal, validation findings,
status, and creation/decision timestamps. Store review decisions separately from
machine matching. Pending/rejected proposals never replace the accepted version.

Use a unique submission key scoped to report, parent and canonical proposal
identity. Repeating the exact submission returns the same revision, including
after a lost response. A different payload under the same key is a conflict.
Lock the assignment/version lineage in the same transaction as insertion or
acceptance. Two competing accept requests cannot both advance the same parent.
Changing baseline, source pack or accepted parent requires a new preparation;
do not silently rebase. A failed transaction leaves no accepted pointer change.

Keep the complete original report snapshot immutable. Preview may create an
in-memory derived report with separate provenance, but must never overwrite
original model-review entries or claim that a human decision was a model review.
Before implementing preview, define the renderer's evidence adapter for human-
reviewed claims explicitly. Do not manufacture `supported` citation objects just
to pass `stock_evidence.valid_citations`.

## API and user workflow

Proposed routes, not currently installed:

1. GET assignment editorial preparation: identities, baseline digests, permitted
   paths, current accepted version, readiness findings and frozen source metadata.
   Read-only; no schema writes, worker wakeup, downloads or model calls.
2. POST proposal: validate bounded schema and frozen identity, record a pending
   revision idempotently, return the specific review destination.
3. GET revision: original/proposed wording, source links, validation findings and
   an explicitly labeled pending preview. Original export stays available.
4. POST explicit decision: investor accepts or rejects the specific immutable
   revision after reviewing changes. Require expected parent and baseline digest;
   return conflict on stale evidence. Automation never invokes this operation.
5. GET accepted revision export: labels revision, review state, dated source pack
   and unchanged public-data snapshot. Never select a revision solely by newest
   timestamp. Original/default and pending preview remain separate URLs.

User-facing labels should explain the gap and change in plain language. Internal
paths belong in expandable details. Show a before/after comparison, evidence
links, accounting period/basis, and missing coverage. Do not call a pending
preview an approved brief or imply matching establishes investment accuracy.
Keep the print export Calibri, black text and a readable review/version footer.

## Safe acceptance evidence

Implement the pure schema/digest contract first, then persistence/routes, then
review UI and a private preparation helper. Pending capture may ship separately
only when visibly labeled as pending and unable to change current outputs. Do not
ship an active Apply control before all server checks exist. Use synthetic
fixtures and disposable database data; never pytest or paid generation for QA.

Required tests include same-key replay, changed-key payload conflict, stale
report/citations/source hashes/fundamentals/parent, unknown and restricted source,
forged client review flags, malformed/scoped paths, full-claim preservation,
no cross-scenario borrowing, concurrent submit/accept, transaction rollback,
exact original retention, rejected/pending version isolation, and no provider
or worker calls. Verify both visual entry points and export versions. Inspect
mobile and desktop review controls and actual print output before release.

Run the full safe suites, compile, build, three release markers and complete
Render/Cloudflare/Mac verification under AGENTS.md. Investor-scored real-source
acceptance remains separate. The existing pending ABT thesis proposal is not
adopted by this feature, and the legacy WELL outcome remains unresolved.

## Accepted-render adapter checkpoint — October 12, 09:00 UTC

T202 capture is live; acceptance remains deliberately unavailable. Code inspection
found that accepting a correction cannot safely be implemented by adding a new
accepted status to `stock_evidence.valid_citations`: that function is shared by
assignment outputs, existing report visuals and readiness checks. A global
relaxation would spread editorial acceptance beyond the selected one-pager and
could affect thesis derivation. `research_onepager.render` also has a second
local `review == supported` check, while `research_infographic.Editor` reads
valid citations directly. Both paths need an explicit adapter, not fake model
review fields or a change to the shared original-evidence contract.

The next implementation should use a dedicated accepted edition with these
boundaries:

- A server-loaded, immutable edition binds assignment ID, report fingerprint,
  exact ordered correction IDs and saved fundamentals digest. The POST decision
  must name all those values and recheck original bytes, current usage policy
  and source passages in one transaction. A changed report, source, quote or
  selected correction is a conflict, never a silent rebase.
- Store edition decisions and provenance separately from T202 pending records.
  Keep T202 submissions immutable, including their original wording and meaning-
  review caveats. Reject competing corrections to the same path within an
  edition. A request cannot change ticker, add sources, or select a correction
  belonging to another report. Do not derive acceptance from a client flag.
- Pass a typed editorial selection explicitly to the two one-pager renderers.
  Its claims have `reviewOrigin=investor`, edition/correction identity and the
  retained original evidence provenance. Original report citations remain
  untouched. Render the factual basis and human-review provenance distinctly;
  matching a quote is not an automatic judgment that it supports the wording.
- The ordinary report, original visual and thesis-import path continue using
  `valid_citations(run)` unchanged. Only a URL explicitly naming the accepted
  edition invokes the adapter. A pending preview must be labeled pending and
  cannot become the default export or acquire an accepted edition footer.
- Reuse claim-length limits and same-row scenario assembly. If a reviewed claim
  does not fit, diagnose the omission; acceptance must not truncate a qualification
  or force the claim into the page. Accepted corrections also need updated
  readiness diagnosis using the same explicit edition selection.
- A printed edition footer includes report ID, edition ID, decision time and
  source/public-data vintage. Human acceptance does not certify source accuracy,
  normalize earnings, approve a thesis, or establish investment materiality.

Acceptance tests must check the brief and illustrated renderer separately,
plus unchanged original/assignment/thesis output paths. Use poisoned client
`reviewOrigin`, unrelated correction IDs, same-path conflicts, stale fundamentals,
source restriction changes and concurrent competing decisions as negative cases.
The explicit investor action is tested with synthetic disposable data only;
automation must never accept a live correction to obtain a passing test.

T202 currently captures one existing field with one source. New rows, compound
scenario packages, multi-source support, specific edition navigation and a private
preparation helper remain unimplemented. No investor acceptance is implied by
this checkpoint.

## Pending-edition contract implementation — October 12, 09:30 UTC

`onepager_edition_contract.py` now provides pure baseline and preparation functions
for a bounded ordered selection of persisted T202 corrections. It binds the full
saved financial data separately from the assignment's vintage-only metadata and
rechecks the original proposal, payload digest, source bytes, eligibility and quote.
Same-path selections conflict; changed report, assignment input, snapshot or parent
requires new preparation. The output is pending and contains no fabricated model
or investor review. This module is not wired to an endpoint, renderer or acceptance.
The future persistence caller must load server records, lock the lineage, and rerun
validation inside its transaction. Synthetic tests do not establish concurrent
edition decisions or investor-approved research; those remain future integration.

## Pending selection persistence — October 12, 10:00 UTC

`onepager_editions.py` now exposes GET/POST assignment-specific pending selections.
The server loads the full raw mp_jobs financial snapshot, locks the assignment,
report, corrections and source rows during submission, and stores an immutable
pending record transactionally. Its submission identity supports replay after a
lost response; changed payloads conflict. GET is read-only and marks changed
baselines stale, but does not claim to revalidate current source restrictions or
bytes. No accepted lineage exists yet, so parentId must be null. Acceptance must
add separate decision storage and lineage locking before any Apply control ships.
There is still no edition renderer, review UI or accepted export. Do not use a
pending record as evidence approval or modify valid_citations to accommodate it.

## Pending edition review UI — October 12, 10:30 UTC

The saved assignment's Infographic view now exposes the pending edition selector
and retained before/after comparisons. It loads the specific report and assignment,
shows claim types and frozen evidence, rejects stale or same-path selections and
keeps an unchanged submission's identity while retrying in the mounted screen.
Only 50 recent corrections and 20 recent editions are shown; deep links and older
pagination remain open. Acceptance and revised exports are still unavailable.
No renderer evidence adapter or accepted lineage is implied by this interface.

## Explicit preview evidence adapter — October 12, 11:00 UTC

`onepager_edition_evidence.pending_evidence` revalidates a server-loaded pending
record, isolates a derived report and returns a typed context with a separate
claim map. Changed claims retain pending_editorial provenance and never masquerade
as model-supported reviews. Their original citations remain untouched and cannot
pass the shared original-evidence gate with changed wording. The adapter is not
yet connected to either renderer. Integration must preserve pending labels on
individual claims and the printed page, update readiness using the same explicit
context, and keep original/default/thesis paths unchanged. Accepted decisions,
accepted lineage, API preview and version-specific export remain unimplemented.


## Pending renderer integration — October 12, 11:30 UTC

The brief and illustrated render functions accept explicit typed pending evidence.
They retain original/default behavior, print pending edition/report notices, label
proposed claims and distinguish pending readiness from model-supported evidence.
The adapter checks the original report, nested context contents and render inputs;
it is not an authorization substitute. Future routes must load and revalidate the
saved record and current source eligibility/bytes under transaction locks before
constructing this context. No public/client preview flag enables the adapter.
Synthetic HTML desktop/mobile checks and tests cover this internal integration;
actual PDF, explicit preview route/UI navigation, investor decisions, accepted
lineage and accepted exports remain future work.


## Explicit pending preview access — October 12, 12:00 UTC

GET /api/research/assignments/<aid>/onepager-editions/<eid>/preview now revalidates
and renders only that saved pending edition. The layout is illustrated (default)
or brief. Shared locks retain assignment/edition/report/correction/original state
through rendering; every frozen original, not only corrected-claim sources, is
checked for exact bytes and current usage eligibility. Stale inputs fail without
HTML. Reads do not create schema or work. The review UI offers both pending layouts
in a sandboxed frame and checks the returned identity. Original/default artifact
routes remain unchanged. No accepted decision, default promotion, revised export,
private authoring helper, new rows or actual PDF acceptance is implemented.


## Pending print inspection — October 12, 12:35 UTC

The T208 sparse synthetic pending map and brief were printed using native Chrome
Save as PDF and inspected as rasterized PDFs. Both use one A4 page, embedded
Calibri variants and black text. Claim wording and pending edition/report notices
survive printing without clipping in these fixtures. Saved HTML matched fresh
current-code renderer output. Private evidence paths and hashes are in AI_HANDOFF.
This is not a production export feature or investor acceptance. Dense selections,
real-source quality, accepted decisions/lineage and accepted exports still need
implementation and separate verification. No live approval was performed.


## Decision intent contract — October 12, 13:00 UTC

`onepager_decision_contract.validate_decision` checks a server-loaded pending
edition against an exact accept/reject request and current server parent. It
rebuilds the selection and checks all frozen originals for exact bytes and eligible
usage, including unchanged claims' sources. The result is only a bounded intent;
it has no accepted status, decision timestamp, investor provenance or claim map.
It is not wired into persistence or an endpoint. Six synthetic tests exercise
staleness, identity, payload replay digests and both actions' complete-pack checks.
Future persistence must authorize the investor, resolve durable decision replay
and conflict, lock lineage/evidence and atomically record the decision and pointer.
New editions are complete selections against the original report: parent identity
provides concurrency control, not implicit inheritance. Show any correction removed
from the prior accepted selection explicitly in the future review comparison.
No live approval or accepted-render/export behavior is available from this module.


## Internal transactional decisions — October 12, 13:30 UTC

`onepager_decisions.record_decision` now implements assignment-serialized decision
storage and accepted-head updates. This is internal and has no route or caller.
Server-loaded evidence is locked/rechecked; decisions retain investor identity and
database decision time separately from immutable pending editions/corrections.
Exact replay acknowledges the old decision without fresh approval; changed identity,
actor or payload conflicts. Rejection never advances lineage. Disposable PostgreSQL
tests cover concurrency, rollback after validation and successor pointer updates,
source policy locks and original retention. Production schema is untouched.
Future integration must authorize the investor using trusted server identity,
expose decisions distinctly in list/detail/preview, use the current parent for new
selections, and provide accepted rendering only by explicit version. Existing
pending routes still assume no accepted lineage and must be updated before the
internal helper becomes publicly callable. No investor approval UI is available.


## Investor decisions and version exports — October 12, T212

The review UI now records explicit investor accept/reject decisions through a
password-session-only route. API-key machines and no-auth development cannot
approve. The accepted adapter and printable HTML exports use explicit version
URLs, fresh original policy/byte checks and separate human provenance. Original
research/thesis gates and default artifacts are unchanged. Complete selections
show removed prior corrections before acknowledgement. Prior accepted versions
remain exportable only while their evidence remains valid. Synthetic UI, PDF and
authorization checks are not real-source investor acceptance.

## Private assistant handoff — T216

`onepager_handoff.py prepare --assignment UUID --workspace /private/local/path`
freezes the report, financial inputs and current accepted parent before authoring.
It saves permitted existing scalar paths and frozen source identities, plus an empty
`proposal.json`. The workspace must be outside this repository and cloud/watched
folders. Inspect the registered originals separately; preparation is not review.
Fill `claims` with 1–20 objects containing exactly `path`, `statement`, `basis`,
`reason`, `sourceId`, `excerpt`, `locator`. Use the existing correction limits and
basis vocabulary; the server matches passages and rechecks bytes/usage. The preparation now lists bounded missing scenario inputs with `before: null`: up
to three indexed rows and nine named scenario fields. Select only these supplied
paths; arbitrary report additions remain forbidden. Additions live only in the
explicit pending/accepted edition, never in the original report. A partial case
is not a complete valuation; year, accounting basis, units and assumptions must
still be stated and source-reviewed. A claim may also supply optional
`additionalEvidence`: 1–7 objects containing exactly `sourceId`, `excerpt`,
`locator`, each naming a different original from the frozen pack. Every passage,
original hash and usage restriction is checked through edition review/export.
The browser supports the same bounded additions; older single-source records retain
their original shape and identities.

`onepager_handoff.py submit --workspace /private/local/path` journals identities
before posting pending corrections and a complete pending edition selection.
Partial correction receipts survive failures. Retry the same workspace after a lost
response; changed content requires a new preparation. The selected corrections are
the entire edition selection, not an implicit inheritance of the accepted version.
Inspect removed prior corrections in the investor review UI. Server checks remain
mandatory, and no acceptance, model call or original research mutation is performed.
The result includes the assignment review destination and exact edition ID.
