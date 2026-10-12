# Versioned one-pager editorial handoff

Implementation contract, October 12, 2026. This is planned work, not a shipped
feature or authorization to approve research. T201 provides read-only evidence
readiness; it does not repair the saved investment argument.

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
review UI and a private preparation helper as one end-to-end release. Do not
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
