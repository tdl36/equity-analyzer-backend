# Next phase: complete a research decision and make the next run remember it

Planning update, September 11, 2026. Based on the user's pasted AI Buildout at Work
conversation and its linked International Demo - Value Analyst Platform Review,
cross-checked against company_memory.py, meeting_memory.py, meeting_commands.py,
research_decisions.py and the decision-platform implementation ledger. This review
did not independently inspect the original demo images or test production behavior.
The demo assessment establishes demonstrated concepts, not reusable APIs or reliability.

## Product objective

Original rationale → new evidence → affected assumption → explicit calculation →
analyst review → management question / PM brief → saved decision → informed next run.

Keep existing collection, generation, review, source restrictions, framework and
version history. This phase connects them rather than replacing them. Reliability
defects in current user workflows remain priority interrupts.

## Baseline

- Shared memory includes current investment case, selected legacy thesis fields,
  investor framework and at most 20 decisions. It does not retrieve original
  documents, prior meetings, complete case history or portfolio context.
- Meeting orchestration separately retrieves some past questions; this is not
  general historical recall across chat, recaps and preparation.
- Case amendment review preserves original passages and immutable changes, but
  does not propagate numeric assumptions into an operating model.
- EPS × P/E scenarios are deterministic sensitivities, not an integrated model.
- Decision reconsideration conditions are recorded, not automatically monitored.
- User reports saving the personal starter framework. That is not team adoption.
- The latest question-quality policy is committed at 702ebf7; deployment and
  real-source output quality must be verified separately.

## Ordered delivery gates

### 1. Relevant recall and issue-specific review

Retrieve dated decisions, actual meeting answers and permitted original passages
for the issue being investigated. Preserve supersession chains and point-in-time
cutoffs; show retrieval receipts, source identity, selected history and omissions.
Do not solve recall by concatenating every document or silently truncating inputs.

Use stable assumption/issue IDs. Record analyst dispositions (view unchanged,
review needed, accepted change, unresolved) with rationale, exact evidence identity,
actor, timestamp and next test. AI assessments remain proposals. Record review
without overwriting the original management statement or analyst view.

Suppress identical reviewed evidence only within the relevant issue and review
scope. Different exports of the same disclosure are not independent corroboration.
New contrary evidence, changed inputs or an expired review condition can reopen an
issue. A dismissal never silently changes the general investor framework.

Gate: a fresh run retrieves an older relevant decision outside the latest 20,
does not mistake a planned question for an answer, explains a previously reviewed
event and reopens the issue when genuinely different evidence is introduced.

### 2. One source-to-model bridge

Start with a single explicit driver and transparent calculation, not arbitrary
spreadsheet automation. Capture source passage, reported observation, analyst
assumption, units, currency, fiscal period, reported/adjusted basis, baseline model
version and valuation basis. Calculate before/after estimates deterministically.
Review proposed changes before applying; reject stale baselines and incompatible
units/periods. Preserve provenance and immutable versions through restoration.

Gate: hand calculation agrees; incompatible inputs fail visibly; changing a saved
baseline invalidates a pending proposal. Label scope as a one-driver bridge.
An actual user-selected model and verified inputs are required for live acceptance.

### 3. Complete the meeting event

Link questions to existing assumptions where appropriate without forcing every
question into the current thesis. Keep actual answers, analyst interpretation and
AI assessment separate. Propose resolved/partly resolved/unresolved status with an
explanation, follow-up owner and due date; analyst reviews the disposition.
Create reviewable case/model proposals and a PM brief from the same record.

Gate: question → answer → reviewed implication → follow-up → next meeting works,
including ambiguity, conflicting answers, failed generation and retry without
duplicate writes. Saved/copy/email versions retain topic and source identity.
No external delivery is performed as a test without explicit authorization.

### 4. Underweight and value-realization pilot

Capture dated mandate, benchmark and holdings context rather than inferring an
underweight from an absent position. Distinguish valuation, quality, recovery,
portfolio constraints and incomplete research as nonownership reasons. Monitor
explicit reconsideration conditions and catalyst milestones with staleness shown.
Rank investigation priorities transparently; an improved business or rising price
is not automatically a buy. Track delays separately from broken assumptions.

Gate: one user-selected case completes the whole loop using verified portfolio
inputs and current valuation; historical replay uses only then-available evidence.
ABT can validate meeting/source behavior but is not presumed an underweight.

### 5. Shared research and decision learning

Add explicit cross-company driver links with geography, segment, period and lag.
Route proposed implications for analyst review and preserve attributed disagreement.
Personal framework, proposed team framework and sector methods remain separately
versioned; do not silently promote personal preferences to team policy.
Multi-user access and source entitlements must precede actual team distribution.

Compare dated expectations with operating outcomes, estimate changes and price
behavior across successes, failures and passed ideas. Framework learning stays a
reviewable proposal supported by multiple examples and counterexamples.

## Interface

- Today: changed evidence, affected assumption, why it matters, owner, next action.
- Company: current case, model drivers, both sides of evidence, meetings, history.
- Assignments: collection, analysis, review and delivery with exact output links.

Use one research-issue detail view for the evidence, calculation, review and next
action rather than adding another top-level dashboard. Preserve legacy deep links.

## Acceptance and limits

Use docs/meeting-question-source-benchmark.md for the ABT premise cases. Add a
balanced real-source replay including unchanged evidence, contradictory evidence,
missing sources, stale assumptions and interrupted jobs. Track expert-rated
premise accuracy, useful follow-ups, missed issues, duplicate alerts, elapsed time,
cost and saved-output identity. Synthetic test counts do not certify research.

Use safe unittest and disposable database fixtures; never run the destructive
pytest fixtures. Do not start paid research solely for synthetic testing.
Mac/Chrome/AlphaSense authentication remain collection dependencies. This plan
does not establish full unattended operation or authorize portfolio transactions.

Financial-model integration has highest analytical value; recall and scoped review
come first because they make its inputs and subsequent decisions dependable.
Existing roadmap items remain tracked, but these gates govern the next connected
research phase. No new application behavior is implemented by this planning note.
