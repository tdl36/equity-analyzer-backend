# Decision-platform implementation ledger

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
