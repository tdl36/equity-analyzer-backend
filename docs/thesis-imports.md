# Subscription-assisted thesis drafts

Release T123 adds **Companies → Import & review thesis**, also available from the
Investment thesis toolbar. The same screen supports an initial thesis for a new
name and an upgrade to a saved thesis. All text in the review workspace is black
Calibri, with light backgrounds and stacked comparisons on narrow screens.

## Investor workflow

1. Enter the ticker and choose **Prepare for ChatGPT**. The downloaded JSON file
   includes the existing detailed thesis (or an empty initial template), a version
   fingerprint and authoring instructions. Attach this file and the selected
   originals in ChatGPT. Source availability and usage terms remain the investor's
   responsibility; Charlie does not upload originals through this workflow.
2. Ask for the completed JSON draft, retaining its schema, ticker, baseline and
   existing item identities. Import that file, not the review PDF or source register.
3. Review the complete before/proposed comparison, sources and limitations. The
   inbox survives reloads and server restarts. Filter by ticker to find prior work.
   Download and revise a draft externally, then reimport it when edits are needed.
4. Check the investor approval box, type the ticker and choose **Approve & save
   thesis**. Only this action changes the live detailed thesis. Dismissal preserves
   the draft and leaves the thesis unchanged.
5. Open the saved thesis to use existing editing, version history, restoration and
   future upgrade workflows. Full/condensed versions, scorecards, one-pagers,
   investment-case models and other reports are separate stores. Review or update
   them explicitly; approval neither regenerates them nor launches paid checks.

Preparation, import, comparison and approval use no model APIs. Normal paid Charlie
research remains available separately. PDF/Word/free-text extraction is not part
of this first integration; the completed structured JSON is required.

## Format and compatibility

Supported schema: `charlie.external-thesis-draft.v1`. The earlier
`charlie.external-thesis-draft.proposed.v1` is accepted for the existing SYK draft.
Top-level fields: `schema`, `ticker`, `companyName`, `baseline`, `analysis`,
`sourceRegister`, `provenance`. Download the live preparation template for an example.

`analysis` contains the native detailed-thesis shape:
- `thesis.summary` and `thesis.pillars` (title, description, optional confidence).
- `signposts` (metric, target, timeframe).
- `threats` (threat, triggerPoints; optional likelihood/impact).
- Text `conclusion` and optional `documentHistory` metadata.

Pillars, signposts and risks accept stable IDs and source reference objects, including
`sourceId`, `filename`, physical `pdfPages`, short `excerpt`, `supportType` and optional
`sha256`. Source register entries have unique `id` values. The importer checks
reference IDs, page ranges when supplied, and agreement between supplied hashes.
It does **not** read originals or establish that a quotation, hash, fact, permission
or author-provided provenance claim is correct. Original source-use notes remain
visible. No source bodies, API credentials or account cookies belong in a package.

Packages are limited to 2 MB, 100 items per thesis section, bounded nesting and 500
entries per general metadata list. Required text, duplicate IDs, mismatched tickers,
company names, unsupported formats, malformed page references and invalid characters
produce actionable errors. Imported approval/history claims are not authoritative.

## Persistence and conflicts

`external_thesis_drafts` stores the normalized package, captured baseline, baseline
hash, canonical package fingerprint, status and approved revision receipt. Identical
uploads return the same receipt, including after approval or dismissal. List responses
show the latest 100 matching drafts; detail/download retains the reviewed package.

Preparation hashes include the saved analysis, company and update timestamp. A changed
preparation baseline is rejected on import. The legacy initial SYK format has no
preparation hash; it requires no existing thesis and explicitly warns that the import
baseline was captured later. Other unbound drafts require careful manual reconciliation
and are labeled as having no preparation receipt.

Approval locks the draft row and briefly locks `portfolio_analyses` against writes,
then rechecks its baseline. The table lock also fences existing writers that do not
use advisory locks, including concurrent first inserts. It does not retrofit optimistic
concurrency onto subsequent legacy saves. A five-second lock timeout fails without
applying anything; refresh the draft and retry after contention clears.

One transaction writes the native thesis, a restorable pre-import journal snapshot
for upgrades, the new `thesis_revisions` entry, an evolution snapshot and the approval
receipt. Failed writes roll back together. Exact approval retries replay the receipt;
other candidates based on the old state are blocked. There is no automatic rebasing:
prepare against the new thesis, reconcile externally and import a revised package.
Prior document history, custom analysis fields and native in-object history are
preserved; obsolete pipeline change/fact-correction banners are cleared. Import
provenance and references are attached under `analysis.externalDraft`. Evolution
snapshots do not invent or rerun scorecard assessments.

## Validation and release proof

Safe tests: `python -m unittest` only, never pytest. The PostgreSQL integration tests
initialize a completely new temporary cluster with UTF-8 and Unix-socket access,
never read DATABASE_URL and never use an existing database. They cover restart,
duplicate/concurrent uploads, approval conflicts, prepared-baseline rejection and
transaction rollback on a forced journal failure. If the local PostgreSQL binaries
are unavailable those integration tests are explicitly skipped.

Run `scripts/test-thesis-import-browser.py` with the project Python for a synthetic
browser + disposable SQL flow. Requires local PostgreSQL 16, Playwright and Chrome.
It tests preparation/download, import, reload, approval, retry, upgrade, stale-state
blocking, dismissal and malformed input at desktop/390/320px. The built-app shell
uses synthetic read responses and forbids writes. No real research is approved.

The private SYK full-source draft was validated locally against the format only.
Do not describe this as a production import or investor approval. Production checks
must remain read-only (plus safe schema initialization); validate authentication,
release identities, draft listing and template preparation without paid model calls.
