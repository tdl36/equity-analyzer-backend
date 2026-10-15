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
