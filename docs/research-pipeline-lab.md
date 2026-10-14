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
- Stock cards show last reported collection state and verified-through dates.
  Stale Mac snapshots and unavailable status are explicit. Existing collection
  schedules/history/recovery are linked per ticker.
- Check selected originals validates stored bytes, permissions, IDs, nonempty
  selections and pack bounds (8 originals, 15 MB each, 60 MB total).
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
- Thesis selections default to a reviewable thesis proposal; note selections
  default to a summary-note draft and cannot request thesis output. Additional
  supported assignment outputs can share that same selection. These notes are
  assignment artifacts, not replacements for the original Pipeline's full
  Generate Note workflow or published research_notes record.
- Browser collection plus generation remains a separate option in the Lab. It
  uses the frozen collection policy, not existing Lab checkbox selections. The
  UI distinguishes these paths. A source-only refresh uses collection controls.

## Outstanding acceptance and implementation

1. Validate the implemented iCloud and Research-storage import paths on a real
   user-authorized selection. Synthetic checks do not establish live acceptance.
   Preserve original paths and same-name conflicts; never resolve by filename alone.
2. Expand browser-collected metadata (type, publisher, author, publication date,
   observed URLs and collection window) and connect receipts to the catalog.
   Present uncertainty; do not classify a publisher as a broker without evidence.
3. Add source relevance explanations and separate task defaults: material changes,
   contrary evidence and foundational sources for theses; event/question-focused
   sources for notes. No false consensus or arbitrary preservation percentages.
4. Make source-only refresh, inventory reload and review of newly imported items
   seamless within stock settings. Keep recurring collection separate from
   permission to spend model credits or change accepted research.
5. Add a full note draft workflow with the original note's quality controls,
   versioning, review/export and accepted-note history, while retaining existing
   Pipeline behavior. No silent publication or thesis acceptance.
6. Offer opt-in per-stock orchestration of collection, frozen-pack assembly and
   chosen drafts with explicit budgets, failure recovery and saved-output links.
   Never interpret a browser schedule as approval for unspecified paid research.
7. Prove the complete path with a user-authorized real run and investor review.
   Synthetic tests prove mechanics only, not actual AlphaSense freshness,
   source sufficiency or professional investment quality.

Historical valuation remains at its separate documented checkpoint. Do not
claim all eight priorities or ticker-only automation complete.
