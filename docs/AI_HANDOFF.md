# Charlie AI engineering handoff

Updated: October 12, 2026

## T241 — historical evidence inspection integrated

Saved recovered one-pager reviews now include a collapsed Historical valuation
evidence panel. It reads retained-original inventory, unresolved retrievals and
pending review versions for the bound edition, then rechecks saved evidence when
opening a version. Empty history, unavailable context and changed baselines remain
explicit. Parallel inventory responses must match all frozen financial/source/
parent-edition digests; obsolete reads are aborted on navigation.

Production registration requires an authenticated investor session or machine key
and is explicitly read-only. POST is rejected before database/provider access,
even if provider configuration is supplied. No retrieval, proposal creation,
approval, automatic valuation calculation or paid call is enabled by this release.
The full historical foundation and synthetic tests are now included as dependencies;
its write/ingestion paths remain inaccessible in production. Targets still require
a valid saved issuer identity and coherent edition; failures do not invent values.

Validation: 106 focused historical tests, 1333 safe backend tests, 157 frontend
tests, compilation, model policy against 75097e5 and production build passed.
Synthetic full-app 390/1280px checks opened the new empty-evidence panel and existing
previews, summary retry, thesis link and download; GET-only and no overflow.
Production verification pending. Next: configured source provenance/entitlement,
source-backed historical diagnostics and investor acceptance; all eight priorities
and ticker-only unattended acceptance are still incomplete. Operations stays separate.

### T241 verified — October 13, 2026, 9:47 p.m. New York

Backend health reports ad41d84412b412c77f83e5ac3b26931f102fb241.
Cloudflare d78d5463-ee53-429d-b72f-b47f2abdce0a serves exact tested app and
service-worker bytes. All three authenticated ABT edition inventories returned
200 with valid baseline metadata and zero historical records, accurately showing
that evidence has not yet been collected. Mac agent healthy at 17.8 seconds;
unchanged and not restarted. Temporary Render replacement error cleared.
No paid research, original changes or investor decisions. This completes read-only
review integration, not historical valuation generation or unattended acceptance.

Next implementation: add a deterministic proposal builder from already-retained
verified SEC/price originals, with explicit missing-evidence diagnostics, before
enabling any retrieval or write endpoint. Keep production read-only until the
source provenance/provider configuration gate and review flow are verified.

## T240 — fair collection pickup

Queued requests now receive durable turns under the existing exclusive worker
lock. After a request is released or its lease expires, other eligible requests
are considered before it is picked again. Previously the oldest import-blocked
request could monopolize every wake. A separate small SQLite table preserves
ordering across restarts without changing frozen requests, source windows, import
retry timing or paid dispatch. Explicit request selection still works and cannot
bypass an active global worker lease. Unknown import outcomes remain unresolved.

Validation: 30 focused collection tests, 1329 safe backend tests and 156 frontend
tests passed; Python compilation, model policy against 75097e5 and production build
passed. Tests cover repeated requeue, restart, expired lease, stale owner, exact
source-window preservation and explicit request selection. No paid QA or research
approval. Deployment verification follows. Next: resume historical valuation
integration, while operations owns the existing import/log-access blocker. Live
unattended acceptance and all eight priority acceptance criteria remain open.

### T240 verified — October 13, 2026, 8:54 p.m. New York

Backend health reports 59e175fa78cfa236aeac55d19b55e4fe4a195e7c.
Cloudflare c1f5f4ab-9363-4cef-a31b-89e75fe44f45 serves exact tested app and
service-worker bytes. Mac agent restarted after checking no running pipeline jobs
or active collection lease, and its fresh heartbeat and durable claim-order table
were verified. Ten eligible queued requests were observed without claiming or
replaying them. Temporary Render replacement 502s cleared. No paid research or
investor decision. This confirms deployment, not end-to-end unattended acceptance.

## T239 — compact research issue banner

The global banner retains the company/status, issue count, saved-work destination,
coverage warning and session dismissal. Long error/recovery text now sits behind
Issue details, with bounded scrolling when expanded, instead of occupying the
reading area on every page. A changed issue resets disclosure state. No alert is
marked resolved or removed; source coverage and saved recovery remain unchanged.

Validation: 156 frontend tests, build and model-policy gate passed. Synthetic
390/1280px full-app checks used a long failure message, confirmed collapsed height
below 180px, expansion retains the complete message, and verified the ABT review
flow still loads/retries/downloads with GET-only requests and no overflow. Backend
unchanged; no paid research, approval or job mutation. Deployment receipt follows.

### T239 verified — 2026-10-13 21:44 UTC

Backend revision d07e3b107c4cca227bc64fff65bb8469291792f5 healthy.
Cloudflare f69730eb-3021-4655-b771-457cf084fe21 serves exact tested app and service-worker
bytes. Agent healthy (24.4 seconds), unchanged; no restart. Temporary Render
replacement 502 cleared. Existing unresolved jobs and research were preserved.

## T238 — saved assignment review workspace

Saved assignment links now open a review workspace directly, with the first saved
deliverable loaded automatically. Current recovered outputs take precedence over
preserved automated outputs. One-pager, stock summary and thesis changes have
separate keyboard-accessible tabs; additional saved notes/meeting questions remain
available. Thesis decisions link to the existing review screen; previewing approves
nothing. Reading panes provide download/print, explicit loading and retry states,
and abort obsolete requests when switching documents. No generation calls occur.

Completed assignments collapse sources/progress/history and earlier failure/output
controls beneath the current deliverables. New research is a separate section on
the assignment list. Focused review collapses company/workspace navigation and
removes nested mobile padding. Partial job recovery controls remain available.

Validation: 156 frontend tests, production build and unchanged-model policy gate
against 75097e5 passed. Isolated mocked full-app checks at 390/1280px verified auto
preview, original/recovered precedence, summary failure/retry, thesis destination,
HTML download and no horizontal overflow; all observed API calls were GETs. No
paid research or approval. Backend unchanged; prior 1326-test safe suite remains
applicable. Deployment/live preview verification follows.

### T238 verified — 2026-10-13 21:31 UTC

Backend health reports efaca0de068b0ea98e0918bb68699e7f4228b3f0.
Cloudflare cb0c64a1-6fa0-432d-8390-8a5132d0160c serves exact tested app/service-worker
bytes. Signed-in ABT assignment visibly opens its actual reconciled one-pager
without an extra preview click, with all three review tabs and collapsed history.
Agent heartbeat healthy (3.4 seconds); no restart required. Temporary Render
replacement errors cleared and the existing page recovered through its read poll.
No research calls or decisions were made. Live screenshot: /tmp/charlie-review-live.png.
The SYK issue remains visible; its long global banner is a remaining polish item.

## T237 — research alerts independent of Activity backlog

The global research banner now reads a dedicated failure inventory. Unrelated
review and active queues cannot consume its 200-row Activity budget. The query
selects the latest unresolved failure per company in each research workflow
without limiting coverage to recent records; parent assignments own linked stock
analysis alerts, including recovered completed assignments. Error/interrupted/
blocked outcomes are included. Source read failures still display incomplete
coverage. Existing Activity history, jobs and research are unchanged.

Validation: disposable PostgreSQL regression covers 250 unrelated attention rows,
an older blocked assignment, a recovered parent/failed child and independent
failure. 1326 backend and 154 frontend tests pass; build, compile and model policy
against 75097e5 pass. No paid research or job retries. Deployment receipt follows.

### T237 verified — 2026-10-13 21:15 UTC

Backend revision 9c5b22ec207b345c9e73355d39434cb311497d22 is healthy.
Cloudflare e3d65db5-0e51-4402-a3aa-df3ecba89ba2 serves exact tested app and service-worker
bytes. Live authenticated research-alert inventory returns 200, no unavailable
or truncated sources, and the existing SYK attention item. Recovered ABT is not
incorrectly alerted through its preserved failed child. Agent heartbeat is healthy
(49.7 seconds); no agent restart required. Transient replacement 502 cleared.
No research dispatch, retry or approval occurred.

## T236 deployment verified — 2026-10-13 21:07 UTC

The original-document feature is released. Backend health reports
8ecd6feadb1205c1b66ca4eb8498d26ae19c4eff. Cloudflare deployment
056617f8-d25b-4ebe-afe1-f7c3ad6a7095 serves exact tested app.js and
service-worker bytes. Authenticated Summary Lab sources return 200; original
retrieval without authentication returns 401. Mac agent heartbeat was healthy
(10 seconds); agent code was unchanged and no restart was required.

The user's UNH experiment 8f9c42a0-a1b7-40a2-aecc-24a788e70e97 completed all
six sections and final review before deployment and remains complete afterward.
No paid QA or automatic rerun was launched. Pending document submission IDs
survive a device reload; mocked browser checks verified reuse and clearing after
success at 390px and 1280px. Visual output quality still requires investor review;
validation used synthetic fixtures, not a paid production visual experiment.
The temporary Render replacement 502 cleared. This documentation-only receipt
uses [skip render] to avoid a second backend restart after verification.

## T236 — Summary Lab original-document review (prepared; deployment pending)

Document intake preserves exact original bytes and hashes on the backend. PDF
parts carry every page (five per part), including scanned pages, directly to the
model; images remain image blocks. Both source review and each section check
receive those originals. Visual observations are explicitly model interpretations,
not text-matched quotations. Source-part labels map back to filenames/page ranges.
Original downloads remain authenticated. Resumes verify original and preparation
hashes, retaining checkpoints. Repeated document submissions share an explicit
request identity; altered payloads cannot reuse it or dispatch a second paid job.

Word/PowerPoint/Excel intake includes structured text, cached chart data and
embedded supported images. Office layouts/chart appearances are not rendered;
formulas are not recalculated. The UI requires acknowledging these limits and
source-pass costs before Generate; PDF export is the alternative for visual
fidelity. Unsupported embedded objects/diagrams/formats fail explicitly. Text,
Markdown, CSV/TSV, JSON and HTML are supported; HTML external resources are not
fetched. Audio/YouTube retain existing transcription workflows. Limits: 8 files,
20 MB/file, 80 MB total, 40 review parts, 200 pages per PDF. No silent truncation.
Import uses no model call. Models, token caps and active text jobs are unchanged.

Validation: six synthetic document regressions (scanned page coverage, visual
passes/checkpoint replay, Office images, Excel coordinates/formulas, original
bytes, explicit acknowledgment and duplicate submission), 1325 safe backend
tests and 153 frontend tests passed. Build, compile and model-policy gate against
75097e5 passed. Isolated mocked browser checks passed at 390px and 1280px with
no overflow and no paid calls. Deployment is held while the user's resumed UNH
experiment 8f9c42a0-a1b7-40a2-aecc-24a788e70e97 is actively running. Do not restart
it, retry it, or describe this feature as released before production verification.

## T235 — shared provider credentials across devices

Charlie resolves Anthropic, Gemini and OpenAI credentials centrally for browser,
backend/recovery and Mac-agent jobs. Settings → Connections validates replacements
using read-only provider catalogs before saving; only explicit validated saves
supersede deployment credentials. Legacy synced settings cannot overwrite keys.
Browser Settings returns readiness only, clears obsolete device key caches, and
routes legacy streaming Claude requests through the authenticated backend. The
Mac agent retrieves shared credentials per new job through an agent-only endpoint;
existing jobs retain their resolved credential. No automatic paid retry is added.

Legacy saved values are not implicitly promoted over deployment secrets: the
older saved Gemini value failed its read-only catalog check. Provider status means
configured, not guaranteed access to every model. This is centralized credential
management, not workload identity federation. Secrets remain in existing server
settings/deployment secret storage. Older app installations should reload once.

Validation: 1319 safe backend tests, 153 frontend tests, touched-Python compilation,
model policy against 75097e5 and production frontend build passed. Synthetic mobile
390px and desktop 1280px Settings checks show shared status, clear stale browser
keys, and no horizontal overflow. No model-backed QA or research approvals.
The old Summary Lab run 86a98b12 remains marked running with a September 21 update;
it was preserved without retry. Deployment verification receipt follows.

### T235 deployment verified — 2026-10-13 20:27 UTC

Backend health reports aaac9fcc96cffc1c60a51351bedbfa0d2f7ffe0b.
Cloudflare bcc16083-ec7d-4fdb-9aa6-940a41075107 serves exact tested app.js
and service-worker bytes. Unauthenticated credential status returns 401;
authenticated Settings omits all three provider keys. Agent-only credential
retrieval and official read-only catalogs return 200 for Anthropic, Gemini and
OpenAI. The working deployment Gemini credential was retained. No key values
were printed or committed. After confirming zero running pipeline jobs, the Mac
agent was restarted; heartbeat age was 5.0 seconds with no recent log errors.
The transient Render replacement 502 cleared. No synthesis/model QA was run.

## T234 — bound automatic catalyst spending

Folder watcher submissions reserve one automatic attempt per folder before the
request. Later source revisions become proposals; uncertain request outcomes
never automatically repeat the paid submission. Previously handled folders migrate
to review-only behavior. Watcher state writes are atomic and corrupt state stops
processing. Linked analyst activities stay pending review instead of independently
launching a second paid recap. User automation settings and existing jobs remain
unchanged. The initial two-minute quiet period still does not prove an upload batch
is complete; deleting the local reservation ledger is not a supported recovery.

Validation: 1311 safe backend tests, 153 frontend tests, Python compilation,
model-policy gate against 75097e5 and production frontend build passed. Counts
include unrelated local historical-valuation tests, whose implementation remains
unreleased. Synthetic regressions cover changed folders, linked analyst routing,
corrupt state, failed reservation persistence and ambiguous network outcomes.
No paid QA, investor approvals or original-document changes. Deployment receipt
will be recorded after backend, hosted assets and Mac agent verification.

### T234 deployment verified — 2026-10-13 19:16 UTC

Backend health reports c7c710336e2d9bf6eb2e0b14c12f7659b60d6603.
Cloudflare deployment dccaa58f-e82e-4ea6-bdf3-7cb91fd72170 serves exact local
app.js and service-worker bytes. Agent was restarted after confirming no running
pipeline/audio work; its heartbeat is current (4.1 seconds at verification).
The transient Render 502 during replacement cleared. No paid synthesis was
started for verification. This bounds the observed duplicate-spend paths; broader
job recovery and the eight-priority acceptance work remain incomplete.

## T233 — verified original-page editorial evidence

ABT recovery exposed an older report whose frozen model excerpts omitted the
reviewed valuation pages. Pending editorial corrections may now explicitly name
`originalPage` (physical PDF page), independently extracted from the exact verified
original bytes. This never expands or changes automated report context/citations.
Current usage policy, bytes, bounded PDF/page/text limits and exact passage matching
are rechecked at submission, edition replay and preview. Old corrections retain
their original contract; all new wording remains pending investor review.
The recovery brief also preserves the report's actual named scenario order.

Validation: 1291 full safe backend tests plus the new edition replay regression;
149 frontend tests, compile, model policy gate and production build passed.
ABT's new pending thesis is saved; the 47-claim original-page edition and final
assignment receipt will be verified after this deployment. Earlier partial correction
receipts remain preserved in their private workspace; no paid model calls or approvals.

### ABT delivery verified — 2026-10-13 10:31 UTC

Assignment f9e363e4-597e-4394-b9d4-36d83def935f is complete for delivery,
with reviewPackage status pending_investor_review. Pending thesis
8c434b10-20b3-46c1-a810-107709d3ac6e is nonstale; pending edition
6a8c989e-4ca2-45e8-8b57-273480bb31bd retains 47 corrections and all eight
frozen originals. Exact replay returned the same package. Original automated
failure/report/artifacts and older proposals remain preserved; no approval occurred.
Live recovery brief and stock-summary routes returned validated HTML. Locally
rendered PDFs are one and four pages respectively, embedded Calibri, black text,
visually inspected, with no mobile horizontal overflow. Signed-in Charlie visibly
shows complete and all three reconciled output links. Private delivery receipts
and PDFs: ~/.charlie-workspaces/abt-finish-20261013/.

Backend d21217d0acf79f98d1d68f0ea4d5e4d980b7e583 verified; hosted app JS and
service-worker match local bytes. Cloudflare b16403b9-09f4-411b-b535-5f48e7db68be.
Mac heartbeat healthy. Remaining ABT action is investor review/decision, not another
collection or paid retry. Do not substitute the earlier linked thesis or original
failed outputs for the reconciled review package. This is bounded manual recovery,
not evidence that all eight product priorities or unattended workflows are complete.

## T232 — ABT manual recovery package delivery

Adds an explicit saved-review-package handoff for blocked summary/thesis/visual
assignments. It rechecks the live thesis baseline, exact frozen original pack,
source policy, pending edition evidence and lineage transactionally. Idempotent
replay cannot duplicate delivery; a conflicting package is rejected. Completion
means drafts delivered for investor review, not investor approval or a passed
original model review. The original failure, artifacts, report and earlier proposal
remain retained. The assignment prominently opens the reconciled thesis and edition.

Pending editions now support 60 distinct corrections (formerly 20), still bounded
and individually source checked; the correction chooser retains 150 recent rows.
A compact recovery brief renders explicitly pending editorial claims, complete
scenario inputs, dated comparisons and original identities in black Calibri.
Missing required corrections fail closed. Local ABT layout candidate is one A4
page with embedded Calibri and no mobile overflow; production receipt verification
will follow deployment. A fresh ABT thesis proposal is pending, not stale:
8c434b10-20b3-46c1-a810-107709d3ac6e, eight exact assignment originals.

Validation: 39 focused tests, 1289 full safe backend tests (including unrelated
uncommitted historical WIP), 149 frontend tests; compile, model-policy gate against
75097e5 and required frontend build passed. No pytest, model calls, research approval
or original changes. Live edition/package delivery is not yet recorded as complete.

## T231 — collection calendar boundary fix

New policy-driven refreshes now freeze the collecting Mac's local calendar date,
matching AlphaSense's date picker. Previously UTC midnight could freeze tomorrow
and block the evening collection until local midnight. Explicit assignment/command
windows and already-created runs remain unchanged; DGX's existing October 4–13
window still awaits the local date rollover. No originals or research were changed.

Synthetic regression verifies the New York UTC/local midnight boundary, unchanged
retry dates, the next new run after midnight, and authoritative explicit windows.
Focused collection suite: 27 passed. Full backend suite: 1229 passed (including 36
uncommitted historical WIP tests); frontend: 148 passed. Compile, model-policy gate
against 75097e5 and production build passed. Release verification pending below.
The agent must restart to reload its imported collection manager. No phase is newly
accepted; next priority remains evidence-backed historical valuation and review.

## T230 — accepted print-header regression (verified live)

T229's print header treated any explicit edition context as pending, including the
validated accepted-edition subtype. Renderer v25 now distinguishes server-validated
accepted contexts from pending contexts and ordinary report drafts. The existing
acceptance decision, source checks, body notices and review requirements are intact.
No approval performed. Synthetic two-page PDFs verified the correct repeating header
and page count separately for accepted, pending and original contexts. Pending and
accepted rendering regression suites passed. Full local suite passed 1202 tests
(including 11 uncommitted historical-valuation tests), plus 148 frontend tests,
compile, model-policy baseline 75097e5 and production build. No active jobs before
release. Historical valuation WIP remains local
and is deliberately excluded from this release until its evidence adapter is ready.

Backend 0f4bc29129c979f9da38bda2180f33b2907324e8 and hosted frontend bytes
verified; Cloudflare 59808801-ed3a-42f7-a59f-f21b7e163611. Live ABT pending
preview uses v25 and retains the pending print header. Accepted export status was
verified with synthetic retained-decision fixtures only; no live acceptance created.
Mac activity available, no pending collection commands; agent unchanged.

## Historical evidence identity WIP — October 13, 02:00 UTC wake

The local reconciliation result now includes a separate deterministic evidence
identity binding adapter version, proposal digest, issuer CIK, provider symbol,
trusted retrieval-date inventory and hashes of the actual referenced originals.
Missing originals remain explicit null receipts; changed bytes or retrieval context
change evidence identity even when the proposed observations are identical. Context
inputs are bounded and validated, and returned context does not alias caller data.
The proposal digest remains separately available and is not evidence certification.

Thirty-six focused synthetic tests and Python compile pass. No live source changes,
provider calls, approval or release. WIP remains local/uncommitted. Next add documented
corporate-action/share-class evidence checks and immutable persistence/review using
this complete evidence identity. Do not expose ratios as validated historical
valuation while `valuationReady` is false. T230 remains live, all eight acceptance
phases remain open, and unrelated dirty files are preserved.

## Historical input reconciliation WIP — October 13, 01:30 UTC wake

Added `check_market_supplement` to reconcile proposed historical price and annual
EPS amounts against both original-byte adapters. Changed prices, EPS, receipt bytes,
wrong issuer/symbol and missing/invalid trusted retrieval dates withhold the ratio.
Proposal-supplied retrieval dates cannot override the separate receipt inventory.
Matched amounts remain explicitly `valuationReady=false`: security/currency mapping,
corporate actions, filing completeness and investor review remain unresolved.
Zero/negative EPS remains not meaningful after both extractions match.

Thirty-two focused synthetic tests and Python compile pass. These uncommitted modules
remain unpublished and disconnected from live research. No paid calls, approval,
source mutation or deployment. Next implement evidence-backed share-basis checks and
bind the complete evidence context (including retrieval inventory and adapter version)
into immutable pending-supplement identity before persistence/review integration.
Do not use the current proposal-only digest as certification of the evidence context.
T230 remains live; no phase acceptance is newly claimed.

## Historical price extraction WIP — October 13, 01:00 UTC wake

Added a pure exact-byte raw daily-close extractor to the local historical evidence
adapter. It rejects adjusted-series shapes, mismatched symbols, missing exact dates,
same-day/unsettled dates, contradictory response vintages, malformed/duplicate JSON,
nonpositive or inconsistent OHLC values and invalid volume. It never substitutes a
nearby trading date. Currency, share class and split basis are deliberately not
inferred from a ticker. Twenty-six focused synthetic historical tests and compile
pass. No account key, paid query, real research mutation or approval was used.

Provider format checked against [official Alpha Vantage documentation](https://www.alphavantage.co/documentation/#daily)
and its public IBM DAILY demo response. This is an optional original-byte adapter,
not a configured provider or proof of account access. Full historical DAILY inventory
requires provider access beyond the free compact window; no purchase or integration
was made. Do not treat this adapter as available historical coverage for Charlie.

All historical WIP remains uncommitted/unpublished. Next bind extracted prices to
supplement observations with verified security/currency mapping and corporate-action
evidence, then add immutable supplement persistence and pending-edition review.
SEC companyfacts availability does not independently certify historical split basis
or complete point-in-time filing coverage. T230 remains the live application.

## Historical SEC earnings adapter WIP — October 13, 00:56 UTC

New uncommitted `historical_valuation_evidence.py` and its unit tests verify SEC
companyfacts bytes/CIK, select the latest annual period and filing strictly before
the price date, withhold conflicting latest values/start dates/accessions and reject
malformed or extreme numeric data without falling back to older values. Binding
checks reject proposed EPS/filing metadata that differ from the extracted original.
Twenty focused historical tests and compile pass. No source-fetching/model calls.

This remains unpublished WIP, not a live historical valuation feature. Next implement
price-original extraction and corporate-action evidence, then immutable supplement
storage, pending-edition linkage, review and rendering. Exact SEC bytes do not prove
share-class compatibility, split basis, completeness of the filing inventory or
correct price extraction. Keep those review requirements explicit. No existing
financial snapshot, research, pending proposal or investor decision was changed.

## Historical valuation inventory WIP — October 13, 00:26 UTC

Extended the uncommitted validator with a 1–60 observation inventory, strict version
and security identity, a canonical input digest, a 250 KB payload bound and original
byte SHA-256 comparisons. Duplicate price dates withhold every affected row rather
than silently selecting one. Invalid observations stay visible with their index.
No filesystem path or URL is opened; source bytes are supplied by the future adapter.
Both source-review and investor-review requirements remain explicit even when the
arithmetic succeeds. Eleven focused synthetic tests and compile pass.

Still not integrated, committed or deployed. Next build the evidence adapter and
publication/share-basis/filing-completeness checks before connecting to pending
editions. Existing byte hashes prove identity only, not the correctness of extracted
prices/EPS or asserted split compatibility. Preserve this WIP and original test file.
No live originals or research changed; no provider calls or release this wake.

## Historical valuation work in progress — October 12, 23:56 UTC

Uncommitted `historical_valuation.py` and `tests/unit/test_historical_valuation.py`
implement a pure, non-mutating arithmetic diagnostic for explicit annual GAAP EPS
and unadjusted-close pairs. Six focused synthetic tests and Python compile pass.
It withholds mismatched security/currency/share bases, later or same-day filings,
unsupported adjusted/TTM inputs, malformed receipts/dates/values, stale annual
periods and nonpositive prices. Zero/negative EPS is not meaningful, not zero P/E.
It labels a ratio calculated, never source-verified or investor-approved.

This is NOT integrated, committed, deployed or ready for real valuation use.
Next: bound and validate a complete supplement/duplicate-date inventory; verify
original receipts, corporate-action evidence and latest available filing selection;
then connect immutable supplement identity to edition baseline/review and rendering.
A claimed matching share basis alone is not proof against split mismatch. The
550-day annual age limit is an initial conservative bound needing adapter coverage.
No source fetch, paid call, research mutation or release occurred this wake. Preserve
these WIP files; production remains T229 (documentation commit 4f0c31e followed it).

## Next valuation increment — October 12, 23:26 UTC audit

Confirmed the current fundamentals adapter has no historical price/estimate-vintage
series and selects potentially restated annual observations. It cannot safely supply
historical point-in-time P/E. The concrete supplement/evidence/acceptance boundary is
recorded in `docs/onepager-acceptance-audit.md`. No live source or snapshot changes,
paid calls or new valuation feature in this documentation-only checkpoint. Next work
is the versioned historical supplement with publication-date and share-basis checks;
T229 remains the verified production release. All eight acceptance phases remain open.

## T229 — print continuation review status (verified live)

Illustrated renderer v24 repeats a pending/not-accepted review header and page X
of Y in A4 print margin boxes. Ordinary report prints carry a research-review-draft
header. Existing notices, source citations, financials and approval state stay intact;
Calibri/black text and body sizes are preserved. Screen layout is unchanged.
Modern Chrome PDF validation of the actual ABT preview with candidate styles found
two pages, both with the pending header and correct page count; page 2 inspected.
This does not achieve the one-page or substantive valuation acceptance criteria.
1190 backend and 148 frontend tests, compile, model-policy baseline 75097e5 and
production build passed. Desktop/phone screenshots were pixel-identical to T228.
Activity showed no running/processing jobs before release.
No paid research, source changes or investor decisions. Further editorial reduction
requires preserving the conditional arguments; do not hide content to force a page.

Backend 080aca3c79caa5e5f3212f2807ee6e903e1e14e7 verified healthy; Cloudflare
108c60ab-e36a-4933-a4ae-ab25f11449a5 deployed, hosted assets match exact bytes.
Live ABT preview remains pending; its two-page PDF has the pending header and
correct pagination on both pages. Mac snapshot available, no pending delivery;
agent unchanged. Post-release receipt retained for the next source commit.

## T228 — pending illustrated review layout (verified live)

Fix invalid nested paragraph markup in the company overview: a source-backed claim
now stays in its own description column beside the market facts. Pending claim
markers read “Pending proposal”; their full status remains in the tooltip and the
visible top/bottom “not accepted” preview notices. No claim wording, citations,
financial inputs, source selection or approval state changes. Renderer v23.

1189 backend and 148 frontend tests, compile, model-policy baseline 75097e5 and
production build passed. Activity reported no running/processing jobs before release.
Focused regression verifies the overview children, claim containment and pending
status. Static candidate from the real ABT pending preview inspected at desktop and
phone widths without horizontal overflow. A4 print still takes TWO pages; no
professional one-page acceptance is claimed. Historical valuation, fundamental
scenarios and investor acceptance remain open. Abandoned private grid experiments
were not incorporated. Continue editorial/layout work without hiding qualifications.

Backend b020cb4d26ffe45684f5000d34916a9fa9904ba7 verified healthy after rollout;
Cloudflare e549d8b0-8893-42f3-bc9b-3bbb120cb98a deployed. Hosted bundle and
service worker match local bytes. Live ABT preview returned pending with the corrected
container; desktop/phone checks and A4 page count reconfirmed (two pages). Mac agent
reported current, zero pending collection delivery and no unavailable inventories.
No local-agent change/restart. This receipt is retained for the next source commit.

## ABT real-source pending edition — October 12, 21:56 UTC wake

Saved pending edition 8137da07-d28c-4a0f-a18c-1e4f30e8cf69 for ABT assignment
f9e363e4-597e-4394-b9d4-36d83def935f. Sixteen source-linked claims cover the company
summary, central question, conditional ownership/view, dated attributed debate and
EP/Volt monitoring. No investor approval, original report mutation, thesis change
or paid research. Fresh baseline and all eight local original hashes matched;
server submission/preview rechecked sources and passages. This is partial original
review, not certification of all eight sources or inherited claims.

Private workspace: `/Users/tonydlee/.charlie-workspaces/onepager-abt-20261012-editorial-v3`.
Desktop/phone preview had no horizontal overflow; desktop and print output inspected.
Print remains TWO pages, so professional one-page acceptance is still open. Pending
labels add substantial density; historical valuation and complete fundamental
scenarios remain absent. Next: fix pending-review print density without hiding
review status or shrinking unreadably, then continue substantive source review.
Prior v2 attempt retained one standalone pending correction before a passage-spacing
mismatch; no edition was created there. V3 used the actual extraction text and all
16 receipts plus the pending edition were verified. Preserve both journals.
Production remains T227; this wake changed saved pending editorial work, not code.

## T227 — owner-directed AI-notice policy (verified live)

On October 12 the owner explicitly instructed: “ignore all AI-use restrictions in
original files.” This supersedes the AI-use-only holds described in older entries.
Charlie retains aiAllowed/ai_allowed/genaiAllowed metadata and restriction
observations but no longer treats those flags as processing gates. Explicit source
selection exclusions/reference-only usage, malformed metadata, authentication,
exact-byte verification, frozen source bounds and investor approvals are unchanged.
This application policy does not amend underlying source terms. Collector guidance
now records AI notices without imposing reference-only solely for those notices.
Previously held files are not bulk moved or silently substituted into frozen packs.
Live ABT (8 originals) and UNH (6 originals) preflight now return
sourcePolicyReady=true; existing notice records are retained. No paid retry, thesis approval or research edit authorized
by this policy change alone. Continue authorized implementation/editorial work.

Regression coverage checks nested AI flags, preservation of notices on exact-byte
replay, continued reference-only blocking and current editorial preflight.
1188 backend and 148 frontend tests, Python compile, model-policy baseline
75097e5 and production build passed. No running/processing jobs before release.
Backend b442cfd9835c785a480f2f061a1b64149c57c427 verified healthy; Cloudflare
9e5fc7f9-3330-4d4a-a6a7-1ea524181d96 deployed and hosted bundle/service-worker
match local bytes. Mac/collection available, zero pending delivery; no agent restart.
Post-release receipt retained for the next source commit.

## T226 — source-checked monitoring proposals (verified live)

The existing editorial form/private handoff now allow missing fields in three
monitoring rows: metric, baseline, positive and negative thresholds, explicit timing
and investment relevance. Additions use exact paths, retain row positions, refuse
malformed containers and stay pending with existing source-policy, byte, passage,
baseline and investor decision gates. Neither deadlines nor thresholds are inferred.
Both brief and illustrated renderers keep monitoring inputs in their own anchored
row, preserve qualifications and show missing inputs explicitly. Historical saved
exports remain unchanged. No live claims, approval or paid research were created.

Synthetic source/edition tests cover additions, malformed paths, unchanged originals,
pending labels, orphan timing and qualified timing in both renderers. Synthetic
Chrome desktop/phone checks showed no horizontal overflow and monitoring sections
were inspected visually; these are not real-source or full print acceptance.
1187 backend and 148 frontend tests passed. Production build, compile and
model-policy baseline 75097e5 passed. Backend b210f3a52f745b3c818bbb763771a55ff46635b0
verified healthy after rollout; Cloudflare 6a5d4cf2-deed-4743-812e-60f1b5bf1c14
and hosted bundle/service-worker exact bytes verified. Live read-only preparation
exposes timing additions while ABT sourcePolicyReady remains false. Mac/collection
snapshots are available with zero pending delivery. No agent change/restart.
Post-release receipt retained locally for the next source commit. ABT/UNH restrictions remain held;
legacy WELL, historical valuation, investor benchmarks and full one-click acceptance
remain open.

## UNH source-policy finding — October 12, 20:54 UTC wake

UNH's six frozen originals were retrieved through verified read routes and matched
independently to their saved SHA-256 values. PDF extraction is not substantive
review; no investment-content assessment or new proposal was completed. The
Evercore September 14 original contains explicit AI-input/use restrictions on
PDF page 7, despite passing saved metadata preflight. Written consent is not
established. T224 restriction recording set aiAllowed=false for that exact source;
T225 live preflight then returned sourcePolicyReady=false and identified it.
Originals and the completed UNH report/outputs remain preserved; no paid replay,
approval or report rewrite. Existing completion does not certify source permission.

Private receipts and held status:
`/Users/tonydlee/.charlie-workspaces/onepager-unh-20261012-source-review`.
Both ABT and UNH editorial packs now require permission resolution or an explicitly
eligible new source pack. Their saved research must not be silently repurposed to
bypass the restriction. No code change/deployment in this wake; production T225.
Continue independent implementation within the eight priorities while these real
source benchmark packs remain held. Metadata preflight is not a substitute for
original disclosure inspection. This finding was notified once; deduplicate it.

## T225 — editorial source-policy preflight (verified live)

Editorial preparation now reads current document metadata for every frozen source
and labels each eligible, restricted or missing. This read-only metadata check is
not original-byte verification or permission certification. The correction form
holds drafting when any source is unavailable/restricted or preflight is absent,
while keeping saved corrections/history and reload available. Private handoff
preparation fails before saving a baseline; unsent claims recheck policy. Existing
edition-response reconciliation retains its exact replay identity and is not
blocked by this preflight. Transactional submission/export checks remain authoritative.

1183 backend and 148 frontend tests, Python compile, model-policy baseline
75097e5 and production build passed. Synthetic Chrome verified the hold and readable history at desktop/mobile widths,
read-only reload and restored-policy form access, with no POST; fixture CSS only.
Backend 58e1e6347efb616f10c134a8e83bfc1bb371e6fa verified healthy after rollout;
Cloudflare d162f00e-7737-4cc6-a7ab-e2af179a72c1 and hosted bundle/service-worker
exact bytes verified. Live read-only ABT preflight returns sourcePolicyReady=false
and identifies the restricted JPMorgan original. Mac/collection snapshots remain
current with no pending delivery. No agent change/restart. Post-release receipt
retained for the next source commit.
No live research, proposal, approval or paid run was created. ABT remains held by
the observed source restriction. Full priority/one-click acceptance remains open.

## T224 — exact-original restriction recording (verified live)

ABT original-source review found an explicit third-party AI restriction in the
October 2 JPMorgan preview (disclosures, PDF page 10), while current cloud metadata
marked it usage=research. Original SHA-256 matches the frozen report. Further use
of this source for the editorial draft stopped. This is a newly observed source
restriction, not a change to the source bytes or a new research-quality judgment.

Added authenticated POST research/<workflow>-run/<id>/source/<sourceId>/restrict,
requiring frozen SHA-256, bounded reason and locator. It locks the saved report and
original, verifies exact bytes, and adds aiAllowed=false with a dated observation.
It preserves all prior metadata, original bytes and research. Concurrent/repeated
requests retain the first restriction; no unrestrict/approval operation is provided.
Existing eligible() gates reject the resulting metadata. No paid replay is allowed.

1181 backend and 148 frontend tests passed, including disposable PostgreSQL
concurrent restriction/replay and invalid hash/bytes/body checks. Python compile,
model-policy baseline 75097e5 and production build passed. No running/processing
activity was observed before release.

Verified backend ada5c387ffa7e649244d43562a5fa6044d5b15d8 and Cloudflare
8be1306b-c6db-43ca-9f0f-c4475d2c1347; hosted /dist/app.js and service-worker
match local bytes. The exact JPMorgan original now has aiAllowed=false, preserves
its prior metadata/hash, and fails eligible(). Private restriction receipt saved
with the source-review workspace. No agent code changed or restart performed.
This resolves recording the restriction, not permission or the affected research.
Saved ABT outputs may already contain this source; they are not silently removed or
rewritten. New proposed editions using the full pack should fail current-policy
checks. Recovery requires resolving permission or explicitly preparing a new
eligible source pack through the established workflow; no ambiguous/paid retry.

## Real-source editorial review — October 12, 19:24 UTC wake

Production remains T223. Read-only ABT preparation and exact-byte retrieval of
all eight frozen originals succeeded. Substantive page-level review covered only
three pages across three originals; the private assessment identifies chronology,
valuation-method and accounting-basis distinctions that a proposed one-pager must
preserve. See docs/onepager-acceptance-audit.md for proof boundaries and private
artifact location. No new research, proposal submission or approval; no deployment.
Real-source work remains incomplete and must continue before quality acceptance.

## T223 — missing investment-section proposals (verified live)

The existing source-checked editorial workflow can now propose missing company
one-liner, central investment question, investment view, up to three ownership
claims and two paired debate rows (including resolution signals). These are exact
allowlisted paths; malformed containers and arbitrary fields cannot be replaced.
Only selected proposals enter a separate pending edition. Original report/citation
state remains unchanged, gaps stay gaps, and investor review remains required.
Existing source-byte, restriction, passage, baseline and edition gates apply.
The UI labels new fields generically rather than calling all additions scenarios.
Private handoffs must prepare again to obtain the expanded permitted field list.

1179 backend and 148 frontend tests, touched Python compilation, model-policy
baseline 75097e5 and production build passed. Synthetic tests cover source-backed
pending additions, exact row positions, original preservation, stale baselines and
malformed/forbidden structures. Isolated Chrome verified a missing ownership claim
with two originals, submission payload and saved pending display at desktop/mobile
widths (fixture CSS only). No live claim, thesis or edition was changed or approved.
Real-source quality, investor acceptance and all eight priorities remain open.
Backend ed2b834d0823fefa00c56af2d216e0d85f87a601 verified healthy after rollout;
Cloudflare 27ef78d0-e4ab-49b3-a653-fcf67e83032b deployed; hosted JS/service-worker
exact bytes verified. Activity returned current Mac/collection snapshots with zero
pending deliveries and no unavailable sources. No agent source change/restart.
Receipt retained for next source commit. Next phase-2 work should use the now
available editorial path to assess source-backed real examples; do not equate
more authoring controls with real-source quality acceptance.

## T222 — independent meeting/thesis output recovery (verified live)

After completed research, an expected meeting-selection/handoff validation failure
no longer prevents independent thesis proposal preparation. Each meeting/thesis
failure is retained in outputErrors, while successful artifacts and receipts remain
available. The assignment stays attention until all requested outputs are saved;
explicit resume skips completed meeting/thesis outputs and clears resolved errors.
Thesis failures retain their existing leading error text for bounded repair and
manual-review gates. Unexpected exceptions still stop for inspection. No source
policy relaxation, provider retry, approval or live research was introduced.
The assignment UI displays each failed output alongside retained draft links.

1177 backend and 148 frontend tests, Python compile, model-policy baseline
75097e5 and production build passed. Disposable PostgreSQL tests verify that a
meeting failure preserves the thesis/visual, resume never recreates the saved
thesis, and simultaneous failures both remain visible. Synthetic Chrome checked
partial failure and retained output links at desktop/mobile sizes without writes;
fixture CSS only. Full unattended acceptance and all eight priorities remain
incomplete. Backend ddb345ee4b9c5d011a5025213fc291c221f9e413 verified healthy;
Cloudflare f5d354f8-92bc-4cfa-89f6-301b3f2c8386 deployed and hosted JS/service-worker
exact bytes matched. Mac heartbeat 18:28:36 UTC, collection snapshot 18:27:25 UTC,
zero pending deliveries and no unavailable sources. Existing ABT/UNH/SYK states
and artifact counts unchanged. No agent source change/restart. Post-deployment
receipt retained for next source commit.

## T221 — assignment submission survives reload (verified live)

Research assignment submission now journals its exact request in browser-tab
session storage before any POST. Reload or navigation retains ticker, dates,
outputs, origin and request identity. An unresolved intent blocks a fresh
submission and offers targeted read-only reconciliation or exact idempotent
resubmission. A completed/attention assignment can be found without POST or
paid-stage resume. Matching receipts clear the journal; unknown responses,
receipt mismatches and storage errors fail closed. Only a first definitive
400/409 rejection releases the intent for editing; uncertain retries retain it.
This covers reloads in the same browser tab, not cross-device/global recovery.
Corrupt storage retains a visible error while server-side assignments remain
inspectable. Existing originals, outputs and paid-stage retry rules are unchanged.

1175 backend and 148 frontend tests passed, production build and model-policy
baseline 75097e5 passed. Synthetic isolated Chrome tested reload after lost
response, exact body/identity retry, completed-record read-only reconciliation,
initial rejection recovery and desktop/mobile widths. Fixture CSS only; no live
submission or paid research. Full unattended acceptance, legacy WELL disposition,
real-source editorial quality and remaining origin entry points are still open.
All eight priorities remain incomplete. Backend fb33db0dfc9713a41816f205424e3438657fa7f9
verified healthy after rollout timeout/502s. Cloudflare version
54edb365-e516-4c9f-a182-d17b73ae8345 deployed; hosted JS/service-worker exact bytes
verified. Activity recovered with Mac heartbeat 17:59:46 UTC, collection snapshot
17:59:55 UTC, no pending deliveries or unavailable sources. No agent source
change/restart. Receipt retained for the next source commit.

## T220 — one-pager refresh origin (verified live)

Saved one-pager edition history now opens a coordinated recent-source refresh,
with visual output required and optional shared-pack outputs. Submission freezes
origin assignment/report, report fingerprint, financial snapshot digest and current
accepted edition. Locked server reads reject a changed origin before queuing any
collection command. Exact request replay retains the original baseline even after
it changes; conflicting reuse fails. The saved assignment links back to its origin.
Existing originals, reports and editions remain unchanged. Prior editorial wording
is not automatically merged into generated research; the UI states this explicitly.
The existing unfinished-company guard remains, including attention assignments.

1175 backend and 146 frontend tests passed, plus Python compile, model-policy
baseline 75097e5 and production build. Synthetic Chrome desktop/mobile form checks
verified visual-only defaults, frozen origin payload and exact lost-response retry;
fixture CSS only, not full workspace visual acceptance. No paid research or live
submission/approval. Thesis/meeting origin entry points, editorial carry-forward,
real-source quality and unattended full-path acceptance remain open. All eight
priorities remain incomplete. Backend 22de183d35825e8ee4ccde93d00e67c6fadf186e
verified healthy at 17:31:53 UTC after rollout 502s. Cloudflare version
fc165c98-04bc-4fc0-bbcf-14e2396615a7 deployed; hosted JS and service-worker exact
bytes verified. Mac heartbeat/collection snapshot available, zero pending delivery,
no unavailable sources. ABT remains attention with two retained artifacts; no
new downstream completion claimed. No agent source change/restart. This receipt
is retained for the next source commit to avoid a docs-only backend restart.

## T219 — scenario arithmetic in edition review (verified live)

Edition history and the explicit investor decision panel now display P/E scenario
arithmetic diagnostics with inspectable EPS, multiple, implied value and dated
return inputs. Inputs come only from saved supported claims plus that edition's
explicit proposals. Unsupported originals do not fill gaps; stale editions withhold
diagnostics. The read-only check neither revalidates originals nor approves or
blocks an investor decision. Source/assumption accuracy is explicitly separate.
Original calculations and reports are never rewritten or rebased to live prices.
Three new tests cover unchanged versus overridden claims, unsupported gaps,
missing rows, incomplete inputs and year/basis mismatches. 1172 backend and 146
frontend tests, compile, model-policy baseline 75097e5 and build passed. Synthetic
Chrome desktop/mobile review displayed a mismatch and its inputs without any
write or decision; fixture CSS is not full workspace visual acceptance.
Real-source one-pager quality and origin-specific refresh remain open. All eight
priorities remain incomplete. Backend c536d101381e743c297c835c5c985fd59e21eb98
verified healthy after rollout 502s. Cloudflare 35583a46-4f4c-4611-872f-9fe9bba8e95a
deployed; hosted JS/service-worker exact bytes verified. Mac heartbeat and collection
snapshot current with zero pending delivery and no unavailable sources. No agent
change/restart. This receipt is retained for the next source commit.

## T218 — bounded missing scenario inputs (verified live)

Editorial preparation exposes missing/null fields for up to three scenario rows
and nine named inputs. Existing fields still use replacement corrections; new
inputs carry before=null and appear explicitly as additions in investor review.
Invalid scenario containers, unknown paths, extra rows and structure replacement
are forbidden. Pending preview inserts only selected additions into a private
copy in any selection order; original reports and financial snapshots remain
unchanged. Every addition uses existing source/policy checks and explicit edition
review. Partial rows remain partial, not a complete or accepted valuation.
Three new synthetic tests cover absent/list/dict/null containers, source-backed
pending additions, retained originals, bounds and stale baselines. Full safe
suites passed 1169 backend and 146 frontend tests; compile, model policy baseline
75097e5 and production build passed.
Isolated Chrome desktop/mobile checks exercised a missing input and two-source
submission using synthetic data and fixture CSS. No live research or proposal
was changed. Real scenario content/accuracy, full workspace visual acceptance,
origin-specific refresh and the eight-priority acceptance remain open.
Backend bd76b9e938b8f1d92dc82d3a8b1fee7be5048171 verified healthy after
rollout 502s. Cloudflare ba3dee71-1ea2-472d-b52a-88e5935ee9f4 deployed;
hosted JS/service-worker exact bytes verified. Mac heartbeat current at
16:30:10 UTC with zero pending collection deliveries and no unavailable sources.
No agent source change/restart. Receipt retained for next source commit.

## T217 — multiple originals per editorial claim (verified live)

One-pager corrections now support a primary original plus up to seven additional
frozen originals. The browser and private handoff share the bounded contract.
Each passage, exact original bytes and current usage restrictions are checked;
edition creation, preview, investor decision and export retain every citation.
Existing single-source payloads and records retain their signatures and shape.
No source discovery, paid call, research overwrite or investor approval is added.
Four new synthetic tests cover multi-source preview retention, changed/missing/
restricted secondary originals, forged evidence, HTTP replay and legacy records.
Full suite passed 1165 backend tests before the final additional HTTP test; all
four new tests passed together afterward. Frontend 146 tests, compile, model
policy baseline 75097e5 and production build passed. Isolated headless Chrome
form validation passed at desktop and 390px mobile widths, including the exact
submitted two-source payload and saved evidence display; fixture CSS only, not
full workspace visual acceptance. No live correction was submitted.
New scenario-row authoring, real-source editorial benchmarks, origin-specific
refresh and full eight-priority acceptance remain open. Backend revision
0391b627909bdfe4c716f32ef03dae4b5e5c2dcc verified healthy after rollout
502; Cloudflare 3ea1291a-4f62-4f0e-b3b4-89c5ab49f721 deployed. Hosted JS and
service-worker exact bytes verified. Agent snapshot available with zero pending
collection deliveries; no agent source change or restart. Post-deploy receipt
retained for the next source commit to avoid a docs-only backend restart.

## T216 — resumable private editorial handoff (verified live)

Added onepager_handoff.py prepare/submit using fixed authenticated correction and
edition routes. Private local preparation freezes report, financials and accepted
parent; no approval route, model call, original download or source-review claim.
The bounded proposal selects existing scalar fields with one frozen source each.
Journaling precedes writes, retains partial receipts, rejects changed submissions,
and reconciles a lost edition response even after the parent advances. Server
validation remains authoritative. Complete selections do not inherit old claims.
Read-only live preparation succeeded for the saved ABT assignment: 268 permitted
scalar fields, retained privately in ~/.charlie-workspaces/onepager-abt-t216.
The proposal remains empty; no live correction was submitted and no sources were
claimed as reviewed. 1162 backend and 146 frontend tests, compile, model-policy
baseline 75097e5 and production frontend build passed.
Seven synthetic tests cover pending handoff, lost responses, stale baselines,
changed content, prohibited fields/locations and receipt mismatch. New scenario
rows, multi-source claims and real-source quality acceptance remain open.
Backend 3d97e538e39b227ef1b27e2af659156676f5ce32 verified healthy after rollout
502s; Cloudflare 41b2158f-182f-4637-b7f9-4389a0455383 deployed, hosted JS and
service-worker exact bytes verified. No Mac agent source change/restart. Final
receipts and documentation clarification retained for the next source commit.


## T215 — bounded metadata recovery (verified live)

Audited the eight acknowledged startup uploads from 10:41 EDT. Two Agilent
originals have empty cloud metadata but retained eligible collector records and
matching local original hashes; six ABBV originals have empty metadata and no
matching collector records. Prior ABBV metadata cannot be inferred. An earlier
ABBV batch logged an ambiguous response, so eight acknowledgements are not proof
of the complete affected set. No research rerun or unknown metadata repair.
Added explicit restoreEmptyMetadata to the authenticated collector import route.
Recovery requires an existing exact-byte original and validated collection payload;
only an entirely empty metadata record can be filled. Nonempty metadata and usage
restrictions remain untouched. Exact metadata replay succeeds without mutation;
missing/different originals fail. Local source policy and original checks still
apply before any recovery submission. Automatic import behavior is unchanged.
Focused tests cover restoration, idempotency, restrictions, other metadata and
missing/changed bytes. Both Agilent records were recovered through the authenticated endpoint after
local policy checks; server exact-byte receipts and independent metadata reads
verified their provenance. Private audit: AlphaSense/startup-upload-audit-20261012.json
under Charlie Application Support. ABBV prior metadata remains unknown.
1155 backend and 146 frontend tests, Python compilation, model policy and build
passed. Backend b2782091b854150f7be7c9fe76be4465d14bdc4e verified; Cloudflare
841b6082-f63d-4dda-84f1-f5400dbdf77f and hosted asset bytes verified. No agent
source change/restart in this increment. Post-deployment receipts are kept here
for the next source commit to avoid an extra docs-only backend restart.
All eight priorities remain incomplete. Next return to the one-pager quality and
origin-specific refresh work after verifying the bounded recovery.

## T214 — fail closed on unavailable startup inventory (verified live)

A Mac restart during the T213 backend rollout exposed an existing recovery bug:
non-200 document inventories were treated as empty, triggering duplicate original
uploads; non-200 note lookups other than actual absence also attempted imports.
Stopped that startup pass, preserved originals, and changed both import paths to
abort on unavailable/unexpected inventories. Only the exact known note-not-found
response permits a note scan. Document inventories require a complete list with
valid filenames. No original bytes are read on inventory failure. Existing files
are not renamed, removed or mass-reprocessed for validation. The affected startup
logged some duplicate uploads before it was stopped; no claim that all source
metadata was audited. Agent resumed with the fix once T213 backend was healthy.
Startup writes additionally use insert-only conflict handling, preserving bytes
and policy metadata if a document arrives after inventory. Counts come from the
server receipt. Four focused synthetic/disposable-DB tests exercise HTTP/auth errors,
timeout, malformed inventories, existing-file suppression and preservation.
1154 backend tests, 146 frontend tests, focused preservation tests, touched Python
compilation, model-policy baseline 75097e5 and required production build passed.
Backend 30fbe2ccde05e29879a8280f01597d427b844cdd verified healthy after rollout
502s. Cloudflare 2084d522-9b90-44f3-a6ce-6735f62dfe36 deployed; hosted JS and
service-worker bytes match. Agent 14:49:10 UTC, collection 14:48:50, no pending
deliveries/unavailable workflows or active jobs. Agent was explicitly restarted
after rollout; the normal collector still reports incomplete catalyst sources,
which are not treated as research completion.

## T213 — shared-report meeting questions (verified live)

Added an optional fifth assignment output with frozen meeting date/type. Existing
four-output defaults stay unchanged. Supported questions are copied verbatim from
the same completed report, with report provenance and no extra generation call.
The handoff locks and revalidates every frozen original's exact bytes and usage
policy before saving. Its persisted selection and deterministic receipt recover
an interrupted response without duplicate meetings or new research runs. Missing
supported questions leave the assignment in attention with other artifacts saved.
This is a question transfer, not the full generated meeting-preparation workflow.
No live approval or paid QA run occurred. Synthetic browser submission verified
required date, meeting type, five selected outputs and queued persistence.
Focused tests cover source restriction/byte changes and interrupted handoff recovery.
1150 backend and 146 frontend tests passed, along with compile, model-policy and
production build checks. Backend cd25e96eb769eb3ddb35de1e47228f43a202d789 verified
healthy after rollout 502s; Cloudflare a9384702-6bb2-4a05-8326-553c9f051f6c and
hosted JS/service-worker bytes verified. Mac reload exposed the T214 startup bug
below; reload completed with its fix. Browser proof: /tmp/charlie-t213-meeting.png.
All eight priorities remain open against the acceptance plan; continue real-source
one-pager quality and origin-specific latest-source refresh after this increment.

## T212 — investor edition decisions and explicit exports (verified live)

Backend 4a54455a47493504354a84d7806662ed258de721 verified healthy after transient rollout 502s.
Cloudflare faf1c078-57b6-4427-b748-8f57202d78c5 deployed; hosted JS and service-worker
bytes match the tested build. Mac heartbeat and collection snapshot were fresh,
no pending deliveries or active audio jobs. No agent source change/restart.

Implemented the full saved-edition review/decision/export path. Only a verified
password-investor session supplies the server principal; machine API keys, open
mode and client identity fields cannot approve editions. Review shows complete
before/after wording and corrections removed from the prior accepted selection,
requires explicit acknowledgement, and retains retry identity. Transactional
accept/reject uses the existing exact evidence and concurrency gates.
Accepted contexts revalidate the saved decision, immutable selection, all frozen
original bytes/policies and financial inputs. Both layouts distinguish human
editorial review from original model support; the shared evidence/thesis gate is
unchanged. Only an explicit accepted-version URL exports printable HTML. Pending,
rejected, changed and restricted evidence fails closed. Current/prior editions
remain separate; no original report, default artifact or thesis is overwritten.
Synthetic disposable-DB UI acceptance advanced the pointer and retained the prior
version; the browser downloaded its brief. Native Chrome printed both sparse
accepted layouts as one A4 page, visually inspected without clipping; Calibri
fonts retained. Desktop and 390px history layouts checked. Evidence remains in
/tmp/charlie-t207-preview/accepted-*-T212.*, with the downloaded synthetic brief
in Downloads and /tmp/charlie-t212-review.png. These are synthetic fixtures,
not investor quality acceptance or a live approval. No paid calls were made.
1145 backend tests and 146 frontend tests passed; focused suites, touched Python
compilation, model policy against 75097e5 and production frontend build passed.
Next continue the actual real-source one-pager quality gaps and cross-workspace
latest-source refresh; the eight acceptance criteria remain open.

## T211 — edition lineage reads (verified live; decision writes internal)

Commit 2c035aa pushed to main. Cloudflare a9b25dd3-bcce-4dfc-976f-b254a1cccd61
deployed; hosted /dist/app.js and service-worker bytes match. Render returned transient HTTP 502 during rollout, then verified healthy at
revision 2c035aace572eed027fb1291502f22319dddf451.
Mac last seen 14:09:03 UTC, collection 14:08:25, pending delivery zero and
unavailable empty. No active audio items observed; no agent change/restart.

Edition listing and preparation now read the separate decision history under the
assignment lock, carry the current accepted parent, and fail closed on missing or
inconsistent lineage/content. Recorded decisions are shown separately from immutable
pending content without exposing the investor authentication principal. Decided
editions cannot open pending previews; a new pending successor uses the current
parent. Replayed submissions report their effective decision state. No decision
write route or investor acceptance control is exposed, and accepted export remains
unavailable. Original research and default artifacts remain unchanged.
1139 backend and 145 frontend tests passed, plus focused tests, Python compilation,
model policy against 75097e5 and the required frontend build.
Three additional disposable PostgreSQL cases cover list/replay states, successor
preparation/preview and damaged lineage; frontend coverage checks decided preview
suppression. Synthetic real-blueprint UI inspection verified pending, rejected and
current accepted cards and the successor preview on desktop; the 390px viewport
had no horizontal overflow. This is synthetic evidence, not investor acceptance.
Next: accepted evidence/render/export and authenticated investor decision UI; keep
the write helper internal until those paths are coherent. All eight priorities
remain incomplete. No live decision, paid call or original-source change occurred.

## T210 — internal decision persistence (verified live; internal only)

Added an internal transaction helper for explicit investor decisions, without a
public route, UI control or automatic caller. It serializes each assignment's
lineage, reloads/locks the edition, report, corrections and all originals, invokes
the T209 validator, and stores a separate immutable decision with trusted caller
investor identity and database time. Accept advances only the separate edition
head; reject leaves it unchanged. Original report, assignment outputs, pending
edition and correction records remain untouched. Exact retries return the retained
decision even after later evidence changes or successor acceptance; changed
payload/actor or a second decision on the same edition conflicts.
Eight disposable PostgreSQL tests cover simultaneous replay, competing accepts,
rejection isolation, identity/scope, stale evidence, late transaction rollback,
successor rollback/advance and source policy locks through validation. No production
decision or schema was created. This helper is not an authentication boundary: the
future route must obtain investor identity from trusted server authentication.
Next integrate authenticated decision/list/preparation handling, current-parent
selection and decided-state previews, then accepted evidence/render/export and
investor review UI. Existing pending APIs are deliberately not wired to this store;
never expose its write operation before those read/lineage paths are coherent.
1136 backend and 144 frontend tests passed, plus Python compilation, model policy
against 75097e5 and the required frontend build. No product UI changed.
Backend f6345fe16464248d6b513f12cef8d39fbd3d9e49 verified healthy at
13:36:36 UTC after transient rollout 502s. Cloudflare
db271dd0-e923-4fa6-84fb-7a5ce13e7c52 deployed; hosted JS and service-worker
bytes match the tested build. Mac last seen 13:36:22, collection 13:35:55,
pending delivery zero and unavailable empty. No agent change or restart.
No live approval, paid call, research mutation or source changes occurred.

## T209 — decision identity contract (verified live; internal only)

Added an internal pure validator for future explicit accept/reject decisions.
It binds the exact edition, ordered corrections, report, financial snapshot and
expected accepted parent, rebuilds the pending record from server-loaded evidence,
and verifies every frozen original including unchanged-claim sources. Both actions
fail on stale/restricted/missing evidence. The returned intent has no accepted
status, human-review provenance or renderer claims; original records remain pending.
Six synthetic tests cover identity/flag rejection, changed parent, stale/tampered
records, complete-pack byte/policy checks and action-bound replay digests.
This is not connected to an endpoint or UI and cannot record investor decisions.
Next: authorized transactional decision persistence, idempotency and concurrent
parent locking, then accepted-context rendering and explicit version exports.
The persistence caller must check prior decision replay/conflict before fresh
validation, lock/reload the accepted lineage and all evidence, and commit its
separate decision plus accepted pointer atomically. A rejected edition never
advances the pointer. Selection semantics are a complete selection against the
original saved report, never an implicit merge of parent corrections. A future
review UI must make removed prior corrections explicit before acceptance.
1128 backend and 144 frontend tests passed, along with Python compilation,
model policy against 75097e5 and the required frontend build. No product UI changed.
Backend 8f84de04c1c1bd042a1ce9df398863fd70c7e4b9 verified healthy at
13:05:37 UTC after transient rollout 502s. Cloudflare
b1b72f47-e302-4150-b9c9-ee0984a11e81 deployed; hosted JS and service-worker
bytes match the tested build. Mac last seen 13:05:23, collection 13:05:29,
pending delivery zero, unavailable empty. No agent change or restart.
No live edition approval, research mutation or model call was performed.

## T208 print verification checkpoint — October 12, 12:35 UTC

Native Chrome Save as PDF was exercised through supported computer-use controls
for the synthetic pending investment map and brief. Both HTML files were compared
with fresh current-code renders and matched exactly. Both actual PDFs are one A4
page, embed only Calibri variants, retain the complete proposed claim, show three
not-accepted labels and two edition/report notices, and contain only black text.
Poppler page renders were visually inspected: no overlapping or clipped text in
these sparse fixtures; extracted character boxes remain within the page. Default
margins, headers/footers off, background graphics off. The temporary Chrome tab
was closed and existing tabs preserved.
Evidence is retained outside the repository and watched original folders at
`/Users/tonydlee/Charlie-private/onepager-review-20261012/pending-print-T208/`:
`pending-map-T208.pdf` SHA-256
`224cb9642be6580256831a8edb4adb61d6c96dc3117310ce796cea1b798304f7`;
`pending-brief-T208.pdf` SHA-256
`3774d03ca7c922bbadae40da6c57fcca66d93e770599d370629f0fdc549b42b7`.
This closes only sparse synthetic pending-print inspection. It does not verify
dense multi-correction pagination, real-source investor quality acceptance,
a production PDF export, accepted edition decisions/lineage or accepted exports.
Next implementation remains transactional investor decisions with exact parent
and evidence checks, followed by explicit accepted rendering/export; automation
must never approve a live edition. No code or live research changed in this
checkpoint, so tests/build/release markers were not repeated. T208 remains the
latest product release; its deployment receipts are recorded below.

## T208 — revalidated pending preview access (verified live)

Saved editions now expose Preview pending investment map and Preview pending
brief. The private/no-store GET route names both assignment and edition and rejects
unknown layouts/approval flags. It locks the assignment, edition, report, corrections
and every frozen original through revalidation and rendering. All pack originals,
including unchanged claim sources, must retain exact bytes and eligible usage.
Missing/changed/restricted evidence or stale financial/report baselines returns an
error without HTML. No schema writes, provider calls, acceptance or research changes.
The UI checks returned edition/report/company identity, uses an empty-sandbox iframe,
clears prior previews before opening/reloading and retains explicit pending labels.
Four disposable PostgreSQL tests cover both layouts, scoped/private read-only access,
no GET schema creation, unchanged-source restrictions, stale/tampered evidence and
concurrent updates blocked until rendering releases locks. One frontend identity
test rejects unrelated or accepted preview responses. 1122 backend and 144 frontend
tests, Python compilation, policy baseline 75097e5 and the production build passed.
Source whitespace checks pass; the generated bundle retains pre-existing Markdown
hard-break spaces in an unrelated template literal. Synthetic desktop and
390px mobile UI opened/switched both layouts against the real blueprint backed by
an isolated disposable database; reload clears the previous preview. Outer mobile
page width remained 390px. Evidence: /tmp/charlie-t208-ui/preview-desktop.png.
Backend f78b1c108fca575d9cad05b4e0c4d2433c92242b verified healthy after
transient rollout 502s. Cloudflare 2c893413-f80e-48ae-94df-dcbfbf9d10b9
deployed; hosted JS/service-worker bytes match the tested build. Authenticated
missing-edition preview returns 404/private no-store without creating data. Mac
last seen 12:07:44 UTC, collection 12:06:36, pending delivery zero, unavailable
empty. No agent change/restart.
The test server and temporary browser tab were closed. No live proposal was created
or approved. Actual PDF acceptance, investor decisions/lineage, accepted exports,
new rows and multi-source correction packages remain unfinished. This is preview
access, not completion of professional one-pager or one-click refresh acceptance.

## T207 — explicit pending renderer context (verified live)

Both brief and illustrated renderers now accept a typed pending edition only via
an explicit internal argument. Ordinary render calls retain identical HTML in
synthetic baseline comparisons. The shared valid_citations gate stays unchanged.
Each proposed claim, including direct titles/timing, carries pending provenance;
print-visible notices retain edition/report identity and deny acceptance. Readiness
separates pending paths from model-supported paths and does not borrow scenario
inputs across cases. Original report, context contents, assignment inputs and the
illustrated financial snapshot are checked before rendering. Long illustrated
claims remain omitted whole and diagnosed, while brief claims retain full wording.
Five additional synthetic tests cover both renderers, the brief's illustrated
forwarding path, stale/tampered context, financial drift, direct headlines, escaping,
readiness and whole-claim omission. Synthetic desktop and 390px mobile HTML were
visually inspected: pending labels/readiness are present, with no horizontal
page overflow. No actual PDF or investor quality acceptance is claimed.
1118 backend and 143 frontend tests, Python compilation, model policy against
75097e5 and the production build passed. Backend revision
b7a9b08f5092b8ad765a4b65d81aaeda93ac47d5 verified healthy after transient
rollout 502s. Cloudflare 6a1291d3-5ca3-4f0d-ac76-7fe8be4d2e89 deployed;
hosted JS/service-worker bytes match the tested build. Mac last seen 11:37:57
UTC, collection 11:38:12, pending delivery zero and unavailable empty. No
agent change/restart, live proposal mutation, paid call or investor approval.
No production route consumes this context yet; the next step is a private,
explicit edition preview route that locks/reloads server evidence in one transaction,
plus review UI navigation. Accepted decisions/lineage and revised exports remain
unimplemented. Original/default outputs and thesis handoffs are unchanged. Do not
broaden shared evidence rules or accept live proposals to validate this feature.

## T206 — isolated pending-preview evidence adapter (verified live)

Added onepager_edition_evidence.py as an internal, currently unused adapter.
It revalidates a server-loaded pending edition and its current original evidence,
then creates a deep-copied report and a separate explicit claim map. Corrections
carry pending_editorial provenance, edition/correction IDs and original source
metadata; they never acquire review=supported or passageMatched on the claim.
Original citations remain untouched and the shared valid_citations gate rejects
changed wording. Canonical array/escaped-key paths preserve same-case identity.
Four synthetic tests cover isolation, original retention, shared-gate exclusion,
source/report drift, tampered/accepted records and exact pointer replacement.
1113 backend and 143 frontend tests, compilation, model-policy baseline 75097e5
and the production build passed. Backend revision
37ba7197a700dbacff79048c5ae8049b69a3a4b3 verified healthy after transient
rollout 502s. Cloudflare f1cdaec9-73f7-43c2-b80b-401d3a3b35ab deployed;
hosted JS/service-worker bytes match the tested build. Mac last seen 11:04:13
UTC, collection 11:04:25, pending delivery zero, unavailable empty. No agent
change/restart, paid call or live research mutation. No new UI/print claim.
No renderer or endpoint consumes this adapter yet. Pending-preview layout, evidence
readiness, footer labels and both render entry points still require integration;
there is no new export, investor decision or source refresh in this increment.
The future server caller must lock/reload inputs in the transaction and pass the
typed context explicitly. Do not enable proposals via a client flag or broaden
shared evidence rules. T205 release receipts are also retained in this checkpoint.

## T205 — pending edition review screen (verified live)

The saved assignment infographic now includes “One-pager editions · pending review.”
Users can group 1–20 existing corrections, inspect original/proposed full wording,
claim type, reason, passage and verified original links, save an immutable pending
edition, and reload its retained comparisons. Report/assignment identity and stale
or competing paths are checked; an unchanged retry retains its submission UUID.
The screen explicitly states that acceptance and revised exports are unavailable.
No pending edition changes original exports, research or thesis state. It shows
up to 50 recent corrections and 20 editions; older navigation remains future work.
Three frontend tests cover ordered identity, bounds, stale/conflicting selections
and labels. Synthetic UI save/reload, desktop comparison and 390px mobile wrapping
were inspected using the actual component and workspace styles (no overflow).
1109 backend and 143 frontend tests, model policy baseline 75097e5 and the
production build passed. Backend e7ad825248b149c60e163d4b6b80834d5b941b8e
verified healthy; Cloudflare 93c06dde-52e9-44b5-b2e1-022828a04f8f deployed.
Hosted JS and service worker match tested bytes. Mac last seen 10:34:52 UTC,
collection 10:35:03, pending delivery zero and unavailable reads empty. No agent
change/restart. Generated bundle diff flags pre-existing Markdown hard-break
spaces in an unrelated template literal; source whitespace checks pass and the
literal was preserved. Synthetic desktop evidence: /tmp/charlie-edition-ui/desktop.png.
The harness used synthetic in-memory API responses with the pure edition validator;
T204's disposable database tests cover persistence. No production POST or paid call.
Next: explicit accepted/pending evidence adapters, transaction-safe investor decisions
and version-specific exports. Investor approval and substantive benchmark acceptance
remain open; grouping corrections is not a repaired professional one-pager.

## T204 — durable pending edition selections (verified live)

Added private/no-store GET/POST at /api/research/assignments/<id>/onepager-editions.
GET returns a baseline and up to 20 immutable pending records, marks changed
report/assignment/financial baselines stale, and never creates schema or work.
It explicitly does not reverify current original bytes or usage restrictions.
POST locks the saved assignment, report, selected corrections and eligible original
rows and uses the T203 contract with the full stored financial snapshot. Same-key
concurrent submissions persist one record; identical replay returns its identity,
changed payload conflicts, and failed validation rolls back. Stopped assignments
cannot receive new editions. Original reports, citations, corrections, assignment
outputs and theses remain unchanged. There is no acceptance endpoint, rendering,
preview or UI control yet; pending selections cannot become default exports.
1109 backend and 140 frontend tests, compilation, policy baseline 75097e5 and
the production build passed. Backend revision
55f45a923f69c25d29202cc99865e4c9a3e59002 verified healthy; Cloudflare
b79c7fde-72c6-4062-8606-e3a54462ceb5 deployed. Hosted JS and service worker
match tested bytes. Authenticated ABT/UNH preparation returned pending_only,
zero saved editions and private/no-store caching; no production POST was made.
Transient rollout 502s cleared, including a retried read-only UNH check. Mac last
seen 10:03:39 UTC, collection 10:03:20, pending delivery zero, unavailable empty.
No agent change or restart. No new UI/print acceptance is claimed.
Five disposable-PostgreSQL tests cover read-only preparation, concurrent replay,
restart, stale snapshots, restrictions, rollback, forged status and retained originals.
Next: explicit edition review navigation, renderer evidence adapters and transactional
investor decisions with separate provenance. Never accept live research for QA.

## T203 — pending-edition validation foundation (verified live)

Added a pure, inactive edition contract in onepager_edition_contract.py. It binds
an explicit ordered selection of 1–20 persisted corrections to assignment/report,
full saved financial snapshot plus vintage, frozen assignment input and parent.
It revalidates current original bytes, restrictions, passage matching and the
entire saved proposal/payload identity; duplicate paths and unrelated/stale records
fail closed. Only pending records are returned. This module has no route, database
write, renderer integration, acceptance action or provider call. Existing exports
and pending corrections are unchanged. The full saved financial snapshot must be
loaded by the future caller; assignment visualFundamentals metadata alone is not
that snapshot. Future persistence must lock and recheck inputs in its transaction.
Eight synthetic tests cover immutable originals, stale identities/financials/parent,
forged status/provenance, source changes, bounded/conflicting selections and order.
1104 backend and 140 frontend tests, Python compilation, model-policy baseline
75097e5 and the production build passed. Backend revision
69fc6fdf8125e8d8e967f31547bf9d492a657f62 is healthy after transient rollout
502s. Cloudflare ddd473b0-cc28-4cac-a91d-0bd5720bc017 deployed; hosted JS
and service worker match tested bytes. Mac last seen 09:34:12 UTC, collection
09:34:37, pending delivery zero, unavailable reads empty. No agent change or
restart, live research mutation, paid call or acceptance. No UI changed beyond
the release marker, and no new visual acceptance is claimed.
Edition persistence, investor decision UI, explicit renderer adapters, export and
real-source acceptance remain unimplemented; this is internal foundation only.

## Accepted-edition adapter checkpoint — October 12, 09:00 UTC

Traced the original-evidence gate across the brief/illustrated renderers,
readiness and assignment derivatives. Do not extend the shared valid_citations
function to accept human edits: that could promote them into unrelated outputs.
Recorded a dedicated, explicitly selected accepted-edition adapter contract in
docs/onepager-editorial-handoff.md, including stale/source/policy checks and
same-path concurrency conflicts. Existing pending corrections remain immutable;
acceptance and derived rendering are still unimplemented. This documentation
checkpoint also commits T202's verified production receipts. No live research,
correction, source or approval was written and no runtime behavior changed.

## T202 — pending source-linked corrections (verified live)

Completed stock reports can capture a proposed correction to an existing scalar
claim, with unchanged original wording, claim type, reason, frozen source passage
and locator. Server checks current report fingerprint, original bytes/hash,
source restrictions and exact normalized passage matching before insertion.
Schema and record are transactional; a submission UUID plus payload digest makes
same-key retries idempotent and changed payloads conflict. GET is read-only and
returns at most 50 recent pending records, labeling stale baselines. Original
report/citation state, thesis and one-pager exports are never updated. There is
no acceptance or apply endpoint/control. New rows, multi-source changes, reviewed
rendering, acceptance and private authoring helper remain future work. This is
a separate review record, not a corrected brief or completed phase.
Seven new tests include disposable-Postgres concurrent submission/restart,
rollback, source restrictions/hash changes, stale fingerprints, forged review
fields and original-state preservation. 1096 backend and 140 frontend tests,
compilation, model policy baseline 75097e5 and production build passed. Synthetic
UI save/reload and 390px wrapping inspected with the real component and workspace
styles; production proposals were not created for QA. Backend revision
c1f8beee8ced266a520134400fd21cb5bf46fa29 verified healthy after rollout 502s.
Cloudflare c507a651-2504-48ea-9147-d712427f729b deployed; hosted JS and SW
match tested bytes. Read-only ABT/UNH correction preparation returned 268/263
fields, 8/6 sources and zero pending records with private/no-store caching.
Mac heartbeat 08:40:35 UTC, collection 08:39:00, pending delivery zero and
unavailable reads empty. No agent changes or restart. Acceptance and application
remain unimplemented; existing exports and research are unchanged.

## Editorial handoff implementation contract — October 12, 08:00 UTC

Inspected report/source download, assignment visual rendering, current/original
view, stock-analysis UI and transactional meeting handoff. Documented the next
bounded end-to-end implementation in docs/onepager-editorial-handoff.md: immutable
proposal identity, source/hash checks, distinct human review, idempotent pending
submission, conflict-safe investor acceptance and original/preview/export
separation. The human-evidence renderer adapter remains a design requirement;
do not forge supported citations or change original research to preview edits.
No API, schema, helper, acceptance control or new runtime behavior was added.
This documentation checkpoint also records T201's already verified deployment.

## T201 — scenario evidence repair checklist (verified live)

The one-pager's collapsed Evidence readiness disclosure now lists the saved
scenario row and exact paths for eight operating, valuation and failure-condition
fields. Missing/placeholder, unmatched/stale and supported inputs are distinct.
Readiness is independent of label eligibility and page selection; no research is
changed. Existing conservative arithmetic checks run only on supported fields
from the same case. The first 20 saved numeric rows are inspected with an explicit
omitted-row count. Rejected claim bodies are not exposed. Long paths wrap and the
entire disclosure remains outside the print page. This is a repair checklist,
not new evidence, accepted valuation, or completion of the professional one-pager
phase. Source-checked editorial handoff and investor acceptance remain open.
Four additional tests cover exact field state, stale evidence, same-case arithmetic,
array/object input, bounds and no mutation. 1089 backend and 140 frontend tests
pass, along with compilation, policy baseline 75097e5 and production build.
Desktop Chrome disclosure inspected. ABT/UNH print bodies are byte-identical
after normalizing only the renderer version; no new PDF or mobile inspection.
Backend 0f3c6601c84f1f9964de52cf2de47751e2055dda verified healthy after
rollout 502s. Cloudflare 4d70eca0-ab65-4064-800b-0a342899add0 deployed.
Hosted JS/SW and both ABT/UNH v21 HTML match local bytes. No agent change
or restart, paid model run, research edit, approval or original mutation.
Activity initially timed out during rollout; read-only retry succeeded. Mac last
seen 07:37:25 UTC, collection 07:38:01, pending delivery zero, unavailable empty.

## T200 — distinguish case labels from financial claims (verified live)

The saved ABT review identified scenario names as hypotheses with no factual
claim to verify. Canonical Bear/Base/Bull names now act as structural labels
only when the same row has evidence-checked earnings/cash flow, multiple and
implied value. A label cannot promote unsupported financial inputs, combine
cases, bypass space limits or imply arithmetic verification. Arbitrary names
and duplicate canonical labels are not rescued. Existing supported-name rows
retain their behavior. No citation, review status or original report is changed.
Four added tests cover array/object rows, unknown/duplicate labels, incomplete
core fields, same-row identity, limits and no mutation. Synthetic desktop layout
inspected; fixed label/explanation separation. Existing ABT/UNH still lack
complete core inputs and correctly retain their scenario gap. This fixes a
rendering eligibility issue, not their substantive scenario deficiencies.
1085 backend and 140 frontend tests passed; Python compilation, model-policy
baseline 75097e5 and production frontend build passed. Backend
ebcbe70a9a72913fd105c3d7f8d4a1831c8ef5f9 verified healthy after rollout 502s;
Cloudflare a263573e-eb5f-47a3-bc90-3a285e7597e4 live. Hosted JS/SW and
ABT/UNH v20 renders match local bytes. Mac heartbeat 07:04:22 UTC, collection
07:04:35, pending delivery zero and unavailable reads empty. No agent change,
paid model run, source refresh, thesis approval or original mutation.

## T199 — core editorial readiness explanations (verified live)

Added read-only diagnosis for investment thesis, paired market debate and
investment view. Missing saved fields, unmatched/stale evidence, incomplete
same-row debate pairs, pairs outside the renderer's bounded rows and supported
text omitted for space have distinct repair explanations and exact claim paths.
The collapsed Evidence readiness section does not reproduce rejected claims,
change source matching, edit research or certify accuracy/materiality. Coverage
and debate gap messages now point to the diagnosis instead of implying that all
missing debates are length failures. Other section completeness, scenario
composition and source-checked editorial replacement remain outstanding.
Seven new synthetic tests cover missing/stale evidence, no cross-pairing, bounds,
placeholders, no claim-body disclosure, no input mutation and renderer integration.
Saved ABT diagnoses unmatched thesis/view and incomplete debate pairs; UNH's three
core sections render. Desktop Chrome disclosure inspected; ABT print preview
remains one A4 page with the disclosure excluded. No mobile viewport inspection
in this increment; existing wrapping and responsive styles are unchanged.
1081 backend and 140 frontend tests, touched Python compilation, model policy
against 75097e5 and production build passed. Backend revision
6847b7ce11477706e1466926e095cf4b72fa01cb healthy after transient rollout
502; Cloudflare 2ef3deb6-0628-41e5-bb29-68d6f5d1e3ed deployed. Hosted
JS/SW and live ABT/UNH v19 HTML match local bytes. Mac heartbeat 06:33:10 UTC,
collection snapshot 06:33:09, pending delivery zero, unavailable reads empty.

## Reference benchmark and evidence diagnosis — October 12, 05:56 UTC wake

Compared the user's DE reference image/calibration with saved ABT/UNH v18
structure and actual v16 Calibri PDF images. Recorded an agent-assessed acceptance
matrix in docs/onepager-acceptance-audit.md; no investor scoring or source accuracy
certification. Calibration has no source register and must not supply financial
facts. ABT has zero qualifying ownership-thesis fields and no complete same-row
debate pair, so these gaps are not print-budget failures. Both companies lack
complete supported scenario rows; do not splice isolated fields across cases or
substitute the separate pending thesis for the frozen report. Next bounded work:
structured readiness reasons (missing/unmatched, incomplete pair, over budget),
then a versioned source-checked editorial handoff. No paid calls, research writes,
approvals, runtime changes or deployments in this documentation-only increment.
All eight-phase acceptance remains open as previously recorded.

## T198 — Abbott historical earnings reconciliation context (verified live)

Verified Abbott's Jan 22, 2025 FY2024 release, annual reconciliation and notes 1–2.
Added a dated public historical supplement beside the GAAP charts and in the
source register. FY2024 diluted EPS $7.64 minus $2.97 net specified benefits equals
issuer-adjusted $4.67. Net earnings $13.402bn minus $5.202bn equals $8.200bn;
pretax/tax adjustments +$2.633bn/+$7.835bn reconcile the net adjustment. The
$7.497bn non-cash tax valuation-allowance benefit is distinguished from all net
specified items. This is issuer-defined non-GAAP, not analyst-normalized earnings.
Exact issuer/CIK, period, units, GAAP EPS/net-income and snapshot vintage gates
prevent attaching the note to mismatched/restated or pre-publication observations.
GAAP data and original research are unchanged. The public supplement is explicitly
separate from the frozen source pack, with publication/verification dates and a
SHA-256 receipt. Original public HTML retained privately, not in watched folders.
Source: https://abbott.mediaroom.com/2025-01-22-Abbott-Reports-Fourth-Quarter-and-Full-Year-2024-Results-Issues-2025-Financial-Outlook
Hash: 1b9f0f60323ba6095222a988a2cb1a99d56e433e96995766f2b495459378923a
Saved ABT v18 still prints on one A4 page in native Chrome/Calibri. UNH does not
receive Abbott context. Four tests cover arithmetic, issuer/vintage/period/units,
changed observations, duplicate rows and no mutation. Remaining years, historical
valuation, ABT thesis/monitor evidence gaps and eight-phase acceptance stay open.
1074 backend and 140 frontend tests, touched Python compilation, model policy
against 75097e5 and production build passed. Backend revision
42144caa496f4f78ca1b709a78e760f2f63563fa healthy after transient rollout
502s; Cloudflare 288770d8-1480-4458-9c68-8bcf6eb31ff2 live. Hosted JS/SW
and ABT/UNH v18 match local bytes; ABT has context and UNH does not. Mac
heartbeat 05:33:19 UTC; pending collection delivery zero and no unavailable reads.

## T197 — explicit saved-scenario arithmetic (verified live)

Added a conservative read-only P/E and price-return check to supported scenario
rows. Same fiscal year, GAAP/adjusted diluted-EPS basis, currency, positive inputs,
real reference date and explicit multiple rationale are required. Decimal arithmetic
compares saved output precision; missing/narrative/range/EV/FCF/total-return inputs
remain unverified. Mismatched results require review and are never overwritten.
Passing arithmetic does not certify assumptions, source accuracy or price targets.
See docs/scenario-arithmetic.md for the deliberately strict supported grammar.
No research schema/prompt, model, cap, source pack or active job changed. Existing
ABT/UNH have no supported scenario rows, so this does not fill their valuation gaps.
Synthetic valid/mismatch/missing rows inspected in native Chrome; saved ABT/UNH
privately re-rendered. Seven new tests cover arithmetic, metadata, precision,
invalid/oversized input, no mutation and supported-only renderer integration.
1070 safe backend tests, 140 frontend tests, touched Python compilation, model
policy against 75097e5 and production build passed. Predeploy bounded Activity
showed no running/processing items or unavailable reads; historical buckets are
truncated. Backend b58ed05e3b3d94c42625fb589dbea14fe1b78610 healthy after
transient rollout 502. Cloudflare 934e4110-f066-434a-8034-d5ebd33aa541 live.
Hosted JS/SW and ABT/UNH v17 renders verified as exact local bytes. Mac heartbeat
05:03:48 UTC; pending delivery zero and no unavailable reads. Later typed authoring
and broader valuation/reconciliation remain outstanding.

## T196 — Calibri export verification (verified live)

Installed existing Office Calibri regular/bold/italic/bold-italic copies in this
Mac user's Library/Fonts without replacing originals or publishing font binaries.
Chrome picks them up without a restart. Calibri metrics exposed a final-reference
spill in UNH v15; v16 tightens section/header/footer spacing only. Research text,
type sizes and source references remain intact. Native Chrome A4 PDF exports of
saved ABT and UNH v16 each contain one page. Poppler confirms every embedded font
is Calibri; rendered page images inspected with conclusion and references intact.
Private QA files: Charlie-private/onepager-review-20261012/*-v16-Calibri.pdf.
This resolves font availability on this Mac only: other devices still require
Calibri or a licensed export solution. No proprietary fonts shipped in repository
or hosted assets. ABT evidence gaps, valuation/reconciliation work, real-source
benchmarks and all eight-phase acceptance remain open. No paid research or
investor approvals performed. Browser test tab closed; existing tabs preserved.
Validation: 1063 backend and 140 frontend tests, Python compilation, model policy
against 75097e5 and production build passed. Backend revision
6e44f6668678b74a58c06562c546676f21b3e7c4 healthy after transient rollout
502; Cloudflare c0dc0235-5027-45b1-bf84-1d0052aad0fc live. Hosted JS/SW
and live ABT/UNH v16 renders match verified local bytes. Mac heartbeat
04:32:30 UTC; no pending collection delivery or unavailable reads.

## T195 — print pagination correction (verified live)

Native Chrome Save as PDF exposed a two-page UNH v14 map with the conclusion
and references spilling. V15 places an absent-scenario notice beside valuation,
retaining real scenario tables at full width, and tightens print whitespace,
peer-table padding and chart height. Existing research text and source references
are preserved; body text sizes are unchanged. Actual private Chrome PDF exports
of saved ABT and UNH v15 are one A4 page each, with browser headers/footers off.
Poppler page images inspected: conclusion and references present, no clipping.
This does not prove all report lengths fit. Both PDFs currently embed Helvetica
fallback despite the Calibri CSS: Calibri is bundled inside installed Office but
is not available to Chrome. Font portability remains an acceptance gap; do not
redistribute proprietary font files in public assets. ABT research gaps remain
explicit; this is not the separate pending rebuilt thesis or professional acceptance.
Validation: 1063 safe backend tests, 140 frontend tests, touched Python compile,
model policy against 75097e5 and production build passed. Bounded predeploy
Activity returned no running/processing items and no unavailable reads; historical
buckets are truncated. No agent code or paid research changed.
Backend 8b86396b643dc1f7f401414ed9b6c7d323291271 verified healthy after
transient rollout 502; Cloudflare e0a5c6a9-040f-4b15-af93-d57fea951848 live.
Hosted JS/SW and live ABT/UNH v15 renders exactly match verified local files.
Mac heartbeat 04:06:21 UTC, collection 04:06:26, pending delivery zero and
no unavailable reads. Private PDFs remain QA artifacts, not accepted deliverables.
Next: Calibri-capable export, valuation inputs/arithmetic and historical/GAAP
reconciliation; continue the eight-phase plan without marking it complete.

## T194 — saved scenario assumptions in investment maps (verified live)

The renderer exposes up to three supported scenario rows with operating conditions,
growth, margins, earnings/cash flow, multiple, implied value, dated return reference
and failure conditions. Every field retains its citation, complete qualifications
and row identity; missing/unreviewed fields remain missing. No derived numbers,
rebasing to current prices, probabilities or arithmetic certification. Mechanical
sensitivity remains separate. Future research instructions require valuation year,
accounting basis, units, multiple rationale and dated return reference within the
existing schema; no pinned jobs, models or caps changed. Full-width desktop table
and stacked 390px phone rows inspected with synthetic data, no horizontal overflow.
Saved ABT/UNH re-rendered privately: neither has supported scenario rows, so both
show the explicit gap. No paid research or thesis change. 1063 backend and 140
frontend tests, touched Python compilation, model policy against 75097e5 and
production build passed. Backend e4faa84eedf05f2e9f26e33880eb9735a88a211f
verified healthy after transient rollout 502s; Cloudflare
d477d224-9a11-4073-8c8a-e03d4eedf514 deployed. Hosted JS/SW exact bytes
and both production ABT/UNH v14 renders verified against private artifacts. Mac
heartbeat 03:32:57 UTC; no unavailable reads or pending collection delivery.
Predeploy bounded Activity showed no running/processing items; historical
buckets are truncated and not an exhaustive inventory. This is a provenance
and display increment, not a calibrated valuation engine or professional acceptance.
Next: typed, dated valuation inputs with arithmetic validation, historical valuation,
GAAP/adjusted reconciliation and actual print/export benchmark. All eight phases
remain incomplete; legacy WELL outcome remains unverified.

## T193 — structured risk and monitoring evidence (verified live)

One-pagers now render supported risk rows with their own earnings impact, timing
and early warning, plus supported KPI rows with baseline, positive/negative
thresholds and investment relevance. Each field retains its own citation and
qualifiers. Missing fields remain explicit; unanchored fields cannot migrate to
another row. Whole overlong rows are omitted with disclosure. Historical
strengtheners/breakers no longer masquerade as forward signposts. Where no KPI
metric is supported, paired debate resolution evidence remains separately labeled
and does not claim quantified thresholds or timing. No new periods are inferred.
1060 backend and 140 frontend tests, touched Python compilation, model policy
against 75097e5 and production build passed. Synthetic structured rows inspected
at desktop and 390px, with no horizontal overflow. Saved ABT/UNH privately
re-rendered without model calls or research changes. ABT loses the misleading
historical monitoring fallback and remains incomplete; this does not repair its
missing thesis or certify either example. Print pagination still unverified.
Backend d47b477e02afc41507c9f7e8a1935a3803f31498 verified healthy after
transient rollout 502s. Cloudflare df7ae8c0-ae49-4e3e-aaf3-54c1a8c1110f
deployed; hosted JS/SW exact bytes verified. Production ABT and UNH v13
artifacts exactly match private re-renders. Mac heartbeat 03:04:14 UTC,
no unavailable Activity reads or pending collection delivery. Predeploy bounded
Activity contained no running/processing stages; truncated buckets remain
non-exhaustive. Next: explicit scenario valuation inputs,
historical valuation and financial reconciliation, then real-source benchmark
acceptance. Phase-1 legacy WELL outcome and later phases remain open.

## T192 — preserve research order in one-pager selection (verified live)

Extracted stable evidence ordering into onepager_editorial.py. Removed numeric
and citation-count scoring, managed-care keyword bonuses, inferred topic labels,
and topic-based suppression. Supported claims follow their saved category and
numeric row order; whole-claim page limits and missing-section notices remain.
Duplicate suppression compares repeated text within a selected section, so a
headline or debate cannot consume the only thesis statement. Distinct arguments
sharing sector terminology remain eligible. Debates stay paired with their own
resolution evidence; unsupported topic labels become neutral numbered debates.
Future generation instructions request financially material arguments first;
existing report order is not retroactively certified as materiality ranking.
No model, token cap, saved research or source pack changed, and no paid QA.
1056 backend and 140 frontend tests, touched-module compilation, policy against
75097e5 and frontend build passed. Saved ABT/UNH were re-rendered privately;
ABT remains incomplete. UNH desktop and 390px layouts inspected; no horizontal
overflow. Neither example is professionally accepted. Backend revision
5dfccd9c691e9f6b03c202e165225f545774b02f verified healthy after transient
rollout 502s. Cloudflare 7cf33567-2b5d-4006-bb12-7fd0ff582b7d deployed;
hosted JS/SW exact bytes verified. Both production ABT/UNH v12 artifacts match
the privately inspected HTML exactly. Mac heartbeat 02:33:19 UTC; no unavailable
Activity reads. Predeploy bounded Activity showed no non-queued active work;
truncated historical buckets were not treated as an exhaustive inventory.
Next: structured prospective signposts and scenario valuation inputs with
provenance; do not confuse this deterministic selection fix with expert judgment.

## Phase-1 process-interruption acceptance — October 12, 02:00 UTC

Added real subprocess death/concurrency tests around the production reservation,
recovery and worker-presence functions, using only a disposable PostgreSQL cluster
on a private temporary Unix socket. No Flask application startup, model adapter,
production DB access or paid call. Four checks cover: death before reservation
recovers exactly once with frozen inputs; hard death after reservation leaves the
attempt uncertain and its batch slot occupied while unrelated work can recover;
four independent claimants cannot reserve the same run twice; separate processes
share the batch concurrency limit. Reporter expiry is evaluated using a supplied
future clock, not a 90-second sleep, and cannot authorize takeover. This extends
the earlier thread-level tests to OS process interruption; it is still synthetic
provider evidence, not proof of live provider cancellation or bill reconciliation.
All 1054 backend tests passed, including these four subprocess checks; the new
test module compiles. No runtime behavior or frontend assets changed; this is a
test/documentation increment. Commit 65c74d31a2b49282d44df4eb9900a73231fb0fa4
pushed; automatic Render rollout verified healthy after transient 502s. Hosted
JS/SW remain exact T191 bytes; no Cloudflare redeploy was needed. Mac heartbeat
02:03:57 UTC; no unavailable Activity reads. Keep T191 as the product release.
The legacy WELL
provider outcome and user-driven end-to-end live acceptance remain unresolved.
Continue the one-pager editorial/valuation work recorded in the saved-output audit;
do not replace missing investment content with more success labels.

## T191 — incomplete one-pager detection (verified live)

Read-only review of the saved ABT and UNH investment maps exposed a material
ABT quality failure: missing company description, paired debate and investment
view, plus a short operating driver substituted for the investment thesis.
The renderer now selects thesis claims only from investment-thesis evidence;
it flags missing rendered core sections prominently, including supported claims
omitted by length constraints. This is a completeness check, not a semantic
quality score or certification. Both saved examples were re-rendered privately
without changing original artifacts, research, sources or accepted theses.
ABT correctly shows four missing sections; UNH has the core sections, but its
professional-quality acceptance is still open. The separate rebuilt ABT thesis
proposal is unaffected. Phone inspection found and fixed ticker wrapping.
1050 backend and 140 frontend tests passed; compilation, model policy against
75097e5 and production build passed. Saved ABT and UNH evidence re-rendered
without new model calls. ABT notice inspected at desktop and 390px; no horizontal
overflow, ticker remains unbroken. Predeploy bounded Activity snapshot has no
running/processing/analyzing items and no unavailable reads; truncated history
is not proof that all jobs were inventoried. Backend revision
7b9afd85cc4f293314768bf3e2265f9f014d8cae verified healthy after transient
rollout 502s. Cloudflare aaf301c3-1fb4-473e-a8f1-1d7a75be292d deployed;
hosted JS/SW exact bytes verified. Production ABT v11 HTML matches the privately
inspected artifact exactly. Mac heartbeat 01:37:26 UTC, no unavailable reads.
See docs/onepager-acceptance-audit.md for the next content/valuation work.
No paid research or approvals. Phase 1 operational acceptance remains open;
this bounded phase-2 audit does not mark the eight priorities complete.

## T190 — run-scoped worker reporting

New analyst attempts persist reporting metadata tied to their reserved execution
identity before model work; a daemon refreshes it every 30 seconds and records
worker exit after cleanup. Activity inspection reports recent, expired (90s), exited,
unknown or unavailable reporter evidence. It never claims provider completion,
cancellation or safe replay. Old runs remain unknown; identities must match saved
reservations. UTC timestamps are explicit independent of DB session timezone.
A blocked final heartbeat is allowed to expire instead of overwriting an exit;
cleanup failures preserve the saved job result. No lease takeover or batch-slot
release is introduced. Synthetic persistence/expiry/identity/failure checks added.
1048 backend and 140 frontend tests passed; compilation, model policy against
75097e5 and production build passed. Predeploy Activity reports no freshly
progressing non-queued jobs in its bounded snapshot. Backend revision
874d9338e6c4a908cbd8031a659ca6e4979c4605 verified healthy after transient
rollout 502s. Cloudflare 15afc2b4-3fd0-40a7-b916-9625e39b15f2 deployed;
hosted JS/SW exact bytes verified. Mac 01:02:50 UTC, no unavailable Activity reads.
Legacy WELL correctly returns unknown worker presence and attention; its history
was not backfilled. Reporter persistence/expiry is synthetically verified, not a
live paid research benchmark. No paid run or approval. Phase-1 live interruption
acceptance and legacy provider outcome remain open; next prioritize acceptance
evidence and professional one-pager benchmarks rather than more status copy.

## T189 — audited unresolved legacy disposition

Old legacy analyst runs can record an unresolved outcome from Activity's saved-run
inspection. The server locks and rechecks the row: no recent progress within one
day, saved output/completion, execution enrollment, reservation, graph callback or
batch enrollment may be present. Malformed dates/logs fail closed. The action
only sets attention and appends a fixed audit event; it preserves previous logs,
outputs and completion fields and never retries or declares cancellation/success.
Concurrent repeats are idempotent. Reserved batches cannot use this action or
release a slot. Unknown provider billing remains unknown. 1045 backend and 140
frontend tests passed; compilation, model policy against 75097e5 and production
build passed. Final backend 1740861e7431c2e8784fb609e38d63073712833b verified
healthy after automatic Render deployments; transient 502s cleared. Cloudflare
3f84a317-a1c0-409e-9034-e6ee91dcafb6; hosted JS/SW exact bytes verified.
Applied the supported action to legacy WELL through its exact saved-run UI:
status attention, 23 logs (22 prior plus audit event), no report/decision/completion,
provider outcome still unverified. Activity retains it in Needs you even when old;
regression covers that visibility. Mac heartbeat 00:34:13 UTC, no unavailable reads.
Desktop outcome inspected and saved at /tmp/charlie-t189-well-unresolved.png.
At 390px, DOM/body/scroll width all 390 and inspection action remains available;
mobile screenshot capture was unusually scaled, so full visual mobile acceptance
is not claimed. No paid call, approval, cancellation or replay. The legacy run is
honestly dispositioned, but provider billing/outcome reconciliation remains open.
Next: durable worker-liveness evidence for new analyst runs, then professional
one-pager acceptance; the eight-phase plan is not complete.

## Phase-1 evidence review — October 11, 23:57 UTC

Read the exact legacy WELL saved-run record through the authenticated inspection
API. It was created March 23 at 18:53:21 UTC using anthropic/claude-sonnet-4-6;
system logs record graph initialization and analysis entry. Its 22 entries end
at 18:57:12 UTC. No saved report, decision, completion, execution reservation or
provider response reference exists. This proves recorded execution entry and
missing saved output, not absence of provider charges or a safe replay. Neither
current backend health nor repeated empty collection claims resolve it.

Next bounded recovery work: add a supported, audited disposition for old legacy
runs whose provider outcome cannot be established. Preserve originals/logs and
unknown outcome, avoid claiming success/cancellation, never create a replacement
attempt or release a reserved batch slot. Distinguish this disposition from proof
of provider reconciliation. New executions also need durable worker-liveness
evidence so future interruptions can be distinguished from stale database labels.
Do not manufacture retrospective ownership for legacy rows. Phase 1 acceptance
remains open; do not keep adding response-display changes as a substitute for
these operational controls. One-pager code/reference inventory was inspected
read-only; no source benchmark or investor acceptance was performed this wake.

## T188 — provider response references for recovery

Activity's recovery evidence now exposes up to 20 distinct saved provider response
references from valid returned graph callbacks. Persisted entries are revalidated
against the response-ID allowlist and valid execution/call UUIDs before exposure;
malformed entries and starts/errors cannot introduce references. Truncation is
explicit. These are reconciliation aids, not billing receipts, whole-job success
or permission to replay. Legacy WELL has no callback evidence, so this does not
resolve its outcome. No provider calls, approvals or job-state mutations occur.
Focused synthetic tests cover deduplication, malformed metadata and bounded output.
1041 backend and 140 frontend tests passed; Python compilation, model policy
against 75097e5 and production build passed. Backend revision
bec3a69295a6ccdcf24abbdbe97ae6dd4374e2a7 verified healthy; Render automatic
deploy observed. Transient 502s during switchover cleared. Cloudflare version
e16ff17a-c903-4a86-9fa2-03f5b3e380af; hosted JS/SW exact bytes verified.
Mac heartbeat 23:31:22 UTC, no unavailable Activity reads. Live WELL evidence
returns empty references and unchanged unverified outcome. UI reference rendering
validated by production build; populated-reference behavior has synthetic evidence
only, not a live provider call. No paid QA.
Continue phase-1 reconciliation.

## T187 — separate proposal association (verified live)

Blocked thesis assignments can associate an existing pending proposal through
POST /api/research/assignments/<id>/link-thesis with draftId. This uses the worker
lock, checks company, current pending status, equal frozen thesis baseline and
at least one exact original hash in common. Repeated same links are idempotent;
a different replacement is rejected. The association preserves the original
failed status, artifacts, source records and error. It neither approves a thesis
nor claims automated quality passed. The assignment UI links to the exact rebuilt
proposal and discloses partial source overlap and possible supplemental evidence.
Synthetic tests cover stale/company/baseline/source rejection, replay, preserved
research, no new calls and pending approval. 1040 backend tests and 140 frontend
tests passed; Python compile, model policy against fcc2598 and build passed.
Backend 2b4074c568c003670b8bfbbd54a4c98ead09e0ba verified healthy. This push
triggered Render Auto-Deploy normally; the earlier missing trigger cause remains
unconfirmed. Cloudflare 9f0985b3-4238-4e9e-9c54-a8f2c7adaaa3 and hosted JS/SW
exact bytes verified. ABT proposal 39df605a-c074-4f8f-8f2d-03e6e05a2807 linked to
parent f9e363e4-597e-4394-b9d4-36d83def935f: 4 of 8 original hashes shared,
10 source-register entries, original attention status retained. Live UI exact link
verified at normal and 390px phone width; no horizontal text overflow observed.
Screenshot /tmp/charlie-t187-abt-linked.png. No approval or paid replay.
Legacy WELL abe12953-4353-44e8-b7e9-00f81920d5cd remains unverified: March 23
running record, no saved report/decision, no reservation or provider callback
receipt. Do not infer no charge or replay it. Continue supported interrupted-job
reconciliation and one-pager acceptance work; all eight phases remain unfinished.

## T186 — exhausted thesis recovery guidance (verified live)

Assignments inspect the saved report on read to identify an already-used quality
revision. The review screen then offers Import & review thesis rather than another
Resume action. Newly exhausted proposals record explicit manual-review guidance;
legacy Activity alerts point to available recovery options without promising a
paid retry. Originals, outputs and accepted theses remain untouched. Read-only
inspection dispatches no research. Regression covers both legacy saved errors and
new exhausted reports. 1038 backend tests, 140 frontend tests, plus 15 focused
Activity tests after the final alert-copy change passed; compile, model-policy gate
and frontend build passed. Source pushed as 557dc3e. Cloudflare deployed
88994d1f-0af9-485f-a388-79cdac8725b9; hosted JS/SW exact bytes verified.
Backend revision 557dc3edf79be4cbeb8a4e6ad18681bc9afcdf04 verified healthy
October 11 at 23:00 UTC. After user signed in via Chrome Incognito, Render
history showed no deployment attempt for the pushed T186 commit. Manually deployed
that existing tested commit as dep-db618o59fdbs73eeh48g; build/deploy succeeded in
2m00s. A transient 502 during switchover cleared. Missing automatic trigger cause
remains unconfirmed; do not describe it as a failed build or proven webhook issue.
Authenticated ABT read confirms exhausted-retry/manual-review guidance, preserved
visual and stock-summary artifacts and eight source records. Hosted app JS and
service worker exactly match local release bytes. Mac heartbeat fresh; Activity
reads available. No paid research retry or thesis approval was performed.
Screenshot: /tmp/charlie-t186-render-live.png. Resume the phase-1 backlog; Render
sign-in is no longer the deployment blocker.

## T185 — protect blocked-job alert visibility (verified)

Activity selection now uses independent 200-row attention and active budgets,
with explicit truncation categories; global alerts warn on incomplete blocked
research coverage. Synthetic regression covers >200 active jobs plus blocked
assignment visibility and >200 blocked jobs. 1037 backend / 140 frontend tests
and frontend build passed. Python compile and unchanged model-policy gate against fcc2598 also passed.
Backend revision 423147a86d09f4cdff9051d7ebf42fdebe4feb26 verified healthy.
Hosted JS and service worker exactly match local release bytes; Cloudflare version
f8c27660-e1e1-4ddc-9686-04ebe28e3949. Live authenticated Activity now includes
ABT parent f9e363e4-597e-4394-b9d4-36d83def935f as attention. Both research
budgets are currently capped, so the frontend also warns of incomplete coverage.
Mac heartbeat fresh, no unavailable Activity reads. No research retries or approvals.
This is bounded visibility, not proof all eight workflow phases are complete.
Three release markers set T185. Preserve unrelated dirty state.

Historical audit of the first assistant proposal (superseded below): audit found
it does not: 4 pillars comply with 3–5, but 17 signposts exceed 4–6 and 16 risks
exceed 3–5. Descriptions/summary exceed sentence limits; named broker narrative
also conflicts with native style rules. Existing import schema validates only
structural compatibility (1–100 items), not the product writing contract. The
assistant incorrectly preserved the bloated baseline instead of consolidating.
User informed candidly. Proposal remains pending, unapproved. It is a cautious
selected-page evidence update, not an objectively strong finished investment
thesis; measurable signposts, prioritization, valuation/variant-view analysis and
broader source coverage remain weaknesses. Do not describe it as compliant.

## ABT rebuilt proposal — October 11, 2026

User-authorized rebuild saved separately as pending draft
39df605a-c074-4f8f-8f2d-03e6e05a2807; read-back verified pending, non-stale,
matching fingerprint and 4 pillars / 6 signposts / 5 risks. Selection prioritizes
economic materiality, measurable monitoring and explicit valuation assumptions.
Selected originals and public primary sources were reviewed; source references,
arithmetic, unchanged original hashes and a three-page Calibri PDF were checked.
Private files remain outside the repository. This is investor-review work, not
approval or proof of future returns. The old bloated draft remains preserved.
The automated parent remains attention: the separate draft does not retroactively
pass its source gate or complete the parent. Do not launch another paid retry.

## Collection monitor check — October 11, 20:53 UTC

Null local claim; no Chrome opened. Authenticated Activity reports zero pending
collection commands, fresh Mac/collection snapshots, no unavailable reads.
Direct assignment reads confirm unchanged ABT attention and separately prepared
thesis draft pending/non-stale; UNH complete and SYK known collection attention.
No retry or dispatch sent.

New alerting defect: Activity reports Research workflows truncation and omits
ABT's blocked parent from its items, although /api/research/assignments returns
it. The global research banner derives alerts exclusively from Activity items,
so the new banner cannot reliably surface this blocked assignment. Activity SQL
sorts attention rows after other nonterminal rows before the 200-row limit.
Recovery priority for implementation continuation: fetch bounded blocked jobs
independently of other active/history rows (or use an explicit alert endpoint),
expose incomplete alert coverage, and add a synthetic >200-active-row regression.
Do not describe T183 global alerts as fully reliable until this is corrected.
Five-minute worker checks read assignments directly and still detect this state.

## ABT recovery outcome after T184

The bounded quality attempt completed all 12 stages but again produced no
supported thesis statement. Its quotation matched; review rejected compound
wording (notably double-digit CRM versus Q2 9.5%). Direct original inspection
also showed the reviewer overclaimed that 8.5% did not describe Medical Devices:
the Q2 transcript explicitly says both Medical Devices and cardiovascular 8.5%.
Do not treat model review as infallible or relax source gates to force completion.
Parent f9e363e4-597e-4394-b9d4-36d83def935f remains attention, with saved summary
and visual verified HTTP 200 (9,335 and 60,837 bytes respectively). No further
paid retry was sent. Automatic workflow acceptance remains incomplete.

Used the documented thesis_handoff prepare/read/submit workflow for a separate
assistant-authored proposal. Eight originals were selectively reviewed (physical
page coverage recorded in the private package); 41 other inventoried originals
were not reviewed. Updated four pillars, 17 signposts and 16 risks, retaining
legacy identities and exposing unverified assumptions. No investor approval.
Draft 0d9203a9-f590-4e0d-854b-3e310bd1dcf3 verified pending, stale=false, eight
source-register records; fingerprint read-back matched. Review destination:
`?thesisDraft=0d9203a9-f590-4e0d-854b-3e310bd1dcf3#view=thesisimports&ticker=ABT`.
Private workspace is outside repository and watched source folders. No private
research or source bodies belong in commits. This draft does not relabel the
failed automated report as reviewed/supported and is not yet linked in the parent.

Next recovery improvements: show exhausted quality repair as a specific manual
review path (current parent error still suggests retry although the server blocks
another quality revision); link a matching separately prepared proposal to its
originating assignment with baseline/source checks; improve author/reviewer
claim granularity and review disagreement handling using synthetic fixtures.
Do not purchase another ABT call for QA or automatically approve the saved draft.

## T184 — bounded thesis quality recovery

ABT finished all 12 T183 stages and saved its summary and visual. Parent remains
attention because neither proposed thesis entry passed the source gate: one
unmatched quotation and one compound claim containing unsupported market-size
language. Existing investor thesis remains unchanged.

Explicit thesis repair now archives only report-0/review-0 returned receipts so
the rejected responses cannot be silently reused. Prior report/citations remain
in qualityRevisions; other ten stages stay saved. The author receives bounded,
untrusted review feedback and instructions to produce one narrow conditional
claim from an exact contiguous source passage. Source review remains mandatory.
The two-call quality revision has a separate one-revision limit; original provider
failure retry limits remain unchanged. Repeated quality repair is refused.

Validation: 1036 safe backend tests, 139 frontend tests, Python compile, production
build and unchanged model policy gate against fcc2598. No paid QA. Deployment
verified at backend 009dece1b5b3c500d11eb9e237642d2dfc122a59; hosted JS/SW
exact bytes matched. Cloudflare 61bac0fa-580a-4630-a42d-6df28fbb238f.
Sent exactly one parent resume, HTTP 200. ABT is running report-0 with ten
completed stages, qualityRetries report-0/review-0, one prior quality revision
retained. Subsequent final outcome is recorded above; the run has settled.

## T183 — quarantine bad citations and surface research failures

User reported repeated ABT stalls and missing proactive alerts. ABT report-4 has
one citation with a source ID absent from the frozen pack. Stock validation now
retains the draft while excluding unknown references and forcing that field to
needs_review/passageMatched=false. Rejected source IDs remain diagnostic metadata;
no guessed attribution or invented source is inserted. Later model review cannot
promote it, and its unresolved attribution stays in the review reason. Shape,
field paths, evidence types/bounds and other validation remain enforced. Other
research workflows retain their existing strict source behavior.

A global research-failure banner polls Activity every 30 seconds while Charlie
is mounted, including first-load failures, with exact saved-work navigation.
Parent/child failures group by company; session dismissal does not suppress a
changed error. Failed/partial Activity refresh is explicit. This is an in-app
banner, not OS push or an always-on offline delivery guarantee. Background
AlphaSense/downstream checking is now every five minutes, with explicit checks
of active/recently blocked parent assignments even when collection is finished.
Mac/Codex availability remains required; unchanged failures remain deduplicated.

Validation: 1035 safe backend tests and 139 frontend tests, Python compile,
production build and unchanged model gate passed. Synthetic CUA desktop/390px
banner inspection passed. No paid QA. Backend fcc2598339a826fdfe3bbe3203f6abf61aa76bac
verified healthy; hosted T183 JS/service worker match exact bytes. Cloudflare
version e23095fe-a9da-4be9-9182-1023e5deaf64. Sent one parent resume (no retry
acknowledgment needed, parsed report-4 already saved), HTTP 200. ABT now has nine
completed stages and is in review-4. The affected /catalysts/2/window citation
is explicitly needs_review, passageMatched=false, with the unknown-source reason.
Do not repeat resume or deploy while active; final outputs remain pending.
This verification note stays local until research settles.

## T182 — lossless report-envelope recovery

Diagnosed ABT report-1's preserved return privately: one premature root closing
brace before citations. Removing only that brace parses both report sections and
all 39 citations without editing their values. New stock-report fallback handles
only this exact envelope defect, rejects duplicate keys/truncation/extra fields,
and retains the original receipt. It does not synthesize missing research.
Explicit recoverSavedResponse on a saved known-failure resume validates the
recovered report against its frozen sources/schema before queuing; stores parsed
response for existing stage reuse and records the repair method. No paid call for
the recovered report stage. Remaining review/research stages retain normal budget,
source, model/version, worker ownership and investor-review checks. Assignment
resume forwards this explicit option; plain resume behavior is unchanged.

Focused isolated PostgreSQL checks passed for stage reuse/no repurchase and
unrepairable rejection without dispatch. 1034 safe backend tests and 137 frontend
tests passed; touched Python compile, build and unchanged model gate passed.
Backend f72a861fcd8570364e1aa4dd1d77a01911825ca2 verified healthy; hosted T182
JS/service-worker bytes match, CSS unchanged. Cloudflare version
113ca9e5-182e-4862-a82b-2eff41b77e5e. Mac healthy at 20:15:54 UTC.
Sent exactly one parent-assignment resume with recoverSavedResponse=true, HTTP
200. Live ABT state now records the repair and completed report-1; review-1 is
running. Ledger report-1 entries remain the two earlier returns at 20:01:26 and
20:02:57 UTC (estimated $0.4234/$0.4115), not new report-stage calls from recovery.
Full outputs remain pending. Do not repeat resume or deploy while active. This
verification note is intentionally local to avoid interrupting the live job.

## T181 — private saved-response inspection

Explicit per-run/per-stage response download returns only saved text and
allowlisted diagnostic metadata as an attachment. JSON MIME, nosniff, sandbox
CSP and private/no-store prevent treating model text as executable content or
shared cached content. Missing runs/stages return 404. No key lookup, worker
wakeup, model call or research mutation. Normal run APIs still redact raw text.
This enables diagnosis of preserved malformed responses without a paid retry.
T181 verified: 1030 backend / 137 frontend tests; compile/build/model gate passed.
Backend 98d85b2a4799f5e756938cad160090b516018746 healthy; hosted JS/SW exact bytes,
CSS unchanged. Cloudflare a5f9779a-7f48-44b9-98ee-a74d43c3f75b. Private ABT
download succeeded; raw response saved outside repo and not logged.

ABT at 20:07 UTC is attention on report-1 with invalid JSON; report-0/review-0
remain complete. Its current failed return is preserved (13,709 characters,
end_turn); one archived response is also reported. No resume was issued this wake.
Inspect the saved response before any retry. Phase 1 is still incomplete.

## T180 — read-only research usage reconciliation

Each company/committee/stock research run now has a no-store GET usage-evidence
endpoint. It verifies the saved run exists and reads at most 100 accounting
records for that exact run ID and feature, with explicit truncation. Only stage,
provider/model, time, token counts, attempt and estimated cost are returned; no
raw detail/source text or credentials. This does not contact a model, wake a
worker, approve research or infer a safe retry from missing records. Charlie's
ledger is not a provider invoice. Added to investigate the user's stalled ABT
assignment; its first response was not preserved by the older release.

Validation: 1029 safe backend tests, 137 frontend tests, touched Python compile,
production build and unchanged model gate against 67aa5c3 passed. Route tests
verify missing-run 404, no-store and no worker start; query tests verify exact
job/feature binding and field allowlisting. No paid QA. Backend revision
5ea4f92c0594f2fef30215f3d028e5a01cb60c4c verified healthy; hosted T180 JS/service
worker bytes match; CSS unchanged from T179. Cloudflare version
33c00234-fc73-4cc2-ae9f-fac28398a4df. Mac healthy at 19:56:34 UTC.

ABT user explicitly requested continued work after retry-cost disclosure. Read-only
ledger reconciliation found one report-0 returned usage record at 18:37:59 UTC:
Opus 4.6, 57,575 input / 6,077 output tokens, estimated $0.4398. Together with the
saved parser failure this establishes a returned failed attempt, not a live call;
it is not a provider invoice. Sent exactly one acknowledged resume to the parent
assignment at about 19:56 UTC, HTTP 200. Report f94fc11b-0bbc-561a-a340-99de8684e5dc
has now saved report-0 (15,375 characters, end_turn) and advanced to review-0
using the same eight sources. First stage recovery is verified; the full report
and requested outputs are not complete. Do not repeat resume. Do not
push/deploy while this user-authorized research is active. This handoff update is
intentionally local pending the live outcome; verify stages/artifacts before
claiming success. Phase 1 remains incomplete.

## T179 — visible saved-response recovery evidence

Stock Analysis, Deep Research and committee review show a shared recovery panel
for attention/cancelled jobs. It distinguishes unknown outcomes, preserved
interrupted returns and known failed responses requiring a new paid attempt.
Public diagnostics include stage, character count, SHA-256 fingerprint and an
allowlisted stop reason; response text and arbitrary provider fields stay private.
No resume semantics, approvals, models or paid dispatch behavior changed.
Older ABT remains unresolved: this cannot recover its unsaved response. Active
workers are not labeled interrupted. Archived count is responses, not all calls.

Validation: 1026 safe backend tests and 137 frontend tests passed; touched Python
compiled and unchanged model gate against 64facec passed. Production build passed.
Synthetic desktop and 390px iframe browser inspection verified disclosure content
and readable wrapping; fixed mobile counter spacing. No paid QA. Pre-release
Activity showed only the legacy unresolved WELL run marked running. Backend
cd133500ec82c1a1254223ab2f89c545789cf390 is healthy; hosted T179 JS/CSS/service-worker
bytes match the build. Cloudflare version 1e40eefb-d613-41a3-8a97-5fa516839e35.
Authenticated ABT read verified the new empty receipt diagnostics, private-state
redaction and unchanged attention state. Mac heartbeat fresh at 19:41:54 UTC; no
agent change/restart. Phase 1 remains incomplete: local parse-recovery controls
and evidence-based reconciliation of legacy attempts remain outstanding.

## T178 — preserve returned research before parsing

Company/committee/stock research now saves returned text and stop reason privately
in the owned job state before JSON parsing or usage logging. The callback follows
the generator checkpoint state, preserving earlier stages and the in-flight marker.
Only these two allowlisted response fields are stored; headers/credentials are
excluded. Normal read APIs redact raw and archived returns and expose only saved
stage names/counts. Existing source, budget, owner and version checks still apply.

JSON parser ValueErrors are classified as known returned-response failures instead
of network uncertainty. A saved return can be reused locally after an explicitly
resumed interrupted attempt; explicit retry of a known failed response archives
its prior return before a new paid call. No automatic retry, larger token cap or
model change was introduced. Persistence failure stops processing. ABT's earlier
response was never saved, so this release cannot recover it retroactively; do not
retry its unresolved reservation without usage review and user acknowledgment.

Validation: 1025 safe backend and 134 frontend tests, Python compile, production
build and unchanged model-policy gate against 63c25b2 passed. Synthetic tests
exercise the actual call function, failed parsing, cached response reuse, private
API redaction, persistence failures and isolated PostgreSQL multistage checkpoints.
No paid QA. Activity showed only legacy unconfirmed WELL running before release;
ABT remains attention. Backend f1875301b5fa48c2ec8d3c0d9c52b37a8c7187da verified
healthy; hosted T178 JS/CSS/service-worker bytes match. Cloudflare version
704d9daa-b061-486d-8b9b-4380a985365b. Authenticated ABT inspection verifies the new
receipt summary, no exposed raw receipts, and its unchanged report-0 reservation.
Mac heartbeat healthy at 19:10:28 UTC; agent unchanged. Next: a reviewable receipt
inspection/local parse-recovery UI and operator reconciliation for legacy attempts.
Phase 1 remains incomplete; later phases remain in the approved order.

## T177 — durable graph model-call evidence

New analyst executions attach a run-scoped LangChain callback through the pinned
TradingAgents propagator arguments. Before each graph model invocation, save the
Charlie execution/call UUID; record return/error events and allowlisted provider
response IDs when available. No prompts, source text, headers or exception bodies
enter this ledger. Evidence persistence errors propagate; no paid retry is added.
Activity Recovery evidence now summarizes recorded calls, returns, errors and
starts without a recorded outcome. Missing/error records never prove no charge,
cancellation or safe replay. Provider outcome remains unverified.

Coverage is explicitly graph callbacks only: initialization, non-model tools and
final signal processing can occur outside it. This does not reconcile legacy WELL
or recover interrupted calls. The pinned TradingAgents graph args path and upstream
LangChain callback contract were inspected; callback/database behavior is verified
with synthetic fixtures, not a paid live model call. Tests cover isolation, strict
metadata filtering, malformed evidence and propagation of persistence failure.
Validation: 1021 safe backend tests, 134 frontend tests, Python compile, production
build and unchanged model-policy gate against 1357b54 passed. Backend revision
6633bd19a9a882019806071b6d125e2143f55630 reports healthy. Hosted T177 JS/CSS/service
worker match exact local bytes; Cloudflare version d866da5c-8908-4fb5-b135-5a0ec0752943.
Authenticated saved-run inspection shows the new bounded modelCalls evidence.
Legacy WELL remains running/unenrolled with zero recorded callbacks and unverified
provider outcome. Mac heartbeat fresh at 18:43:09 UTC; agent code unchanged.

ABT live collection evidence (October 11): the user-authorized assignment completed
all four collection categories with eight eligible originals, iCloud preservation
and eight exact-byte Charlie receipts. Parent assignment advanced to research, then
stopped with a JSON parsing failure. Read-only inspection after deployment confirms
attention, inFlight=report-0, zero saved responses/completed stages and no artifacts.
No retry was issued. This is collection acceptance evidence only, not completed
thesis/summary/one-pager or all eight phases. Next phase-1 priority: preserve returned
company-research responses before parsing, classify parse failures explicitly, and
provide a safe inspection/recovery path. Current parser throws before the response
is saved; a paid retry requires inspection of usage and explicit acknowledgment.
Do not treat the existing ABT reservation as proof that no call ran.

## T176 — explicit execution evidence in Activity

Saved analyst-run inspection now exposes execution enrollment, reservation count
and up to five validated Charlie execution UUIDs. Evidence is derived from all
saved logs, independent of the last-100 display window; arbitrary log fields and
report bodies are not added to the response. The UI explains that these are
Charlie references, not provider request IDs, and that a legacy run with no
marker may still have executed. Provider outcome remains explicitly unverified.
This is a diagnostic increment, not reconciliation, cancellation or retry.
Legacy WELL remains unresolved; no production job was mutated.

Validation: 1017 safe backend tests and 134 frontend tests passed, plus Python
compile, production build and unchanged model-policy gate. Synthetic evidence
tests cover legacy/malformed logs, bounded UUID validation, and reservations
older than the displayed log window. Backend revision
f24deec58f425682364087b1f04b9964ca9f4440 verified healthy; hosted T176
JS/CSS/service-worker exact bytes and version verified. Cloudflare version:
bd272ef7-111d-43c6-bee6-afc5683da487. Read-only WELL inspection confirms legacy
unenrolled record, zero saved reservations, no report/completion, and unverified
provider outcome. Activity has no unavailable services or pending collection
commands; Mac heartbeat fresh. Agent unchanged/no restart. No paid QA.
Next: capture durable provider
request/outcome evidence and build supported recovery controls without treating
missing records or elapsed time as proof that paid work did not run. The graph
currently records coarse lifecycle logs rather than per-provider receipts; a
Charlie reservation UUID alone cannot resolve that gap retroactively.

## T175 — durable batch identity and pre-execution recovery

The user approved the eight-phase order and one-click latest-source refresh;
follow docs/charlie-investor-workflow-plan.md. Continuation now explicitly follows
that plan. All phases remain in progress, with phase 1 first.

New batch submissions atomically save frozen inputs, child run IDs and a request
receipt. Same-ID retries reuse children without dispatching another worker;
changed-input reuse is rejected. The browser retains uncertain submission IDs in
session storage and blocks simultaneous clicks. Batch enrollment logs persist
identity and concurrency; a database advisory transaction lock serializes claims
per batch. Outstanding running attempts retain their slots after interruption.
Recovery discovers only never-started new jobs; saturated batches cannot hide
unrelated eligible jobs. Existing single recovery and all legacy rows are retained.
No ambiguous attempt is retried or marked complete. Activity recognizes new batch
start recovery. This is per-batch concurrency fencing, not a global distributed
concurrency cap or live-provider reconciliation.

Validation: 1015 safe backend and 134 frontend tests pass; final focused tests
also cover missing-child refusal. Python compile, production build and unchanged
model-policy gate pass. Isolated database tests cover concurrent batch retries,
frozen-input conflicts, cross-worker concurrency, slot retention/release and
recovery fairness. No paid research or production submission used for QA.
Backend 7f99ba4b99819030358d995dd565d4025350d1e1 and exact hosted T175
JS/CSS/service-worker bytes verified; Cloudflare
30ac85b1-ec20-4a58-9988-75e1cf8d409c. Authenticated capabilities and Activity checked;
Mac code unchanged/no restart. Next in phase 1: supported interrupted-attempt
reconciliation and operational recovery controls, including legacy WELL. Later
phases and the full one-click update experience are not complete.

## T174 — resume observing selected analyst work

Analyst polling now follows the selected run and screen lifecycle. Returning to
the analyst screen immediately reads its saved state; leaving aborts pending reads
and fences late responses. The first read replaces old logs and later reads append
new entries. The confirmed run ID and ticker are retained in session storage;
same-tab reload starts in Restoring saved run and reads status when the screen
opens. No report bodies are cached, and no POST/research retry occurs on restore.
Unavailable browser storage does not crash startup. Selected completed runs can
be restored too; recognized terminal states stop observation.

Validation: 1013 safe backend and 133 frontend tests pass, production build and
unchanged model-policy gate pass. Synthetic observer tests cover immediate reads,
queued/running/complete, failed reads, cancellation, late-response fencing and
return reads from cursor zero. Selection tests verify identity-only restoration.
Backend a494b80bf4289cb7b70a79214becc0ebd055a6ca and exact hosted T174
JS/CSS/service-worker bytes verified; Cloudflare
74258956-9b0e-42d0-9254-1acae9e709fa. Activity health checked after rollout;
Mac code unchanged/no restart. No paid research, approvals or production job
mutations for QA. Next: durable batch recovery, batch submission
identity and supported interrupted-attempt reconciliation. This restores only the
last confirmed single-run selection in this tab, not cross-device navigation.

## T173 — duplicate-safe single analyst submission

The single-run screen retains a request UUID in session storage for the exact
settings until the server confirms the matching job. Double-clicks are blocked
synchronously; uncertain/failed replies preserve the identity for a deliberate
retry, including after reload in the same tab. Storage failure prevents dispatch.
The backend inserts that UUID once with conflict handling, compares saved frozen
inputs on replay, returns the same job and starts no second worker. Changed inputs
with the same identity are rejected. Existing callers without requestId retain
legacy behavior; batches and separate tabs are not deduplicated. Pending reads
and submission failures remain visible. No ambiguous paid attempt is replayed.

Validation: 1013 safe backend tests and 130 frontend tests pass, plus compile,
production build and unchanged model-policy gate. Six concurrent same-ID
submissions create one row; replay after reservation creates no new attempt;
changed-input reuse is rejected without altering the original. Screen tests
cover double-click blocking and persisted identity reuse. Backend revision
605b1e8fb883f06c4cf3d5982d7a89efddf86799 and exact hosted T173 JS/CSS/service-worker
bytes verified; Cloudflare 6b0a013c-d1dc-439c-8b05-de9067d3b716. No paid research
or production submission used for QA. Activity health checked; Mac code unchanged
and no restart. Next: durable batch recovery and submission
identity, selected-run reload restoration and interrupted-attempt reconciliation.

## T172 — keep queued analyst runs tracked on screen

The single analyst screen now starts at queued, keeps polling through queued and
running, and stops only for recognized terminal states. Failed/malformed reads
retain the last saved display. Reads are serialized and bounded to 15 seconds;
old poll responses cannot overwrite a newer selected run or stop its timer.
Queued runs disable the run button and show Waiting to start rather than Run
Complete. Failed runs say Run stopped. This fixes the T168 queue-state mismatch;
it does not add HTTP submission deduplication or reload restoration.

Validation: 1012 safe backend tests and 129 frontend tests pass, including the
actual screen polling function through queued, failed reads, running and completion.
Production build and unchanged model-policy gate pass. Backend revision
ef0d7e6e510cf1cc21333ff14f037772b71c11e1 and exact hosted T172 JS/CSS/service-worker
bytes verified; Cloudflare 9c2c87b1-f0c7-4ad4-bbe9-a2cd98a81648. No provider or
production job calls for QA. The status-read warning clears after a successful
read. Mac code unchanged/no restart; Activity health checked after rollout.
Next: batch restart bounds, submission deduplication, reload restoration and
supported interrupted-attempt reconciliation.

## T171 — distinguish unstarted jobs from uncertain execution

Activity projects durable enrollment/reservation markers without changing jobs.
Enrolled queued jobs with no reservation are labeled Waiting for first execution,
with explicit automatic-start recovery availability. After 15 minutes (or unknown
or future time), they remain in attention, including beyond seven days, and ask
for a backend worker check without duplicate submission. Already-reserved or
legacy uncertain jobs retain provider-outcome reconciliation. These flags are
saved evidence, never proof a worker is live. Exact-run inspection now labels the
creation timestamp Created rather than implying execution started then.

Validation: 1012 safe backend and 126 frontend tests pass; compile/build and
unchanged model-policy gate pass. Isolated PostgreSQL tests cover fresh/delayed/
old queued work with and without recovery, and reservation switching to uncertain
execution. Backend 4b4f405f5aaf20d50476359e7e9760e956229329 and exact hosted T171
JS/CSS/service-worker bytes verified; Cloudflare
03eafaf4-46be-4c8f-b641-a11a0ece9d56. Live Activity has no unavailable sources;
legacy WELL remains running/attention with execution unconfirmed, not pending
start. Mac heartbeat current at 16:11:02 UTC; no agent code change/restart.
No research, provider calls or job mutations for QA. Next: durable batch concurrency, submission deduplication and supported
interrupted-attempt reconciliation, then further research workspace simplification.

## T170 — recover single analyst jobs before first execution

New single-company analyst jobs opt into restart discovery. Every minute the
backend checks up to ten enrolled queued jobs older than two minutes, using their
saved ticker/date/provider/model. The existing committed atomic reservation fences
normal dispatch, concurrent recovery loops and process restarts. Once reserved,
an attempt is never automatically retried, even after a crash. Legacy and batch
jobs are excluded; batch concurrency is not yet durable. A retired model fails
before graph imports rather than acquiring a replacement. Recovery runs serially
within each process's loop; this is not a global concurrency queue. Capabilities
report single-start recovery separately from durable batch execution.

Validation: 1012 safe backend tests and 125 frontend tests pass, with compile,
production build and unchanged model-policy gate. Concurrent disposable database
recovery tests confirm one claim, preserved saved parameters and excluded jobs.
Backend 112172c93e97abc14307bda11ce7d56b23e7bd41 and exact hosted T170
JS/CSS/service-worker bytes verified; Cloudflare
7a768218-166c-4bf1-bbb0-cc5b780da557. Authenticated capability read confirms
single-start recovery enabled and interrupted retries disabled. WELL remains
running with 22 logs and no outputs. Mac heartbeat 15:42:30 UTC; agent code
unchanged/no restart. No paid research QA or live restart scenario induced.
Next: batch concurrency
persistence, submission deduplication, interrupted-attempt reconciliation, and
clearer recovery status in Activity. Existing legacy WELL state is untouched.

## T169 — truthful saved-report presence

Activity's exact analyst-run inspection excludes batch enrollment metadata and
empty values from saved-report presence. This applies regardless of recorded job
status and preserves every stored report, batch marker, log and job state. The
endpoint still returns only presence flags, never report bodies. Disposable
PostgreSQL coverage exercises queued/running/failed/complete states, metadata,
empty fields and genuine Markdown, and verifies unchanged records after reads.
1011 safe backend tests and 125 frontend tests pass, with focused database tests,
compile, production build and unchanged model-policy gate. Backend revision
782b046f510ab3592a657b8d9704fd0cae459c5e and exact hosted T169 JS/CSS/service-worker
bytes verified; Cloudflare a3463efd-1e0a-4e40-82c5-31490009de8f. Live read-only
WELL inspection remains running with 22 logs and no saved outputs; it was not
restarted or altered. Mac heartbeat current at 15:11:16 UTC, collection delivery
pending zero. No Mac code change/restart or paid research QA. Next: safe recovery
of enrolled jobs that never started, separate-submission deduplication and
supported reconciliation of interrupted attempts; never automatically replay
an uncertain paid request.

## T168 — one-shot analyst execution reservation

New single and batch analyst runs enroll as queued with a durable marker. Before
loading the research graph, a PostgreSQL conditional update reserves exactly one
execution attempt and records its identity/time. Concurrent workers, terminal
runs, already-attempted runs and legacy unmarked runs cannot start that saved job.
Reservation commits before provider work. No expiry/takeover or automatic paid
retry is added: interruption after reservation remains an uncertain outcome.
This does not deduplicate separate submissions or resume pre-dispatch queued work;
those are next steps. Existing logs, reports, model pins and legacy states remain.

Disposable PostgreSQL tests cover concurrent claims, replay refusal, legacy and
terminal states. A synthetic graph-entry test proves reservation refusal returns
before graph imports or writes. 1010 tests in the broad safe suite and the added
graph guard test pass, as do 125 frontend tests, compile/build and unchanged model
policy. Backend 501b0f56ab23e9b153d877977def1317cd2f2a91 and exact hosted T168
JS/CSS/service-worker bytes verified. Cloudflare
624b944b-2e52-4beb-bdd1-244cf242f7a2. Read-only live inspection confirms WELL
remains running with 22 logs and no outputs; no legacy job mutated. Mac agent
code unchanged/no restart; heartbeat observed 14:40:39 UTC. No paid research QA.

## T167 — inspect the exact analyst run from Activity

Analyst-team rows open a read-only inline inspection of that exact saved run,
including model, recorded state, output-presence flags and the latest 100 log
entries. No full report body is returned by this diagnostic endpoint. Refresh
retries only a read. Missing runs and failures remain visible; stale responses
are cancelled on selection/unmount, and ticker/run mismatches are rejected.
Desktop/phone layouts wrap long steps and bound the log list. Original run state
and any saved research remain unchanged; no automatic paid retry is introduced.

Validation: focused PostgreSQL tests verify bounded logs, invalid/missing IDs,
no report-body disclosure and unchanged database state. Synthetic browser checks
cover desktop/phone, long messages, read failure/refresh, close and GET-only
requests. 1007 safe backend tests, 125 frontend tests, Python compile, production build
and unchanged model-policy gate pass. Backend c9e427f74330b8c64ddff1985d9352892ab7af2d
and exact hosted T167 JS/CSS/service-worker bytes verified. Cloudflare
a713b887-368f-4e32-94a1-9905d2bee23a. Live authenticated WELL inspection
returns 22 saved log entries and no report/decision; anonymous access is 401.
Mac agent code unchanged and not restarted; observed heartbeat 14:11:41 UTC.
No production job or research mutation. Next: durable
execution ownership and a supported reconciliation action for interrupted runs.

## T166 — visible reconciliation for legacy analyst jobs

Activity now uses the last saved analyst-team log timestamp and keeps unfinished
legacy jobs with missing, future, or over-15-minute-old progress in attention,
including jobs older than seven days. It explicitly labels execution unconfirmed
and warns against a duplicate paid retry. This is a read-only projection: original
status, logs, reports and provider calls are untouched. Fresh logs retain the
normal active lane; completed jobs retain their completion state.

Build continuation restored as `continue-charlie-improvements`, without the prior
two-hour expiry. Remaining priorities include supported legacy-run reconciliation
and durable execution ownership, sourced historical valuation context, broader
issuer profiles and analyst-graded real-source output benchmarks. T166 does not
claim to resolve the WELL provider outcome or clear the model-maintenance gate.

Validation: 1006 safe backend tests and 125 frontend tests pass; Python compile,
production build and unchanged model-policy gate pass. Disposable PostgreSQL
coverage verifies projection and unchanged stored status. Backend revision
04712d4d6547625770eaf60a78620524786640c7 and exact hosted T166 bundle/service-worker
bytes verified; Cloudflare 91c4cff6-0c84-4781-ae31-c30f98a1e37e. Authenticated
Activity places the existing WELL job in attention with execution unconfirmed.
Mac heartbeat current; no agent restart, research dispatch or job mutation.

## T165 — preserve citations in printed visuals

Renderer v10 places citation destinations and safe original source links in the
visible footer. The expandable register is omitted from printing; its hidden
anchors previously left the one-page PDF without working citation destinations.
Source wording and original artifacts remain unchanged. 1004 backend tests, 124
frontend tests, compile/build pass; final footer change also passes focused tests.
UNH/DE/SYK render as one page with Calibri/black text and valid source links.
Backend 9a2549d and exact hosted T165 bundle/service-worker bytes verified.
Cloudflare 88aa4cf7-f3a7-40f3-a028-edd617356cb0; Mac healthy. Authenticated
production UNH export renders one page with 23 citation/source links, Calibri/black
text, and no text outside page bounds. No model calls or approvals used for QA.

## T164 — precise saved-data labeling

Illustrated renderer v9 calls the financial series "Saved GAAP history". Replaying
an older snapshot must not imply it was refreshed with the latest filing-selection
logic. Existing snapshot dates, observation register and original exports remain.
This is a caption correction, not a data refresh. 1003 backend tests, 124 frontend
tests, compile/build pass. Backend 4739060 and exact hosted T164 bundle/service-worker bytes verified;
Cloudflare 188d90ea-bd27-4bf6-a2e1-5d6ca0ece8b2. Mac healthy. Live UNH report
shows renderer v9; its exported PDF is one page, Calibri/black, no out-of-page text.

## T163 — saved financial history workbook

Research assignment and report visual views can export the retained annual financial
observations as Excel. Financial history and a separate observation register preserve
native units, zero/negative values, dates, source tags/calculations and filing links.
Missing values stay blank; incompatible units are withheld from standardized history
while the original observation remains inspectable. Calibri/black text, light fills,
frozen headers and filters support analyst use. This is a saved-value workbook, not
a forecast model or live market-data refresh. Export is authenticated and read-only;
no model, new research, thesis approval or mutation is performed.

1002 safe backend tests and 124 frontend tests pass, with compile/build.
Read-only export, missing data and unchanged assignment state are covered by isolated
PostgreSQL tests. Both workbook sheets rendered and inspected using saved UNH data.
Backend 8d72ac1 and exact hosted T163 verified; Mac healthy. Live UNH workbook
downloaded and parsed, with anonymous access rejected. No research dispatched.

## T162 — consistent visual across report and assignment

The Stock Analysis Infographic tab now uses its matching assignment's saved
research/public-data snapshot and current one-pager renderer. Viewing and exporting
never refresh prices, run a model or rewrite the original export. Snapshot date and
renderer version are visible; the original saved visual has a separate export.
Company/report identity and ambiguous-parent guards prevent picking another
assignment's visual. Standalone legacy Studio visuals remain a separate workflow;
its cache version was bumped for the current evidence checks.

998 backend tests, 123 frontend tests, compile/build pass. Isolated PostgreSQL
coverage verifies the report route returns the assignment layout, preserves the
original and creates no Studio output or model call. Backend a48ac3f and exact hosted T162 assets verified; Mac healthy.

## T161 — meeting preparation agenda and exact saved-meeting links

Activity includes a bounded seven-day preparation agenda using New York dates.
Saved question packs are distinguished from meetings needing preparation; neither
means analyst review or attendance. Completed/cancelled/archived meetings are
excluded. Undated and out-of-window meetings remain in Meeting Preparation.
The agenda does not create work, sync an external calendar or change job counts.

Meeting links now retain the specific meeting across reload and Back, including
links from research assignments and the research desk. Company mismatches and
missing meetings remain visible; stale reads cannot replace a newer selection.
A background meeting reload cannot redirect another workspace.
995 backend tests, 123 frontend tests, compilation and build pass. Additional
isolated PostgreSQL tests verify concurrent handoff reuse and deleted destination
handling. Backend 4616ce9 and exact hosted T161 assets verified; Mac healthy.

## T160 — saved research questions to Meeting Prep

Completed Stock Analysis reports can copy 1–10 selected diligence questions into
a dated meeting without a model call. Exact wording and originating report/path
are retained; source review status remains explicit. Selected questions become
must-ask planned questions with the existing answer-capture and version views.
No source passages are certified or copied into the meeting; originals stay with
the report. No thesis, decision or actual answer is inferred.

A single transaction saves the meeting, question set, planned answer records and
durable receipt. Identical report/selection/date/type submissions reuse the saved
meeting, including after a reload. Changed question fingerprints are rejected;
a deleted destination is not silently recreated. Inspecting the selector is read-only.
991 backend tests, 122 frontend tests, compilation and build pass. Isolated browser
save/open/answer/duplicate flows passed with synthetic data, including native date
input handling. No production meeting or answer was created for QA. Deployment
Backend a4148a7 and exact hosted T160 verified; Mac healthy.

## T159 — derivative evidence identity and fuller thesis arguments

New visual renders, summary drafts, stock summaries and thesis handoffs require
an unchanged populated report field, model-reviewed matched passage metadata and
an original belonging to the frozen report. Orphan/stale/unavailable citations
cannot enter a derivative. Original saved exports and research are retained.
The fallback one-pager denominator counts populated report fields, not just the
accepted subset. Source-change comparisons still distinguish wording from receipt
changes. Zero-valued fields remain valid; this is consistency, not factual proof.

The illustrated map now prioritizes complete investment-thesis arguments over
short driver labels. A generic investment-question placeholder is removed; an
actual question appears only if supported. Whole-claim length budgets remain.
Saved UNH and public-only DE/SYK layouts remain one A4 page with Calibri/black text
and no outside-page text. 986 backend tests, 120 frontend tests, compilation and build pass. Backend dda7b9b and exact hosted T159 assets verified; Mac healthy.

## T158 — due issue reviews and consistent evidence status

Activity and the existing research follow-through queue share the same issue selection.
Activity includes unsuperseded analyst issue reviews due on or before the current
New York date. Old due reviews remain in Needs you instead of aging into Older
unfinished. Reads are bounded, do not mark a review complete, and never dispatch
research or change a thesis. Exact links open the saved decision, with a deliberate
follow-up draft action. A superseding review/date preserves the prior history.
Business conditions are not automatically evaluated; there are no external alerts.

Stock report badges, counts, review checklist and thesis-draft preparation now
share the same current-field/source identity check. Orphan citations, changed
wording, absent originals and unavailable placeholders cannot earn a matched
badge or enter a thesis draft. Reported zero is preserved. These checks remain
mechanical passage/model-review metadata, not independent factual adjudication.
982 backend tests, 120 frontend tests, compile/build pass. Full extraction metadata no longer implies excerpts in derivative warnings. Backend 851a4b6 and exact hosted T158 verified; Mac healthy.

## T157 — financial vintages and calculation register

Annual observations now select the newest filing across supported tags; an old
revenue tag cannot mask a later amendment or changed tag. Conflicting newest
values are withheld, never replaced with an older value. Dates, accession format,
units, non-finite values and same-filing arithmetic are checked. New snapshots
record warnings and the restatement basis. Existing frozen snapshots are retained.
The visual's source register adds readable periods, unscaled values, definitions,
calculation formulas and SEC filing links. Quote dates do not certify the vintage
of provider-defined estimate multiples. No calibrated targets are inferred.

Frozen public SEC UNH/SYK/DE facts support five-year parser replays. DE's 2023/2024
operating margins are withheld because inputs come from different filings. These
are parser regressions, not independent investment-quality certification.
Saved UNH and DE/SYK public-only layouts remain one A4 page, Calibri/black, no
outside-page text. 978 backend tests, 117 frontend tests, compile/build pass. Backend 7e9aa42 and exact hosted T157 assets verified; Mac healthy.

## T156 — extraction completeness before model calls

New company/committee/stock runs reject an oversized or silently clipped
per-document extraction before any research call. The 160,000-character bound is
unchanged. New frozen sources record exact returned-extraction ranges and hashes;
shared-pack selection replaces this metadata only where text was actually omitted.
Word/table-order, cached spreadsheet-value and decoding limitations are explicit.
Full returned extraction is not full-document review or factual validation.
Existing frozen sources, reports and originals are not reprocessed.

Direct report links now put the saved report ahead of new-assignment setup.
971 backend tests, 117 frontend tests, compilation and build pass. Backend 5bb9e55 and hosted T156 verified; direct UNH report links keep setup collapsed.

## T155 — exact-original evidence downloads

Stock Analysis, Deep Research and Committee source/evidence views offer verified
original downloads. The authenticated read route resolves a source only within
its saved run and company, verifies both frozen identities and current stored
bytes, and returns an attachment with no-store/nosniff/sandbox headers. Missing,
changed, malformed or over-bound originals fail explicitly; no newer file is
silently substituted. The browser checks SHA-256 again before offering the file.
No extraction, model call, collection or job wakeup is added. Originals remain
unchanged. Human source inspection does not approve the research.

Synthetic exact-byte/tamper/unknown-source tests and focused disposable-PostgreSQL
route tests pass. 966 backend tests, 117 frontend tests, compilation and build pass. Backend 9ae97ea and hosted T155 assets verified. Read-only UNH original download matched the frozen SHA-256; anonymous access returned 401.

## T154 — research coverage and review checklist

Stock Analysis has a deterministic Coverage & review tab and export. It walks
saved report fields, distinguishes unavailable items from unresolved claims, and
counts a match only when unchanged wording has model review plus a matched passage
from a frozen original. Orphan citations cannot inflate coverage. Source selection
omissions, extraction limitations and missing coverage metadata remain visible.
Section links open the underlying report; no research or thesis is changed.
Counts include labels and periods and are not an investment-quality score.

Saved UNH inspection found 65 populated matched fields, 171 needing source review,
27 unavailable items, two limited originals and four without coverage metadata.
These are metadata/structural observations, not new factual adjudication. No paid
research was started. Exact assignment focus now hides the unrelated default
report until its explicit report link is opened; R0 no longer says saved thesis.
115 frontend tests and build pass. Backend 76576f4 and exact hosted T154 assets verified. Live UNH counts, section navigation and mobile width checked.

## T153 — exact saved-work links

Activity links select a specific research assignment, stock analysis, Deep Research,
committee review or Mac collection request. Work identity is bounded to compatible
route sections, persists through reload/Back, and is cleared on company/section
changes. A mismatched/missing item never falls back to a different saved job.
Collection remains limited to the latest 50-request Mac snapshot and states when
the linked request is outside it. Focused assignment reads use exact ID + company
and do not wake the coordinator. Existing authentication and mutation gates remain.
Assignment setup is collapsed while inspecting a direct link. No paid calls used.
964 backend tests, 111 frontend tests, compile/build pass. Backend b5c46da and exact hosted T153 assets verified. Activity opened the specific UNH assignment; reload retained its identity.

## T152 — collection follow-through and bounded investment maps

Collection opens with unfinished requests, attention first, scoped to the company
in the route. Investors can search companies, inspect saved searches/document
states, include finished requests, and use the existing exact-request recovery
controls. Policy creation is collapsed; current policies stay editable. Counts
explicitly describe the latest bounded Mac snapshot, not downstream completion.
Stale GET responses cannot overwrite newer reads. No frozen requests changed.

Investment maps retain whole source-supported claims and whole paired debates
within section budgets. Oversized claims are omitted with a full-research notice,
not truncated or automatically rewritten. Monitors remain paired with selected
debates and catalyst timing retains citations. Saved UNH and public-data-only
DE/SYK render as one A4 page with Calibri/black text and no outside-page text;
synthetic 200/400/800-character claims also stay one page. These checks do not
establish research completeness or universal layout fit. Existing saved outputs
are not overwritten. 963 backend tests, 109 frontend tests, compilation and build pass. Backend 6523f69 and exact hosted T152 assets verified. Collection company scope and mobile width inspected; no recovery actions executed.

## T151 — issuer-specific visuals and segment fidelity

The investment map adds sourced Deere/Stryker business profiles, bounded peer
sets and conceptual industry artwork. Healthcare wording/art remains UNH-specific;
general investment-test and debate labels no longer impose a recovery narrative.
Stryker segment aliases deduplicate only when values agree. Deere uses the mapped
operating-segment context and actual USD unit definitions. Missing research fields
remain explicit. Existing stored originals/reports/fundamentals are unchanged.

See docs/visual-business-profile-validation.md for public-source regressions,
actual retrieval observations and PDF checks. 960 backend tests, 106 frontend
tests, compilation and build pass. Backend 35e2e1f and exact hosted T151 assets verified; Mac agent current.

## T150 — direct workflow navigation

Activity and Today route known job types to the appropriate research-desk or
Stock Analysis section. Those sections have bounded URL parameters, survive reload
and browser Back/Forward, and can be bookmarked. Stock Analysis company switches
also update the address. Navigation does not dispatch research or apply proposals.
Links select a company/workflow, not an individual job ID; the saved job selector
remains necessary when multiple jobs share that company/workflow.

Validation: 106 frontend tests and production build pass. The preceding backend
change passed 956 tests. Backend 095a20f, exact hosted assets and authenticated reads verified. Cloudflare
8d9025d5-bf9e-4ff4-b8e4-7fd9f23b77a1. Live section links, reload and Back
verified; Activity opened SYK collection directly without submitting work.

## T149 — conservative catalyst issuer attribution

New catalyst candidates require an explicit ticker prefix or a saved company-name
prefix before automatic collection is eligible. Related-company feed membership
alone no longer qualifies. Short ticker words, missing company names, other issuers
and relationship headlines remain reviewable without consuming collection quota.
Saved attribution basis is a triage hint, not primary-source or event-date proof;
collection still must verify the issuer, actual event date and materiality. Existing
frozen requests are unchanged. No new provider/model lookup is introduced.

Validation: 956 safe backend tests, 103 frontend tests, Python compilation and
production build pass. Backend d2ccf67 and exact hosted T149 bytes verified.
Cloudflare 790b863b-6f4c-4c6e-aa74-15999d48bf89. No paid QA.

## T148 — explain waiting intake and expose its existing daily cap

Activity and automatic intake controls now explain waiting sources using the
worker's policy, UTC daily reservations, saved-thesis prerequisite and active/open
company proposal gates. The controls also inspect current server-key/budget gates;
Activity deliberately does not assert those checks succeeded. Reads do not dispatch
or mutate jobs. The existing 1–5 daily-comparison setting is now editable in the
UI, with a clear cost implication; current user settings are not changed by deploy.
Polling cannot overwrite a newer settings response. T147's empty state now directs
companies without a saved case to Current thesis rather than an absent baseline.

Validation: 952 safe backend tests, 103 frontend tests, Python compilation and
production build pass. Backend ebc195a and exact hosted bytes verified; authenticated
intake and Activity reads return actual review/baseline wait reasons. No settings
or research changed during live QA. See the finite checkpoint.

## T147 — saved-evidence change brief

The investment-case workspace adds What changed, a read-only brief combining case
revision differences, newly accepted evidence links and unresolved investment-case
proposals. Its default cutoff is an actual recorded evidence-proposal review, not
an inferred whole-thesis approval. If no usable baseline exists, the investor must
choose a saved revision. Pending proposals before the cutoff remain visible; their
updated timestamp identifies later repairs. Excerpts, source filenames, provenance
and unresolved checks remain visible. Coverage is bounded to the existing 100-item
history and 30-proposal windows, not a complete news/document scan. No model calls,
review decisions or thesis mutations occur. Unsaved editor changes are excluded.

Verified T147 backend 3893c89315a045a67ac5b60d085e9a9d733ebd0f and hosted
asset bytes. Cloudflare 95fe631d-2f2e-4d67-9b9f-bfb26cbade70. Auth checks passed;
live UNH correctly reports no investment-case baseline. Desktop/mobile inspected,
390px without horizontal overflow. 947 backend/103 frontend tests passed.

This is the first increment of the authorized two-hour investor workflow session;
see docs/investor-workflow-progress.md for the checkpoint and finite cutoff.

## T145 / T146 — investor exception inbox

Activity defaults to Needs you, separating investor review/proposals from jobs
whose progress needs checking. Processing and Waiting have separate counts and
filters. Today prioritizes exceptions before routine progress. Long-waiting queued
work stays Waiting; a stale processing state prompts inspection, never asserts
that a paid call stopped. Every row includes a next action. Catalyst signals now
show their saved review reason and workflow name rather than a generic status.
T146 removes duplicate collection issue headings and folds long diagnostics into
expandable details for mobile readability. Catalyst roundups, personal share-dealing
headlines and ambiguous multi-company feed associations now require review before
automatic collection, without consuming collection quota. This is a conservative
triage rule, not complete issuer/date verification; existing jobs are unchanged.
The Activity inbox is read-only triage: no jobs are resumed, proposals approved or limits changed.

Investor workflow priorities, in order:
1. Reliable exception handling and clear dependencies (T144/T145 improve this).
2. Strengthen catalyst issuer/event-date validation before expanding automatic
   collection. Live attention records include mismatched issuer and stale-event
   signals; existing frozen requests must be reviewed, never silently relabeled.
3. Evidence-led change briefs: material changes since the investor's last reviewed
   thesis, with source links, contradictions and explicit proposed edits.
4. Reusable coverage policies and event-triggered research with transparent budgets;
   extend existing intake/catalyst automation rather than creating duplicate engines.
5. Consistent professional one-pagers across issuers, then valuation history and
   defensible scenario assumptions; expand real-source analyst quality benchmarks.
6. Meeting/earnings preparation assembled from saved research and open questions;
   test recovery across every upstream dependency before calling this unattended.

Do not market complete automation: AlphaSense access/source restrictions,
ambiguous paid outcomes and investment approvals still require human involvement.
Validation: 947 safe backend tests and 98 frontend tests pass; Python compilation
and production build pass. Dark-theme inspection identified and corrected health
panel contrast by using theme variables. T145 backend revision was verified as
c022253098e5cfeb6a75761deaa93fba4a235037, with matching hosted assets and
390px mobile layout without horizontal overflow. Final T146 backend revision
verified: d458c5c68f719f85be0eaa31b51f639a9515dcc8. Cloudflare version:
e71a5c13-0233-45b2-9dd0-16a3c1b3d8df. Hosted index/bundle/service-worker bytes
match; authenticated reads returned 200, unauthorized access 401, invalid submission
400. Mac agent connected, collection snapshot current, no queued cloud delivery.
Desktop (1280px) and mobile (390px) have no horizontal overflow; Waiting filter
works and viewport was restored. All retrieval/dispatch failure-path tests used
isolated fixtures; no paid research was launched. Existing attention jobs remain
unresolved and unchanged.

## T144 — collection delivery visibility and public-data recovery

Activity and Today now distinguish Mac heartbeat, collection-report freshness,
and queued cloud collection commands awaiting Mac acknowledgment. Queue metadata
is aggregated independently of capped activity history. Unknown, future-dated,
partial and stale snapshots never appear healthy. The panels explain the next
check; a connected agent alone is not evidence of successful collection.

Completed visual assignments with incomplete public fundamentals expose a retry.
The retry is guarded by the last retrieval timestamp and the assignment lock.
Failed or still-incomplete replacements preserve the entire previous snapshot;
successful replacements archive it. Healthy v2 replays remain cached. Research,
original exports, saved thesis proposals and paid calls are untouched.

Outstanding priorities after this increment:
- Extend issuer-specific one-pager profiles and peer curation beyond UNH; verify
  density, evidence coverage and readability against the supplied visual standard.
- Add sourced historical valuation context and defensible scenario assumptions;
  current sensitivity arithmetic must not be presented as calibrated forecasts.
- Broaden real-source quality benchmarks with analyst grading across document,
  audio and company types; synthetic passing tests are not live quality proof.
- Audit interrupted-job recovery across credentials, process restarts and ambiguous
  provider outcomes; keep explicit confirmation for uncertain paid retries.
- Resolve individual collection source/authentication blockers through the managed
  workflow. Do not bypass restrictions or equate collection with research completion.

Validation: 945 safe backend tests and 97 frontend tests pass; focused coverage
checks old-command visibility, stale/future health timestamps, failed retrieval
preservation and duplicate retry caching. Python compilation, frontend build and
unchanged model-policy gate pass. T144 backend revision verified as
62986a763341410423fc8544bef764273ccb5ecd; hosted asset bytes matched. Cloudflare
version 17d80dcd-3cf0-4178-9a67-e926ee258947. Live Activity showed current collection
reports and no pending cloud delivery. T145 follows with theme/triage improvements.

## T143 — editorial investment map

Rebuilds the visual renderer as an A4 editorial map with conceptual healthcare
artwork, public UNH business descriptions, endpoint changes on five financial
charts, distinct supported thesis drivers, paired debates and resolution signals.
Public fundamentals v2 adds curated UNH comparables (CVS, CI, HUM) with matching
quote dates. The valuation matrix is arithmetic EPS/P-E sensitivity, explicitly
not calibrated forecasts or price targets. Other issuers retain a generic
source-backed layout; company-specific artwork/peer curation is currently UNH.

Completed visual supplementation upgrades v1 evidence to v2 once, archives the
previous evidence snapshot, and preserves original exports and report/thesis
references. Failed public retrieval leaves the prior snapshot untouched. No
research model or thesis approval is invoked. Missing peer data remains visible
in retrieval gaps. Historical valuation ranges and calibrated scenario forecasts
remain absent; these must not be invented to fill the reference template.

The actual existing UNH evidence rendered to one A4 page locally. 943 safe backend tests and 96 frontend tests pass; production build, Python
compilation and unchanged model-policy gate pass. PDF inspection confirms only
Calibri/Calibri Bold, black text, and no text outside the page. Production verified at `78652f8f6d9d006a9de3d7ea1e516825a5c60fa2`, Cloudflare
`9498f4ce-5fb2-4d14-894c-d29883bdf227`. Hosted index, bundle and service-worker
bytes match. Authenticated reads passed; unauthorized reads and invalid submissions
remain blocked. UNH v2 supplementation returned no gaps at
`2026-10-10T19:57:02.555767+00:00`; replay kept the same timestamp. Original
export bytes and saved report/thesis references remained unchanged. The actual
production HTML rendered to one A4 page with Calibri/black text and no clipping;
desktop/mobile iframe width equals scrollWidth (746px/353px). Mobile viewport
was restored. Mac agent remained connected. No paid research was run for QA.
User deliverable: Downloads/UNH-investment-map-v4.pdf and matching HTML/PNG.

## T141 / T142 — illustrated stock brief and public financial supplement

Implements an opt-in public financial supplement for visual assignments: five
annual SEC XBRL periods, issuer-verified inline-XBRL segment revenues, and a dated
Yahoo Finance quote with provider-defined multiples. New assignments default to
this visible choice; the selected window still governs event research. Existing
completed visuals can append a cached supplement through the authenticated
`visual-supplement` action without changing frozen inputs, original exports,
research claims, or thesis approvals. Retry is idempotent; no model calls are used
for supplementation. Public retrieval failures preserve research and are visible.

The SVG/HTML renderer supplies nine numbered panels, business and company
illustrations, five financial charts, source-linked views, valuation cards and
monitoring conditions. Calibri/black text and light fills follow user preferences.
Calculations require matching periods and accession numbers; invalid, future,
conflicting or nonfinite data are withheld. Intersegment sales are explicitly
non-additive. SEC receipt hashes and calculation inputs remain in the source
register. No claim of independently reviewed forward estimates is made.

The actual UNH saved report and public evidence were rendered as a one-page A3
PDF and inspected visually. Peer comparisons and calibrated scenario prices are
still incomplete and labeled as such; this release does not establish that the
full reference quality has been reached. Public data depends on SEC/Yahoo/network
availability; non-US or unsupported accounting mappings need attention.
937 safe backend tests and 96 frontend tests passed; production build, Python
compilation and unchanged model-policy gate passed. T142 verified at
`fd7b23280a8abd1962dad6270ca364d3c501ae23`, Cloudflare version
`4ef1728b-317f-4512-8b8b-f44a676d261d`: exact backend revision and hosted asset
bytes match. Authenticated APIs passed, unauthenticated access remained blocked.
UNH production public retrieval succeeded with no retrieval gaps; repeated
supplementation returned the same timestamp, original bytes and report/thesis
references remained unchanged. The hosted HTML was rendered to one A3 page with
only Calibri/Calibri Bold and black text. Desktop preview inspected; mobile iframe
368px width/scrollWidth has no horizontal overflow; viewport restored. Mac agent
continued running through deployment; unrelated pending catalyst sources remain.
T142 explicitly sets SVG chart labels to Calibri after PDF font inspection
found an SVG fallback; the regenerated PDF embeds only Calibri and Calibri Bold. No paid research ran for QA.

## T139 / T140 — visual brief repair

Replaces the coordinating assignment's generic first-two-fields boxes with a
source-linked editorial brief: paired competing views, recovery/pressure panels,
complete statements, broker-estimate labels, and explicit company valuation and
scenario/history gaps. Existing visual exports render from their saved report
without a model call or database mutation; `?render=original` retains the saved
original. Stock Analysis adds a sandboxed inline preview. No thesis was changed.

The actual UNH saved report was rendered offline to a one-page A4 PDF and inspected
visually using installed Calibri fonts, black text, and light fills. This verifies
that specific artifact, not arbitrary future report lengths. Long future claims
remain whole and may need more than one printed page. Missing company valuation
and financial history remain substantive evidence gaps. No paid QA research ran.
932 safe backend tests and 96 frontend tests passed; frontend build and unchanged
model-policy gate passed. T139 production revision and hosted bytes verified; desktop preview works. Mobile
inspection found nested panels squeezed the iframe; T140 expands it into a
full-width mobile overlay and aligns document reading order with section numbers.
T140 verified at `f025393f56eedd6a2c3061188be43975ee4d2d99`; Cloudflare version
`4e087b8a-15b4-40e2-9d3c-d49cef22d78e`. Backend revision and all hosted asset bytes
match. Authenticated production UNH HTML exactly matches the local renderer;
`render=original` retains the old export and the parent timestamp is unchanged.
Mobile preview fits at 390px (353px iframe content, no horizontal overflow),
desktop preview inspected, and viewport restored. Mac heartbeat healthy.
The follow-up passed 21 focused backend tests, 96 frontend tests and build.

## Start here

T134 repaired Mac delivery; T135 adds bounded source excerpts and overdue delivery visibility.
T136 repairs known-response recovery and reduces structured-output verbosity.
T136 is verified at `2bfc6257c395923647af5ba039c2fe61f0eddab2`. T137 Activity and
explicit thesis-quality recovery are verified at `27dafdfde7655343241cb3cf4cb4a986d60a953d`.
UNH completed all 12 stages, then its unsupported thesis group blocked handoff. Its
explicit two-stage repair completed with two supported thesis statements and the
other ten stages retained. Handoff then exposed legacy risks lacking triggerPoints.
T138 marks those missing legacy fields explicitly in the pending draft and corrects
Activity status grouping after production data verification. No further paid calls
were needed for the adapter repair. T138 is verified at `36cc9cdbfc506a89ec3b1d82095cf9bbb0d9d700`
with Cloudflare version `fd719525-2af3-47e6-b249-103c8a5b0966`. Hosted assets match locally built bytes;
928 backend and 96 frontend tests passed. Authenticated Activity succeeds; unauthenticated access
remains protected. Mac heartbeat is healthy. Desktop navigation and 390px mobile overflow checks passed.

UNH assignment `e998e805-0bd4-49ce-aafa-2af123e1aace` completed at 17:23 UTC October 10.
All 12 research stages are saved, with 66 supported fields and 171 requiring review.
Draft `5733d627-eedb-4b33-8628-0713f528272f` is pending, not stale, investorApproved=false:
five pillars, eight risks, seven explicitly marked legacy trigger gaps, six original source references.
The live thesis was not changed. The visual is saved and exportable; HTML metadata and coverage
warning were checked. Browser policy blocked local-file rendering, so no new visual-render claim is made.
Two restricted sources remain held; two eligible originals were excerpted with omissions recorded.
Activity shows the parent as Ready for investor review and opens UNH directly.
Other companies and legacy queued/attention jobs were preserved; their completion is not claimed.

- App: `https://charlie-deployment.tonydlee.workers.dev/?release=T138#view=activity`
- Backend: `https://equity-analyzer-backend.onrender.com/health`
- Repository: `/Users/tonydlee/Projects/equity-analyzer-backend`, branch `main`.
- Previous release: T132 / `66f67aa4`; T122 model policy remains unchanged.
- Local agent shares the model registry, reloads changed registry files between jobs, and reports its registry revision in its heartbeat.

Read `AGENTS.md` before changing anything. Preserve unrelated dirty and untracked files.

### T138 — current work versus review backlog

Production verification exposed legacy `needs_review`, `awaiting_approval`,
`proposed` and evidence-intake `submitted` statuses being treated as active.
Activity now classifies these accurately, keeps work unchanged for over seven days
under Older unfinished, and paginates visible rows. No jobs are deleted or marked
complete by this display change. Active rows are prioritized independently of
review-history volume. The Today snapshot prioritizes end-to-end assignments. Legacy risks without
triggerPoints are carried forward with an explicit missing-field note, preserving
original text/IDs and the untouched live thesis. Local adapter validation using
the actual UNH saved report and existing thesis succeeded without submission.

### T137 — central activity and bounded thesis-quality recovery

Adds an authenticated, read-only `/api/activity` inventory across saved research,
Mac collection, document processing and audio work. Active/attention records are
selected independently of recent completions, so old stuck work stays visible.
Parent assignments suppress their report/collection child duplicates. Payloads,
source bodies and credentials are not returned. Partial source failures and caps
are explicit. Activity is in the main navigation; Today shows a compact snapshot.
Desktop/mobile layout uses the existing workspace theme, with status filters,
search, elapsed times, Mac reporting status and company/workspace links.

The live UNH review found unsupported thesis fields. The matching-source gate is
unchanged. Explicitly resuming that specific quality failure retries only the first
author/reviewer pair, within the existing retry bound; it preserves old output in
`qualityRevisions` and all other completed groups. It may add two paid calls and
is never an automatic unbounded quality loop. Author guidance now favors one narrow
proposition and a source-first conditional thesis over compound claims.

The existing collection heartbeat remains every 15 minutes; its expired build
extension was removed. Empty local queues no longer count as proof that cloud
requests were delivered. Keep the Mac awake, Codex running and AlphaSense signed in.

Validation: 925 safe backend tests, 96 frontend tests, production build and unchanged
model policy. Synthetic desktop/mobile Activity checks passed, including ticker
shortcut propagation. Production verification is still required.

### T136 — bounded response and known-outcome recovery

The live UNH first author response reached its unchanged 10,000-output-token cap;
usage recorded one call at $0.5359, with zero completed report stages. Tightened
field/quotation lengths and populated-field guidance within the same six author
and six reviewer stages, same model and same token ceilings. Confirmed provider
returns (output cap/invalid JSON) now have a distinct known failure receipt; network
uncertainty still requires explicit retry acknowledgement. Resumes count retries
against the existing bound. Returned structured JSON is persisted before schema
validation so a validation repair can reuse it without another paid request.
Tests distinguish known failures from timeouts and prove no rebilling on repeated
validation failure. Mac report timestamps now explicitly identify UTC.
Live UNH remains the same authorized assignment/report, not a new QA research run.

### T135 — source context and stalled-delivery visibility

The live UNH assignment verified six exact-byte cloud originals (one conference,
one company release, four brokers); two more originals are held locally because
their disclosures explicitly prohibit AI input. All 57 broker result titles and
15 release titles were screened; presentation search was empty. Full-text review
of all matching results is not claimed. The six-source extraction was 187,848
characters, exceeding the existing 160,000-character ceiling before model stages.
Stock Analysis now fairly shares the same context ceiling across exact front/back
excerpts of longer sources, retaining smaller sources complete. Original hashes
remain frozen; supplied-text hashes, included ranges, omitted counts and coverage
warnings are saved and exposed in the report and derived drafts. This is partial
source coverage, not exhaustive document review. Per-document extraction/OCR
failures still stop safely. Other research workflows keep their existing bounds.
No model, output-token cap or call count was increased. Existing frozen source
packs are not modified on resume.

Undelivered commands older than five minutes and collection snapshots older than
three minutes now show actionable nonblocking warnings, which clear on recovery.
Checks: 914 safe backend tests, 93 frontend tests, production build, unchanged model
policy; local extraction of the actual six eligible originals fits 159,999 supplied
characters. No model QA run was launched. Resume the existing authorized UNH job
only after deployed revision verification; preserve its deterministic report ID.

### T134 — Mac research-assignment delivery repair

UNH assignment e998e805-0bd4-49ce-aafa-2af123e1aace stalled before collection:
the LaunchAgent uses system Python 3.9 without Flask, while the collection manager
imported its planner from the Flask coordinator. The exception was logged only at
debug level; healthy heartbeats did not prove command delivery. Moved validation
to a standard-library-only shared module and made collection-sync failures visible
at warning level. Regression validation must include Python without site packages
and the actual LaunchAgent interpreter, not just the repository virtual environment.
The existing UNH command was replayed idempotently through the normal sync path;
its receipt now points to queued request bb20d70d-f5fe-4b66-a440-6f3ac38b7142.
Collection/research completion is still pending, not validated by that receipt.

### T133 — one end-to-end research assignment

Stock Analysis takes ticker/date range/output choices and coordinates managed
AlphaSense collection, iCloud STOCKS handoff, exact-byte cloud imports, the existing
bounded Stock Analysis engine and saved source-linked drafts. Stable child IDs,
PostgreSQL ownership, local cancellation tombstones and frozen source/baseline
checks protect restart and duplicate dispatch. It advances on Mac heartbeats even
when the browser page is closed. Existing policies and live theses are preserved.
Output defaults are editable. Summary note, stock summary and compact draft visual
share the full report; a thesis draft/update goes to the existing pending-review
inbox. Thesis updates are additive evidence proposals, not automatic reconciliation
or approval. Partial outputs remain accessible after a later failure.

Dependencies remain Mac awake/online, running Mac agent, scheduled Codex worker,
signed-in Chrome/AlphaSense, source permission/selection, and configured model
budget/key. Unknown paid outcomes require explicit retry acknowledgement. At most
eight selected originals; no exhaustive-market or fresh-consensus claim. See
`docs/research-assignments.md` and the updated collection worker runbook.

Synthetic validation covers actual stock worker to actual thesis inbox with a
mock model, temporary SQLite collection/import/cancel/replay and disposable
PostgreSQL recovery/baseline/hash/duplicate tests. Desktop/mobile checks at
1440/390/320px cover submission, progress and stop with intercepted API calls.
907 safe backend tests and 93 frontend tests passed; production build and unchanged
model-policy gate passed. Additional focused reruns cover final draft rendering.
No new live collection, paid research, thesis approval or portfolio mutation was
performed for QA. Render recovered from a transient rollout 502 and reports
`d0cbd5283cfb51c378f603ad8a222d71f8351d8c`. Cloudflare deployment
`c694081d-117e-477c-b783-c1dcbfa92198` serves byte-matching T133 HTML, bundle and
service worker from 13 whitelisted assets. Authenticated assignment/default reads
returned 200, unauthenticated access 401, invalid empty submission 400 before
dispatch. The Mac agent was restarted after zero active pipeline/collection checks;
RunAtLoad/KeepAlive remain true, recent stderr had no errors and the post-restart
heartbeat was healthy (12.4 seconds old at verification). Live user-authorized
AlphaSense-to-generated-deliverable acceptance remains outstanding; no paid QA
assignment was launched.

### T132 — automatic originals into Stock Analysis

New collector-approved STOCKS handoffs now enqueue exact-byte cloud imports.
`collection_original_sync.py` provides durable retry/lease state, agent-authenticated
non-overwriting imports and hash receipts. The Mac processes three per manifest
cycle; collection snapshots expose the last 100 records and Stock Analysis shows
status/failures with snapshot time. No paid research or thesis changes are triggered.
Validation: 895 safe backend tests and 93 frontend tests passed; production build
and unchanged-model policy gate passed. Synthetic browser checks at 1440/390/320px
verified queue failure visibility and no horizontal overflow. Render health verified `66f67aa4bb4c927e059886b6c19039e0b7b2f2ec`; Cloudflare
`bb214e10-3b92-402f-a1d5-26b281d1fca3` serves matching T132 bundle/service worker.
The six user-authorized SYK originals have exact-hash import receipts and all six
are eligible in the live Stock Analysis picker. No Stock Analysis generation was
started. The Mac agent was restarted with zero active pipeline jobs; its existing
startup imports may delay the first normal polling cycle.
Only explicitly replayed older handoffs are backfilled; CATALYSTS retains the
existing command import path. See docs/stock-analysis.md for dependencies/recovery.

### T131 — native Stock Research Studio

Stock analysis is a new investment-case workspace tab with Snapshot, Deep research
and Update thesis modes. The supplied Studio schema/doctrine now run through
Charlie's durable frozen-source worker in a separate stock_analysis_runs ledger.
Reports capture canonical company ID, case baseline, prior report, original hashes
and field-level citations. Six structured generation/review pairs checkpoint before
paid calls. Source changes and wording changes are compared separately; no thesis
breaker or thesis edit is automatically accepted. Draft handoff remains unsaved until
the investor reviews and saves it.

Nine-panel source-linked HTML infographics are rendered without an image model and
saved idempotently in existing Studio storage. Comparable positive historical series
have conservative charts with readable labels and bounded horizontal scrolling; unavailable or unresolved fields remain gaps. Analyst
figure review is required before saving a visual. Model review is not factual proof.
No new secrets or model defaults. Current source-pack bounds and manual restart/retry
rules are retained. Legacy Deep Research and Committee remain available.

Validation: 885 safe backend tests, 93 frontend tests, production build and model
policy gate passed. Disposable PostgreSQL tests cover worker/storage/restart and
synthetic browser checks at 1440/390/320px cover report generation, evidence,
comparison, visual persistence and unsaved draft handoff without page errors or
overflow. No paid research or production case saves were run for QA. Real-source
acceptance, licensed consensus feeds and live web retrieval remain outstanding.
See docs/stock-analysis.md for scope and recovery. T130 first integrated the feature;
T131 keeps chart labels readable with bounded horizontal scrolling on narrow screens.
The real built application route and chart rendering also passed at 1440/390/320px.
Render reports `07a5a3b590acb4f92df0b3bbd80af8fd02a493e5`. Cloudflare deployment
`f2282ddc-3fd5-403d-992f-dc7b727d1b7c` serves byte-matching T131 bundle and service
worker, and release HTML references the versioned T131 bundle. Live authenticated
SYK listing returns 200 with 16 sections, unauthenticated listing 401, and invalid
empty submission 400 before model execution. No production report or case was saved
for validation. Audio/pipeline checks found no active processing; the unchanged
September 21 legacy Summary Lab running entry was preserved. Mac agent is healthy
with unchanged model registry `e372116305335f31`; no restart was required.

### T129 — Explain upload connection handling

Explain dispatch, status polling and follow-ups now use the app API proxy, avoiding
an unnecessary direct cross-origin backend connection. Dispatch is asynchronous,
so the old long-request timeout rationale does not apply. Network failures now
preserve attachments and describe uncertain submission without automatically
retrying a potentially paid POST. Oversized uploads, unavailable backend responses
(including HTML) and expired sessions show actionable errors. Initial polling stops
on terminal client errors instead of waiting twelve minutes for an expired job.

Regression tests use synthetic screenshot payloads and simulated network/HTTP
failures; no paid explanation was run. The reported original browser failure was
not reproduced; direct backend CORS preflight was healthy during diagnosis.
Validation: 876 safe backend tests and 90 frontend tests passed; production build
passed. Live authenticated empty-input submission through the hosted proxy returned
400 before any model call. Render recovered from a brief rollout 502 and reports
`3bfbb1f081b46eb28ed8c8fd351016f56ce2364f`. Cloudflare
`05cf4a19-adb7-44d1-8038-7c67013422f2` serves T129 with byte-matching
versioned bundle and service worker. Unversioned bundle remained intermediary-cached;
the release HTML uses versioned URLs. Mac agent unchanged; no restart required.

### T128 — historical balance sheet and investment committee

Operating scenarios now accept optional fiscal-year-end cash and total debt, with
explicit definitions, manual references or reviewed-source receipts. Both use the
model year/currency; net debt is computed only after scope comparability is confirmed.
Zero is supported; negative cash/debt are rejected. Forecast debt assumptions are
unchanged. New saves and previews recheck original permissions/hashes and linked
value/year/currency/definition; historical exports preserve receipts.

Committee is a separate workspace tab and durable `investment_committee_runs`
ledger, using the existing bounded research worker infrastructure. Five first-pass
roles (lead, upside, downside, accounting/evidence, valuation/expectations) see only
the same frozen originals and case, not other role outputs. Five evidence checks,
one linked challenge round and one lead response follow (up to 12 calls, 6,000 output
tokens each, plus explicit retries). Source text remains bounded to 160,000 characters
and eight originals. Same configured research model and monthly budget checks;
usage is recorded as investment-committee. These are roles of one model, not
independent humans. Lead responses are unverified draft interpretations; original
dissent remains visible. No automatic case changes or approval controls exist.

The shared worker now checks original hashes as well as permissions before calls.
Submission identity, per-ticker unfinished-job exclusion, database worker ownership,
pre-call reservations, saved stages, stop/resume and unknown-outcome acknowledgement
apply independently to each ledger. Interrupted jobs require explicit resume; there
is no automatic paid retry. Baseline drift is flagged; completed runs are immutable.
Committee IDs cannot be used as research receipts or proposal handoffs. Exports
preserve partial state, source hashes, dissent, challenges and unaccepted responses.

Validation: 876 safe backend tests and 87 frontend tests passed, with Python
compilation and production build. Synthetic browser checks at 1440/390/320px cover
cash/debt source linking, net debt, confirmation reset, committee findings, dissent,
source hashes and export; no overflow or page errors. Model-policy gate passes with
no protected defaults changed. Render recovered from a brief rollout 502 and reports
`72743fa21ddb2eb060d7dd42c7364f19c1382d43`. Cloudflare
`9665d04f-c122-4be1-bba7-e96f923a1af5` serves byte-matching T128 bundle/service
worker. Live authenticated committee listing confirms five roles and the new schema;
invalid submission returns 400 and unauthenticated listing 401. Synthetic nonpersistent
preview verifies historical net debt 200.00 and unchanged 21.95 share value; missing
scope confirmation returns 400. No paid committee or live investment-case saves
were used for QA. Predeployment audio/pipeline/SYK research checks found no active
jobs; the unchanged September 21 legacy Lab row was preserved.
Mac agent and model registry unchanged. Next: user-selected real-source acceptance
and comparison with the existing single-reviewer research on identical originals;
additional financial metrics, signed source observations and deeper sector templates.

### T127 — itemized EBITDA reconciliation

An optional bridge now records a signed starting EBITDA, its definition/source,
and up to 20 signed adjustments with source/rationale and recurrence classification.
Decimal arithmetic must exactly reconcile to the positive historical EBITDA used
by the operating model. Duplicate names, missing references, zero/nonfinite
amounts and mismatched totals fail preview and save. Calculated totals are replaced
on the server, not trusted from the client. Existing valuation formulas are unchanged.
Manual references do not constitute verified source receipts or GAAP reconciliation;
recurring/uncertain adjustments remain visible with a sustainability warning.
Edits reset confirmations and hide stale results. Saved exports retain the bridge.

Validation: 868 safe backend tests, 84 frontend tests, Python compilation and
production build passed. Synthetic real-component browser checks at 1440/390/320px
cover signed arithmetic, confirmation reset, mismatched totals, removal and layout.
No paid research or live case writes were used for QA. Render recovered from a
brief rollout 502 and reports `e4b40b7700d1096b769dd1eeafd0630ba300c1a7`.
Cloudflare `9e4c2aa8-f6b2-4da7-a32d-c1704073f6a9` serves T127 with byte-matching
hosted bundle/service worker. Authenticated nonpersistent synthetic preview verified
180 + 20 = 200 and unchanged 21.95 share value; mismatch returns 400, no auth 401.
Predeployment audio/pipeline checks showed no active jobs; the unchanged September
21 legacy Lab running row was preserved. Mac agent unchanged; no restart required.

Downloads access was verified on October 7 after the investor enabled it. Fifteen
permission-only requests (DGX, DHR, DLR, DOV, ELV, ETN, GD, GILD, HCA, HON, HST,
HUM, IDXX, IQV, LH) were resumed as queued with frozen windows/progress intact.
This is queue recovery, not completed collection. Issuer-mismatch and other pauses
remain for separate verification. Next: broader source-linked financial observations
and independent committee review, plus a user-selected real-source acceptance pack.

### T126 — industrials / medtech historical EBITDA baseline

The investor selected industrials/medtech as the first model expansion. Operating
scenarios now optionally accept positive consolidated base-year EBITDA, a source
reference and explicit historical/forecast definition comparability. Decimal
calculations show historical EBITDA margin, target-minus-historical margin in
percentage points and total (not annualized) EBITDA growth over the forecast.
Existing EV/EBITDA valuations and saved models without a baseline are unchanged.
This is not GAAP-to-adjusted reconciliation, a loss/recovery model, or automated
financial extraction. Meaning and comparability remain analyst attestations.

The existing reviewed-passage flow now supports EBITDA as well as revenue, with
separate receipts. Linking EBITDA requires the model year/currency to match and
sets its definition; changing value/year/currency/definition invalidates the link.
Preview and each new save recheck source permission/hash and derive receipt fields
on the server. Exact save retries replay the existing revision. Historical restore
retains the receipt without re-certifying current availability. Parenthesized
numbers are excluded from positive-number candidates rather than turning losses
into positive EBITDA. Unlink keeps the value as a manual input. Saved investment
map exports preserve both evidence receipts and the historical comparison.

Validation: 864 safe backend and 83 frontend tests passed; synthetic real-component
browser checks cover 1440/390/320px, arithmetic, stale-result hiding, definition
confirmation, evidence link/unlink and no page errors/overflow. No production
cases or research were saved, approved or generated for QA. Python compilation and production build pass. Render recovered from a brief rollout
502 and reports `6d87a60ba9f6fa871c741774c1aff3eb2228954e`. Cloudflare
`fe8baa60-474b-40d8-a370-0cd9b02688e2` serves byte-matching T126 bundle and
service worker. Authenticated nonpersistent synthetic preview returns 20% base
margin, 21% total EBITDA growth and unchanged 21.95 share value. Invalid evidence
returns 400; unauthenticated access returns 401. No active audio/pipeline jobs
preceded deployment; the unchanged September 21 legacy Lab row was preserved.
The Mac agent is unchanged and was not restarted. Next: explicit reported-to-adjusted EBITDA reconciliation and
broader financial observations, followed by independent committee review. A real
user-selected acceptance pack and macOS Downloads access remain outstanding.

### T125 — saved credential recovery

Automatic Summary Lab and legacy Improved-note recovery now prefer the server
Anthropic key and fall back to the existing saved Settings `apiKey`. The lookup
runs each sweep, so rotation/removal takes effect without a restart. It does not
persist keys on jobs or change eligible jobs, models, checkpoints, ownership,
concurrency or retry limits. Missing keys start nothing; database lookup failure
leaves jobs intact and is logged by exception type only.

Validation: six synthetic credential/recovery regressions, 860 safe backend tests
and 82 frontend tests passed. The broad suite still emits existing mocked Lab
worker exceptions from incomplete SQL fixtures; no live provider work was run.
Python compilation and production build passed. Render reports `14dd0278b08b04ad4ff3160cacb6b1af1d2e166a`; Cloudflare `b5e0eadd-175b-41ba-87b1-dad622af2471` serves byte-matching T125 bundle and service worker. No originals or research were modified. Downloads access remains denied
by macOS; restore Codex Files and Folders permission before resuming collection.
Next: complete a user-selected real-source acceptance journey, then broaden typed
financial observations/accounting reconciliation and sector templates before the
independent committee. Collection failure is not successful empty coverage.

### T124 — local assistant thesis handoff

`thesis_handoff.py prepare` captures the live baseline and inventories a source folder
into a private workspace outside watched source folders. The assistant reads selected
originals and writes the draft; `submit` validates, freezes the exact package before
upload, reads back its fingerprint, and saves a direct review receipt. Uncertain
uploads retry idempotently; changed baselines/sources block fresh submission. A local
file lock prevents simultaneous submission in one workspace. No approval or model
endpoint is accessible through this helper. Credentials use the existing Mac agent
convention and only the fixed production origin; redirects are refused.

`?thesisDraft=UUID` selects a specific review independently of the latest-100 list and
survives reload and app hash routing. Investor approval remains explicit. The UI and
AGENTS.md explain the assistant path; ordinary chats without local access retain the
manual file workflow. See docs/thesis-imports.md for recovery and dependencies.

Validation: 853 safe backend tests plus a new passing helper/disposable-SQL integration
test, 82 frontend tests, compilation and production build
pass. Synthetic browser/disposable SQL checks cover reload of the exact selected draft
and built-shell direct links at 1440/390/320px, without JavaScript errors or overflow.
The real SYK handoff preparation discovered an already approved draft; no duplicate
was submitted and no live thesis was changed. No paid model calls or original uploads.
Deployment verified: backend health reports `6b6d63c5cb53b718cce80a3d7340b9107f409411`.
Cloudflare deployment `fc3b6b42-5e99-4648-8f2c-040c9a8fa97f` serves T124 with matching
hosted bundle and service-worker bytes. Authenticated draft list/preparation return
200 and unauthenticated requests return 401. Only whitelisted assets were deployed;
active audio jobs were zero beforehand. The local agent was unchanged. A brief backend
rollout interruption recovered before final verification. SYK's existing approved
review matches the full-source draft's four research sections; no duplicate submitted.
Future requests: ticker + accessible source folder + research instructions, then investor
review/approval through the returned link. Requires the configured local Mac assistant.

### T123 — external thesis draft import and review

Companies → Import & review thesis (also linked from the Thesis toolbar) offers
Prepare for ChatGPT → Import draft → Review changes → explicit investor approval.
Preparation exports the current detailed thesis and a version fingerprint. JSON
imports support new names and upgrades, including the earlier proposed-v1 SYK file.
A durable inbox keeps pending/approved/dismissed drafts and source/provenance metadata.
Complete before/proposed sections show removed items, with readable desktop/mobile
layouts in black Calibri. References and supplied hashes are not certified originals.

Approval checks the unchanged baseline, then saves the native detailed thesis,
restorable history, evolution snapshot and idempotent receipt in one transaction.
Duplicate uploads reopen the same review; duplicate approvals replay the receipt.
Concurrent edits block stale approval; failures leave drafts recoverable. No model
calls or automatic downstream research, scorecard or report generation occur.
Full/condensed theses, scorecards and other derived outputs remain separate, with
an explicit warning to review/update them. Existing API execution is unchanged.
See `docs/thesis-imports.md` for format, lifecycle, dependencies and limitations.

Validation: 846 safe backend tests and 81 frontend tests passed; focused tests use a
new disposable PostgreSQL cluster (never the existing or production database) to
exercise restart, concurrent uploads/approvals, conflicts and forced journal rollback.
The browser flow passed with synthetic research and disposable SQL, including restart,
retry and conflict screens, plus the actual built app shell at 1440/390/320px.
Python compilation and production build pass. The private SYK JSON passed local format
validation; no real thesis was imported or approved for QA. Release verification is
recorded below.

Deployment verified: Render `/health` reports
`b27fb44db7dfc749e77c86c3eb4f9fb2ff8bd2c3`. Cloudflare deployment
`b9e0f3d3-2a3c-43d7-b291-ee127a7d013a` serves T123 with matching hosted
bundle and service-worker bytes. Authenticated draft listing and SYK preparation
return 200; both reject unauthenticated requests with 401. SYK preparation reports
initial mode; the production inbox is empty. No real draft was imported/approved,
no original was uploaded and no paid model QA was launched. Active audio jobs were
zero before deployment. Only whitelisted assets were uploaded. The local agent was
unchanged and did not require restarting. Next user action: import the full-source
SYK JSON, review it and approve explicitly in Charlie when satisfied.

### T122 — reviewed model defaults and recurring maintenance

`model_registry.json` centralizes primary backend/Mac workflow defaults, picker
choices, request settings, text prices and announced retirement dates. Sonnet 5.5
runs routine work and Lab checks with no up-front thinking (`between_tools`, high
effort); GPT-6 Luna handles bounded OpenAI validation/fast fallback with reasoning
off. Opus 4.6 stays the research/composition default. Opus 5.5 and GPT-6.1 Sol are
optional choices. Existing saved preferences and environment overrides stay pinned.
Browser recap/direct analysis obtains its settings from the authenticated API.
OpenAI fallback now sends native PDF parts instead of discarding the source.

Settings → AI models shows the deployed policy, sources, changes, retirement
notices and durable maintenance check status. Unknown ledger prices are NULL and
flagged separately; new shared adapter/Lab usage includes cache accounting and
registry revision, including GPT-6 long-context pricing. This remains an estimate,
not a complete invoice for tool fees, images, browser-direct or all Mac/helper calls.
Lab checker identity is checkpointed; partially checked legacy runs retain their
old checker. Known retired selections stop rather than silently remap a pinned job.

A daily Codex heartbeat `charlie-model-maintenance` is ACTIVE in the existing chat.
Its exact procedure is in `docs/model-maintenance.md`. It checks official sources
and account catalogs, documents migration evidence, applies the offline cost and
compatibility gate, runs safe tests and deploys eligible routine upgrades. Protected
research/image defaults or uncertain price/quality changes require attention. It
needs this Mac, Codex, network and deployment credentials; it is not a Render cron.
Unchanged checks only update the durable status, with no unnecessary deployment.
Interrupted releases resume from their recorded stage. No paid research QA is allowed.

Validation: 836 safe backend unit tests, 78 frontend tests, Python compilation and
production build pass. Real SDKs were exercised with synthetic HTTP responses for
Sonnet thinking options and OpenAI PDF/reasoning parameters. Policy tests reject
missing evidence, increased cache/token costs and protected research changes.
Synthetic panel checks at 1440/390/320px verify layout and error visibility.
No paid model quality evaluation, research approval, original-file edit or email.
Deployment verified: Render `/health` reports
`1f5db725e165f8fe54bba8a84038a480a9cfb607`; Cloudflare version
`362b7d36-7c08-4157-ba8f-000948eafbb1` serves matching bundle bytes and
`20261005-122` service worker. Authenticated model policy, picker, Lab default and
usage endpoints return 200; unauthenticated policy/picker requests return 401.
The deployed registry is `e372116305335f31`, with Sonnet 5.5 request options
confirmed and Opus 4.6 retained. The initial maintenance report is persisted.
The Mac agent was restarted; its heartbeat reports that same registry revision,
and its stderr has no recent traceback. Its separate Python 3.9 / Anthropic 0.86
runtime passed a synthetic request-serialization check (backend/dev SDK is 0.96).
No current transcription/pipeline jobs were active at rollout. One legacy v3 Lab
run has been labeled running since September 21 with no updates; it was preserved.
The October 23 GPT Image 1 / o4-mini retirement choices remain attention items;
there is no approved automatic image/agent replacement with proven equal task cost.

### T121 — Meeting Notes

Adds a separate shorthand Meeting Notes section to original Summary and Summary Lab.
The shared prompt in `meeting_notes.py` preserves substantive coverage, figures, qualifiers,
attribution and open questions while using fragments, bullets, familiar abbreviations and
occasional arrows. It forbids invented first-person participation, unsupported mental-state
claims, external legal corrections and converting uncertainty into certainty. The supplied
CI notes were used as a style reference only; no private source bodies are committed.

Original paste/document/audio/YouTube/podcast generation persists `meeting_notes` through
an additive schema migration. Existing rows are not backfilled. An unavailable extra pass
leaves the other sections intact; the saved Summary offers Generate Meeting Notes, which
runs only this section from saved source text. Its transaction advisory lock rejects concurrent
submissions and reuses existing output after a successful retry. A backend failure after a
model response but before commit can still require another model call; no exactly-once
provider guarantee is claimed. Existing clients omitting this field do not erase saved notes.

Lab v5 adds this section to source review/revision/checkpointing and all section actions.
Korean-only output includes Korean Meeting Notes; English/bilingual use English notes.
Older Lab runs retain their existing section plan in the UI and are not silently regenerated.
An older interrupted Lab run still follows the existing version guard (start a new experiment).
Both workflows support section copy/email/iCloud Word and full-document/bulk exports.
Note typography is black Calibri with fallbacks. Model usage increases by one original
composition pass and the Lab's additional draft/check/revision passes; no paid QA was run.

Validation: 824 safe backend unit tests, 76 frontend tests, Python compilation and production
build passed. Mocked browser checks cover Summary/Lab display at 1440/390/320px, no
horizontal overflow or page errors, and the saved Summary section-only generation button. Synthetic tests cover source input, empty provider output, SQL parameter
shapes, isolated export, legacy compatibility, section-only retry, concurrent requests and
Lab checkpoint reuse. Real generated-note quality remains for a user-directed source run.

Deployment verified: backend health reports `f568654ff7350e51bfefca27c2380a096d25a333`;
Cloudflare deployment `5f629f48-b15d-4098-a9dc-867ccf414eab` serves T121, matching
the local bundle hash and `20261002-121` service worker. Authenticated Summary listing
returns 200 with the new field on all 443 existing rows. Missing-summary section generation
returns 404 without a model call; unauthenticated requests return 401. Zero active audio
jobs were reported before deployment. No existing sources regenerated, emails sent, or
production note content modified for QA.

### T120 — revenue observation → operating model → case revision

Operating scenarios can select a printed number from a completed same-ticker
Deep Research run's supported reported-fact passage. `financial_observations.py`
rechecks exact passage identity and original content/permission, preserves the
printed token, converts units/thousands/millions/billions to millions, and requires
explicit analyst confirmation of issuer, consolidated annual revenue, fiscal
year, currency, accounting basis and original page/section. This is not automatic
financial-statement extraction, accounting reconciliation or an independent
verification of metric meaning. Guidance/estimates/unresolved claims are excluded.

The revenue amount, year and currency populate an unsaved model. A server-derived
receipt retains research/claim/source identities, excerpt, original/extraction
hashes and the specific checks. Preview and save recheck source permission/hash
and reject mismatched model revenue/year/currency. New case saves recompute the
receipt; forged client provenance is discarded. Exact retry receipts replay
without duplication. Historical restoration retains historical evidence without
claiming current source availability. Unlink is explicit and keeps the number
as a manual input. Existing models without this optional field remain compatible.

The in-app **Ticker-to-case workflow & test guide** explains all nine steps,
expected states, paid actions, recovery and current limitations, and downloads a
black Calibri HTML guide. Its isolated synthetic calculator check uses only the
nonpersistent preview endpoint, checks known outputs and zero-share rejection,
and never saves company data or calls a model. The same guide is checked in at
`docs/charlie-ticker-workflow.html`. Revision history now exposes operating-model
metadata alongside existing scenario/signal/evidence data.

Validation: 816 safe backend tests, 75 frontend tests, Python compilation and
synthetic browser flow. Browser checks cover source selection, exact token/unit
conversion, receipt rendering, valid arithmetic, stale-link rejection and the
isolated guide check; no overflow at 1440/390/320px and no console errors. Tests
cover changed/restricted/missing originals, unsupported claims, wrong ticker,
forged receipts, duplicate saves, unit/period mismatch and historical restoration.
No real research/observations/cases were created or accepted for QA. Production
build passes. Render health reports `b53d4e22f7cacfb2cd6468d9890bb20ce69d9ee7`;
Cloudflare deployment `ba2fcdd7-ef1d-4c31-a493-ae13f2d47615` serves T120 with matching
hosted bundle hash and service-worker version. Live ticker entry and guide-to-model
navigation passed. The hosted guide's isolated calculator check passed, including
zero-share rejection. Authenticated missing-evidence input returns 400 and the
unauthenticated observation endpoint returns 401. Full evidence attachment/save
was tested with hermetic synthetic API fixtures, not a real production case.
Paid research generation/quality was not exercised.

Next: richer typed metrics and accounting reconciliation, sector templates,
then bounded committee review against frozen research/case/model references.
Full paid research quality still needs legitimate investor-directed research;
synthetic checks are not real-source validation.

### T119 — empty operating-model workspace

Live T118 UI verification caught a null preview access when no model existed.
The guarded preview lookup now supports opening an empty case and removing a
model from the unsaved draft. A real JSX rendering regression covers empty,
newly added and removed drafts. No saved case data was affected.

Verified production revision: `882bbbe6145b39a42cc6db29bb10d580ad2ffff4`.
Render `/health` reports this revision with PostgreSQL healthy. Cloudflare version
`692a8a91-9f13-485b-b39f-fd4f154377db` serves T119; hosted bundle SHA-256 matches
local `dist/app.js`, and service worker is `20261001-119`. The hosted full app
successfully opened AAPL's empty Operating scenarios tab, added an unsaved model,
and removed that draft. No case was saved. 72 frontend tests and production
build pass; unchanged backend last passed all 808 tests for T118. Only whitelisted
assets were uploaded. No paid research or production data mutations used for QA.

### T118 — deterministic operating scenarios

Investment cases → Operating scenarios adds an optional EV/EBITDA model with
analyst-entered revenue CAGR, EBITDA margin, valuation multiple, net debt, other
senior claims, nonoperating assets and diluted shares. Shared currency, millions
and fiscal-year boundaries are explicit. This first template requires positive
EBITDA; it is not appropriate for banks, insurers, REITs or recovery models.
Existing EPS/P-E sensitivities remain unchanged in Current thesis.

`operating_model.py` validates and computes with Decimal; no model/provider call
is used. `/api/research/operating-model/preview` is nonpersistent. Case saves
recompute outputs, ignore client-supplied calculated values, and freeze normalized
inputs and `ev-ebitda-v1` results in the same immutable investment-case revision.
Source references remain manually entered and unverified. Changes invalidate
proposal context hashes and hide stale draft previews. Investor explicitly saves.

Snapshot, historical comparison and HTML investment-map exports include the
selected revision's scenario inputs and outcomes. Sensitivities perturb base
EBITDA and multiple independently. Reverse EBITDA holds the equity bridge and
multiple fixed; it is not observed consensus. Negative equity is explicitly
floored at zero and flagged, not represented as a negative share price or a
restructuring forecast. No probabilities, dividends, FX or discounting included.

Validation: focused arithmetic, reverse bridge, source-definition requirements,
unit/period rejection, negative-equity handling, preview purity and immutable
input normalization tests pass. 808 backend tests and 71 frontend tests pass;
synthetic 1440/390/320px checks cover editing, calculation, stale-output exclusion,
save payloads, invalid periods and overflow. No production models/cases or paid
research were created for QA. T118 backend health matched `5f7e981` and hosted
assets matched the build. Live nonpersistent arithmetic returned the expected
21.95 synthetic share value; the empty UI issue is fixed in T119 above.

Next: reconcile source-extracted typed observations before treating any model
input as verified; add sector-specific templates, then bounded committee review
against frozen research, case and model revisions. Real-source research quality
validation still requires an authorized user-selected pack, not paid testing.

### T117 — scheduler database-driver compatibility

Startup verification of T116 caught SQLAlchemy's newer default PostgreSQL driver
trying to import uninstalled psycopg v3. The scheduler now explicitly selects
Charlie's installed psycopg2 driver for generic PostgreSQL URLs, preserving
credentials and connection options. Other explicit drivers remain unchanged.
An isolated test verifies URL preservation without importing the app, connecting
to a database or starting any scheduled work. Frontend assets match T117.
Verified October 1, 12:52 UTC: Render `/health` reports
`f8a9f54adfaafa2252ace95d9cf3ca61bc3a4e57`; its startup log confirms
"Media tracker scheduler started". Cloudflare version
`c03418a5-8f61-4eb7-8980-032ec6f5c8da` serves T117. Hosted app bytes match
the local bundle; the service worker matches. Authenticated research listing
works through the Worker proxy. Live read-only navigation reaches Deep Research
and correctly disables submission with no selected originals. Full research
execution and generated report quality have only synthetic validation so far.

### T116 — durable source-backed Deep Research

Investment cases → Deep Research selects up to eight permitted stored originals,
freezes their byte hashes and the exact case baseline, and saves 22 structured
sections in six bounded model calls followed by six claim-review groups. Exact
passage matches are checked deterministically; model support review remains
separate from fact/interpretation classification and investor approval. Missing
coverage and unresolved claims remain visible. Source permissions are checked
before every stage. No web collection or paid QA is performed by this workflow.

PostgreSQL checkpoints precede every provider call. Duplicate submissions reuse
an ID, worker ownership prevents duplicate execution of a run, and uncertain
provider outcomes require an explicit paid-retry acknowledgement (bounded to
three such retries). Completed stages survive restart; recovery is explicit.
Stopped/partial runs remain readable. No automatic provider retries or fallback.
Full HTML reports retain unresolved claims; compact investment maps omit them.
Both exports identify source hashes, research revision and case baseline.

Completed research hands its originals to the existing reviewed thesis proposal
workflow with a durable researchRunId. Changed originals or case baselines block
that handoff. Initiation can prepare an unsaved case draft; the investor must
review and save it. Research completion never accepts a note or thesis change.

Render's prior build failed because floating TradingAgents HEAD required Google
SDK v2 while Charlie uses v1. Freeze TradingAgents 0.5.1 at its release commit and
langchain-google-genai 4.2.0. A clean dependency resolution dry run passed locally;
Render deployment must still be verified. No current environment was upgraded.

Validation: 802 safe backend tests and 69 frontend tests passed; isolated API
fixtures cover idempotency, restrictions, stale baselines, stop/recovery and
immutable completion. Synthetic browser checks at 1440, 390 and 320 px cover
selection, submission, recovery controls, source inspection, initial-draft
handoff and unresolved export exclusion. No paid or authenticated live model
run was used. Backend revision `4bbb545` is live and authenticated research listing returns
22 sections; unauthenticated requests return 401. No live research was started.
Frontend publication is bundled with the T117 scheduler compatibility fix.

Next: typed financial facts and a deterministic scenario engine, then bounded
committee synthesis and monitoring. Automatic issuer resolution, complete
historical/consensus coverage, cross-report reconciliation and simulation are
not implemented by this increment. See `docs/integrated-investment-workspace.md`.

### T115 — saved company Snapshot foundation

Research desk → Investment cases & portfolio now opens a Snapshot of the selected
company's saved investment case. It shows thesis, expectations, variant view,
change conditions, assumptions, current-wording excerpt links and missing case
fields. Historical snapshots and explicit earlier-revision wording comparisons
share existing immutable case history. HTML investment-map exports retain the
selected revision, evidence wording, uncertainty labels and black Calibri text.
Update thesis leads to the existing source-selection and reviewed-proposal flow;
it does not submit research or accept changes. A successful case save stays
available to Snapshot even if the subsequent history refresh fails; unavailable
save dates are explicitly labeled rather than invented.

This is the first foundation increment, not completion of the integrated research
workspace. No new backend, database migration, full 22-section research generator,
issuer resolver, committee or financial engine is introduced. Next: durable
structured research revisions with source inventories, section coverage and an
explicit case baseline, followed by the Deep Research workflow. See
`docs/integrated-investment-workspace.md` for the full sequence.

Validation: 791 safe backend tests and 67 frontend tests passed. Frontend checks cover saved identity, stale excerpt
exclusion, assumption additions/removals, and escaped versioned exports. Synthetic
browser checks cover desktop/mobile comparison, update navigation and historical
export; no paid research or investor acceptance was used. Local agent unchanged.

Deployment observation October 1, 12:14 UTC: source commit `27f4088` is pushed
and Cloudflare serves T115. Hosted frontend SHA-256 matches the local build and
the service worker reports the matching release. Render remains healthy but
still reports `195fd14` after several minutes; its redeploy is not confirmed.
This increment has no backend changes and uses existing endpoints. Do not claim
an authenticated live research run was tested; browser verification was synthetic.

### T114 — issuer ambiguity and follow-up screening

Leadership headlines without an explicit leading ticker now require review before
collection. A related-company feed does not establish which employer changed its
leadership. This deliberately also holds company-name-only leadership headlines;
full issuer-name resolution is still outstanding. Explicit ticker-led leadership
announcements retain the existing automatic path. Plural question headlines and
undervaluation/overvaluation follow-up questions are held as commentary. Held
signals remain visible and consume no automatic collection quota. No existing
research or pending requests are rewritten by these guards.

Operational inspection September 29: the latest 50 requests initially included
25 queued, 17 attention, seven completed and one collecting. AMAT's two saved
originals were confirmed under STOCKS/AMAT despite an incorrect status-message
abbreviation. AMAT's failed export and Agilent's now-current date-window request
were resumed with partial progress retained. This is queue resumption, not proof
of collection or analysis completion. Duplicate and issuer-mismatch holds remain.

Validation: 791 safe backend tests and 63 frontend tests passed; focused tests
exercise review visibility, quota preservation and subsequent valid dispatch.
No paid model runs, email or research acceptance were used for QA.

### T113 — repeat-headline collection guard

Exact normalized headlines for the same ticker/category within three days now
remain visible for duplicate review rather than automatically collecting again.
Generic clinical headlines still require a named clinical signature. This does
not establish event identity across differently worded follow-up articles.
Same-day nonclinical similarity no longer suppresses a new headline when its
numeric anchors differ, protecting changed guidance/authorization amounts.
Existing assignments and saved research are untouched. No collection is launched
by this release. Review-only signals cannot trigger the new cross-day hold.

Validation: 789 safe backend tests and 63 frontend tests passed; Python compilation
and production frontend build passed. Synthetic identity and dispatcher tests cover next-day republication,
review visibility, linkage to the original command, no quota use, changed amounts,
issuer separation and the bounded window. Live unattended end-to-end collection
and synthesis still require the proof described below.

### T112 — commentary screening before automatic collection

Catalyst discovery retains valuation commentary and industry/peer comparisons as
reviewable signals instead of automatically launching AlphaSense assignments.
These signals consume no automatic event quota. This is a bounded headline
heuristic, not entity resolution or proof of novelty; genuinely new event
announcements remain eligible under the existing rules. Existing requests are
not cancelled or rewritten.

Operational ledger inspection: LLY, BMY and MRK event collections are complete;
ABBV and BSX remain attention pending primary-event confirmation. JNJ remains
attention for inconsistent AlphaSense filters and a restricted broker source.
Other MRK requests have been held as possible duplicates. Collection completion
does not establish recap completion or accepted thesis changes.

Validation: 785 safe backend tests and 63 frontend tests passed; Python
compilation and production build passed. Focused catalyst tests cover review
visibility, no collection or quota use for commentary, and a subsequent genuine
approval still being queued. No paid research or email was used for QA.

### T111 — durable audio branch and collection safeguards

Opted-in SUMMARIES recordings now save the original Summary, completed audio-job
record, and queued automatic Lab experiment in one database transaction. The Lab
uses a stable audio-job identity, captures its source/baseline and prompt version,
and starts only after commit. Existing bounded recovery can find it after a
restart. A success-alert failure no longer stops the Lab branch or marks the saved
note as failed. Original and Lab prompts, output settings, and the current `**`
filename opt-in rule are unchanged. No historical recordings are backfilled.
Recovery after restart still requires the server environment API key; client-only
Settings keys cannot be recovered after process loss. No keys are persisted here.

AlphaSense workers now have a read-only `preflight --request ID --owner TOKEN`
check for live ownership, request state and enabled ticker policy. Lease renewal
and completion enforce those checks; completion checks again before reserving
paid dispatch and recording success. Explicit manual refreshes remain allowed
when the recurring policy is paused. A pause cannot undo an already sent request;
dispatch reservations still require inspection instead of blind retry. The worker
runbook distinguishes unavailable browser control from an actual signed-out/MFA
screen, and directs workers to check the supported native Chrome app surface.

Operational check September 26: native Chrome was controllable and AlphaSense
signed in. The existing ABBV regulatory request's September 20–26 press-release,
broker and transcript searches were reviewed. No matching approval-event source
was selected; unrelated results and pre-approval background were recorded as
exclusions. Request remains attention pending primary-event confirmation, with
zero downloads and no synthesis dispatched. Four existing LLY/BMY/MRK/JNJ requests
paused solely for unavailable native Chrome control were requeued after confirming
access. This is resumed work, not proof that downloads or analysis completed.

Validation: 783 safe backend tests and 63 frontend tests passed; Python compilation and production build checked. New synthetic
transaction tests cover rollback, committed recoverable rows, stable identity,
version preservation, and no automatic retry of failed experiments. No paid
research, email or thesis acceptance was used for validation. A future user-driven
opted-in recording remains the live proof of the new atomic-save path.

### T110 — faster heat-map prices

Supersedes T109's serial 40-stock downloads. The browser now uses three parallel
20-stock requests and a 15-minute session-only market-price cache, keyed by API,
period and ticker. The backend shares histories by ticker across overlapping
portfolios/indexes, coalesces simultaneous same-ticker downloads, and caps provider
concurrency at eight workers across requests. One-day tiles fetch five days of
history instead of two years; longer periods fetch/reuse two years. Missing prices
are not cached as successful results and failed batches remain retryable. Abort
checks prevent old views updating the current map. Per-quote fetch dates preserve
cache freshness. This still uses daily adjusted prices, not real-time streaming.

Validation: 774 safe backend tests, 63 frontend tests, compilation and production
build. Desktop/mobile synthetic browser checks cover all universes and failures.
A live local eight-stock sample loaded in 0.71 seconds; three overlapping stocks
then reused the server cache in under 0.01 seconds. This is not a full-index or
production latency guarantee. No private portfolio data or research was changed.

### T109 — heat map universe choices

The Universe dropdown now includes My portfolio, SPX (S&P 500), Nasdaq (Nasdaq-100,
confirmed by the user), RLV (Russell 1000 Value), and RLG (Russell 1000 Growth).
Index views use explicitly labeled iShares ETF equity holdings proxies: IVV, IQQ,
IWD, and IWF. These are not official index constituent feeds. Public issuer CSVs
provide dates, companies, sectors, and market values; weights are derived from
market values to retain small positions rounded to 0.00% in published weights.
Cash/derivatives are excluded. Holdings older than seven days display a warning.
Issuer files are cached for six hours; errors provide retry and never replace
saved portfolio holdings. Index holdings cannot be saved over the user's portfolio.

Prices load progressively in 40-stock batches using the existing authenticated
returns endpoint. Switching universe or period aborts old work; partial price
failures are visible with retry. No 100-stock truncation is applied to index views.
Published source dates and ETF proxy explanations remain visible.

Validation: 772 safe backend tests and 61 frontend tests passed. Production build
and compilation passed. Desktop (1440px) and mobile (390px) browser checks used
real public holdings with synthetic quotes, covering all four choices, full tile
counts, issuer failure/retry, partial-price retry, and portfolio preservation.
Live issuer checks returned 504 IVV, 101 IQQ, 870 IWD, and 371 IWF equity rows dated
September 24. No portfolio writes or paid research were performed for QA.

### T108 — portfolio heat map

Today → Portfolio heat map (`#view=heatmap`) is a separate holdings workspace. The
investor enters a dated portfolio name and percentage weights, or imports CSV/TSV.
Research coverage is never silently promoted to holdings. One shared snapshot persists
in PostgreSQL (`portfolio_heatmap_snapshot`); optimistic revision checks prevent one
device from overwriting newer edits. No real holdings were entered during QA.

Tiles use absolute position weights, grouped by editable sector, and signed adjusted
stock returns for color. Shorts retain negative weights; colors still show the underlying
stock return, not short P&L. Seven daily-data periods, search, sector filtering, a mobile
list, accessible blue/orange colors, per-tile dates, company navigation, and CSV export
are included. No portfolio-return or attribution claims are made from snapshot weights.

Market data uses the existing yfinance dependency through a bounded authenticated API.
Dividend/split-adjusted daily prices are cached for 15 minutes, with history reused across
period switches. Missing prices remain unknown; insufficient full-period history is not
substituted with an IPO-to-date return. Returns expose exact observation/baseline dates.
1m/3m/6m/1y use 30/91/182/365-day boundaries, YTD uses previous year-end, and 1d uses the
latest two daily observations. This is not an extended-hours or streaming feed.

Validation: 767 safe backend tests and 61 frontend tests passed; Python compilation and production build passed. Full-app browser
checks at 1440px and 390px used synthetic API fixtures and verified layout, tile selection,
period changes and saving without JavaScript errors. One live read-only ABT quote check
returned Sep 25 versus Sep 24 observations. No production holdings or research were changed.


## Product intent

Charlie is intended to become the primary research and decision-support workspace for a professional equity investor. It should ingest evidence, preserve source provenance, synthesize information in the investor's voice, maintain investment theses longitudinally, prepare meetings, coordinate bounded research agents, and automate repetitive work without silently making investment decisions or accepting research changes.

The design standard is institutional: concise hierarchy, readable outputs, defensible claims, explicit uncertainty, durable recovery, and usable desktop/mobile interfaces.

## System map

| Layer | Current implementation |
| --- | --- |
| Frontend | React 18. Most legacy UI and routing remain in `src/app.jsx`; newer workspaces use separate `src/*.jsx` modules. |
| Frontend build | `build-frontend.sh` runs Babel with required block-scoping transformation, then esbuild. Generated assets are committed in `build/` and `dist/`. |
| Edge | `worker.js` hosts static assets on Cloudflare Workers and proxies `/api/*` to Render. `service-worker.js` manages client caching. |
| Backend | Flask/PostgreSQL. `app_v3.py` remains the large central application; newer workflows are isolated in modules. |
| Mac agent | `charlie_local_agent.py`, managed by launchd as `com.charlie.local-agent`. It scans iCloud, handles local sources/audio/catalysts, syncs manifests, and participates in managed collection/recovery. |
| Primary storage | Production PostgreSQL plus user originals under iCloud `STOCKS`, `CATALYSTS`, and `SUMMARIES`. |
| Research collection | Managed AlphaSense browser workflow requires the Mac, signed-in Chrome, and scheduled worker availability. Public SEC/FDA/ClinicalTrials supplements are bounded and provenance-checked. |
| Production deployment | Push to `main` triggers Render backend deployment. Cloudflare frontend deployment is explicit through Wrangler. |

## Latest production changes

### T107 — broader catalyst discovery and safer duplicate handling

`catalyst_watch.py` now detects eleven bounded event families instead of four: clinical,
regulatory, financial results/guidance, M&A and strategic transactions, capital allocation,
leadership, legal, commercial, operational, workforce, and activist/ownership developments.
Routine earnings-date notices, conference appearances, ordinary dividends, previews and
rumours remain excluded. Lower-confidence commercial, workforce and ownership matches are
visible but wait for review rather than launching collection automatically.

Clinical identity now understands combined phases such as `2b/3`, recognizes a broader but
bounded set of asset suffixes, and can anchor on trial names wherever they appear. This
specifically prevents the two differently worded MRK BRUNELLO headlines observed in
production from launching two managed research requests. Non-clinical headlines also get
a conservative same-company, same-day, same-category similarity check. A match pauses the
second request for review; it never silently merges evidence or declares the events equal.

Validation: 762 backend unit tests and 56 frontend tests passed; touched Python compiled;
the production frontend build completed. The backend suite still prints the already-known
mock Summary Lab advisory-lock thread exceptions after those tests release their fake DB;
the suite passed. No paid research was launched for validation.

### Post-T106 — Summary Lab quality, cost accounting, and transcription evaluation

Commits: `7d321c4` through `d7ddb20`. These are in GitHub `main`; the backend is live at
`d7ddb20`. The frontend remains T106 because the only frontend-bearing commit in this
series (`e76115d`) shipped with the T106 markers and every later change is backend,
prompt, benchmark, or test code.

**Summary Lab model choice and cost visibility.** The composer can select from the
host-approved Anthropic models, while the existing default remains unchanged. Every Lab
model call now records provider, model and token use. Usage is recorded as soon as a
response returns, including responses that later fail validation or exceed the step's
budget; a failed paid call is therefore no longer missing from the ledger. Source markers
follow the reader's display preference when emailed: hidden markers are removed, shown
markers become restrained superscripts rather than raw `[P1]` text.

**Opus 5 token budgets.** Adaptive thinking counts against `max_tokens`. Two real Opus 5
runs spent their allowance thinking and returned no usable section. Each Lab step now has
a named budget with sufficient headroom, and an incomplete response reports the specific
limit. This changes the ceiling, not the default model, and does not itself make a run use
the full allowance.

**Analyst voice and quotation discipline.** Lab v4 treats a takeaway as a conclusion,
not an excerpt. Direct quotation is reserved for wording that is itself evidence: a
commitment, refusal, hedge, non-answer, qualified figure, or characterization that loses
meaning when paraphrased. Transcript repairs that context can settle are made silently
and logged in the review record instead of repeatedly narrated in the note. Genuine
ambiguity appears once where it changes the conclusion. This restraint does not suppress
analysis: a material inference is stated clearly and labeled as an inference.

**Transcription is measured before it is changed.** `scripts/compare_transcription.py`
is a read-only-by-default harness that compares role separation and fidelity, including
figures, date-like digit runs, disfluencies, and named-entity retention. The tested
`gpt-4o-transcribe-diarize` candidate lost material entities and meaning on the MCK excerpt
and rejects audio longer than roughly 23 minutes in one request. The temporary OpenAI
fallback was therefore removed. Production remains on the Gemini chain:
`gemini-2.5-flash`, then `gemini-2.5-flash-lite`. Audio usage is now priced and recorded.
Do not treat the comparison harness as a production transcription path.

**Proof boundary.** Prompt behavior is covered by focused tests but Lab v4 has not yet
been accepted on a paid real-source experiment. Do not launch one merely for engineering
QA. The next user-driven experiment should check quotation density, transcript-repair
noise, analyst judgment, entity preservation, section completeness, and the cost ledger.

Validation on September 22: 60 focused backend tests passed, 56 frontend tests passed,
and `app_v3.py`, `summary_lab.py`, and `scripts/compare_transcription.py` compiled. Two
mocked Summary Lab worker threads printed advisory-lock `fetchone()` exceptions after
their tests released mocked database state; the unittest process still passed. Treat that
test-fixture noise as worth cleaning up, not as proof of a production worker failure.

### T105–T106 — per-section actions, and one doctrine for both pipelines

Commits: `498271e` (section actions), `fac7b14` (shared doctrine). Worker
`2026-09-20T105`; backend `fac7b14`. 682 backend tests, 54 frontend tests.

**T105 — every Summary Lab section now carries the Summaries tab's four actions:**
Copy, Email, Email w/ options, Save to iCloud. Save reuses the existing Word exporter
the way `summary_comparison` does — `summary_lab.export_row` projects an experiment's
sections into the schema `_generate_summary_docx_bytes` already reads, rather than
teaching the exporter a second shape. Model text is escaped, not interpreted; the
transcript is never attached to a section export; an unfinished section returns 409
instead of an empty document. `create_blueprint` now takes the renderer and filename
helper, and answers 503 rather than failing if a server lacks them.

**T106 — the two pipelines were answering to two copies of the doctrine.** It lived in
`app_v3.py`, which `summary_lab.py` cannot import without a cycle, so the Lab carried an
older copy. `research_doctrine.py` now holds it, imports nothing, and both sides read it;
the Original's `TRANSCRIPT_DATE_RULE` and `RESEARCH_DOCTRINE` are reproduced **byte for
byte**, so its prompts are unchanged. Four Lab defects close with it:

| Defect | Was | Now |
| --- | --- | --- |
| Credibility rating | Lab forbade "numerical credibility scores" — the restriction T94/T95 overruled | Same defined 1–5 scale as the Original; interior-state claims still refused |
| Date shorthand | T97's `101` = 10/1 rule never reached the Lab | Lab gets the substance via `LAB_DATE_RULE`, without the bucket framework it has no equivalent of |
| Speaker labels | "Speaker 1/2" printed 13 times in Follow-up Questions, while the reviewer note warned the labels swap mid-P2 | Raw labels forbidden; attribute by role, or say the role is unclear |
| Section headings | Executive Brief headed itself "Key Takeaways" | Each section must head itself with its own name |

`VERSION` moves to `source-reviewed-lab-v3` so a re-import builds a fresh experiment
rather than matching the v2 row under the automatic-experiment conflict key.

**Unproven.** The four prompt changes are verified by inspection and tests, not by a paid
run. The next real SUMMARIES file should be checked for: a `Rating:` on the 1–5 scale in
the Lab assessment, no bare `Speaker 1/2`, no digit run read as a count, and a Brief that
heads itself as a Brief.

### T102–T104 — Summary Lab readability: mobile density, citations, reading layout, archiving

Commits: `4cbd5ce` (scroll button + mobile density), `6258a71` (source markers),
`39bd3dc` (reading layout, archiving, design tokens). Production: worker
`2026-09-20T104`, backend `39bd3dc`. 661 backend tests, 54 frontend tests.

**T102 — the scroll-to-top button never appeared in Summary Lab.** A document-level
listener already existed, but it only acted when the scrolling element's `className`
contained the Tailwind string `overflow-y-auto`. Every workspace styled with its own CSS
was silently excluded. The listener no longer gates on a styling class, so the button
follows any element that scrolls, including views added later. Same release: Summary Lab
on a 375px screen nested page + panel + reader padding and left ~263px for text (70% of
the screen); narrow-screen rules bring that to 339px (90%). The 600px block must sit
**after** the 900px block — equal specificity means the later one wins.

**T103 — 327 bracketed `[P1]` citations made the MCK note hard to read.** Each run now
renders as one deduplicated superscript, hidden by default, with a toggle that remembers
the choice. The transform runs on the rendered HTML outside tags, so it applies to every
saved experiment with no regeneration, and the stored text is untouched — Copy, Download
and Email still carry the literal markers.

**T104 — the reading layout wasted the window.** Opening an experiment kept a 330px rail
of source and list controls beside the document. It collapses into one command bar above
the reader; the reader is centred at its 94ch measure (766px with equal 227px margins at
1440px) instead of pinned left with 390px dead to one side.

**T104 — experiments can be archived, not deleted.** `archived_at` plus
`POST /api/summary-lab/archive` with `{ids, restore}`. Nothing is destroyed: the saved
Summary, its transcript and the iCloud original are untouched, and Restore returns the
row. Two guards matter — a **live worker refuses to be archived** (409, checked through
the same advisory lock `stop` uses) so paid work cannot be hidden mid-run, and the
**recovery sweep skips archived rows** so an archived stale `running` row cannot quietly
resume and spend money. Verified on production against the two dead Korean rows:
6 active → 4, restore returned one (4 → 5), re-archived → 4.

**T104 — Summary Lab ignored the house design system.** It hardcoded fifteen colour
literals, including `#c9a857`, a hand-matched copy of Dusk's brass, so it only ever
looked right in one theme. Every colour now resolves through `var(--accent)`,
`var(--line)`, `var(--surface)` with a fallback, matching the rest of the app, and status
renders as a toned chip. The reader's printed-paper surface keeps local `--lab-paper*`
tokens rather than raw hex.

**Everything listed here as open was closed in T106, recorded above.**

### T98–T101 — the MCK run: fan-out proven, orphan recovery, concurrency, clearer status

Commits: `08a3e11`, `5d9bc63` (orphan sweep), `73a0d3a` (section concurrency),
`b8661c9` (status line).

A real folder-dropped file, `MCK Mgmt Meeting @ DB - 091626.m4a`, finally proved the
recent work end to end — after two failed attempts that were **caused by deploying while
it was running**.

**Everything verified on the real note:**

| Check | Result |
| --- | --- |
| T96 — SUMMARIES fan-out | Lab experiment `e6edd9bc`, `automatic: true` |
| T97 — tiers agree on source type | Both `MGMT 1:1` |
| T97 — corrections log | 357 chars of real corrections; 0 identity entries, 0 "rendered correctly" (was 18 glossary entries) |
| T97 — date shorthand | Note carries `7/1`; no bare "71 states" |
| T95 — credibility | `Rating: [1-5]`, no score out of ten |
| T82/T85 — Lab recovery | `recovery_attempts: 1` — a deploy killed it mid-run and the sweep resumed it from checkpoints to completion |

**Operational hazard, learned the hard way.** Every push to `main` redeploys Render and
kills in-flight audio transcription. Two MCK attempts died this way, one of them to a
*documentation-only* commit. `AGENTS.md` says doc commits need no deployment; it does not
say pushing one causes a deployment regardless. **Do not push to `main` while audio is
processing.** Summary Lab jobs recover; transcription jobs restart from zero.

**T98/T99 — orphan recovery.** `_cleanup_orphaned_transcription_jobs` already existed but
ran only at import, so a job orphaned by one deploy waited for the *next* deploy to be
released; MCK sat stranded for half an hour. It now also runs every five minutes, keys
staleness on `updated_at` rather than `created_at` (a long transcription legitimately has
an old `created_at`), and both transcription and summarisation mirror their progress so a
live job stays outside the window. T98 briefly added a duplicate function; T99 removed it.

**T100 — section concurrency.** Measured: transcription ~4 minutes, five sections ~16,
run serially. The sections do not read each other. Key Takeaways, Questions, Assessment
and Meeting Summary now run concurrently; the Brief still follows Key Takeaways because
since T97 it reuses that tier's source-type classification. Results are keyed by section
name, never positional, so completion order cannot decide what is stored where — a test
makes them finish in reverse order and asserts it. Fatal/non-fatal semantics, tiers and
token budgets are unchanged. **Unproven:** no file has run under T100; expected ~16
minutes down to 8–10.

**T101 — status line.** It read "3 / 3 source parts reviewed · running" while four of five
sections were unverified, with Email all disabled and unexplained. It now reports sections
verified and names the phase, including the final cross-section review that used to look
idle because every other counter had topped out.

### T97 — spoken rate-cycle dates, the corrections log, and tier agreement

Commit: `0cb4230` — `Read spoken rate-cycle dates, and stop the corrections log becoming a glossary`

Three defects found in the real CNC management-meeting note. None were caused by T94/T95;
all pre-dated them in the original Summary prompt.

**Spoken dates survived as nonsense quantities.** Managed-care speakers say effective dates
as bare digit runs, and transcription renders them without separators. The note reproduced
**"71 states"** — there are 50 — and left **"101 implementation"** uncorrected although the
questioner said "October 1st" aloud in the same exchange. The user supplied the domain
reading: `11` = 1/1, `71` = 7/1, `101` = 10/1, and `"71 states"` means states whose rate
cycle begins 7/1. `TRANSCRIPT_DATE_RULE` adds this as a named correction class in STEP 1,
including the year-suffixed form (`"11 27"` = 1/1/27) and a prohibition on rendering a bare
digit run as a count when the sentence is about timing.

**The Transcript Corrections Log became a glossary.** It listed eighteen entries shaped like
`"ICHRA" → ICHRA … Transcript rendered correctly`, including NDR and RADV, which do not
appear in the transcript at all. The section now logs only terms whose wording actually
changed, and names the identity entry and the acronym definition as the defects they are.

**The two tiers disagreed on source type.** The Brief classified the meeting
INVESTOR/PUBLIC while Key Takeaways classified it MGMT 1:1. Each tier classified
independently, and nothing reconciled them. The Brief is generated after Key Takeaways, so
`_classified_source_type()` now extracts the classification already made and the Brief is
told to reuse it verbatim.

**What did work:** the T95 doctrine produced exactly the intended assessment — *"Rating: 4 —
mostly supported, with minor unsupported assertions"* on the defined scale, with drivers
named and unsupported claims separated out, while staying candid ("notable squishiness",
"investors hoping for an early read got shut down"). No score out of ten, no interior-state
inference.

Validation: 629 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no summary generated under T97. The checks are
whether "71 states" becomes 7/1 states, whether the corrections log returns to real
corrections or "No corrections required", and whether both tiers agree on source type.

### T96 — the SUMMARIES fan-out was silently dead; job state no longer discarded

Commit: `2f80943` — `Stop transcription discarding the fields the caller set on a job`

A real folder-dropped file (`CNC CEO CFO Small Group - 091626.m4a`) transcribed and
summarised correctly but produced **no Summary Lab experiment**. The job reported
`summaryLabState: "not_requested"`.

T82 stored the upload's `origin` on the in-memory job dict. When transcription finishes,
`_run_transcription` **replaced that dict wholesale** with five keys of its own, so `origin`
was gone before the fan-out check ran — and the check correctly concluded that no fan-out
had been requested. The same replacement discarded `autoProcess`, which
`_mirror_transcription_state` reads when persisting the job.

The Mac agent was never at fault: it detected the file, sent `origin=summaries-folder`, and
its process was started after the T82 edit. The loss was entirely server-side.

- `origin` now travels as an argument through `_run_auto_process_audio_path` into
  `_run_auto_process_audio`, where no dict lifecycle can lose it.
- `_run_transcription` merges into the job dict instead of replacing it, on the success path
  and on all three error paths.
- A test walks the AST of `_run_transcription` and fails on any wholesale assignment to
  `_transcription_jobs[job_id]`. It caught a fourth site during this change.

**Lesson:** state set by a caller and read later by a different worker must travel as an
argument or in the database, not on a shared mutable dict that an intermediate stage owns.
The failure was silent because the fan-out check behaved correctly on the input it was given.

**Blast radius:** T82 shipped at 11:02 today and CNC was the first folder-dropped audio
after it, so one file is affected. Generating its Lab experiment now requires starting one
from the saved Summary in Summary Lab, which is a paid run and the user's call.

Validation: 617 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no folder-dropped audio has run under T96.

### T95 — the assessment stays candid; the thesis contradiction is fixed

Commit: `d64e4db` — `Keep the assessment candid, and fix the thesis contradiction it caused`

T94 over-corrected. Three of its rules removed capability the original prompt had
deliberately, and one of them broke a live feature. The user caught all three.

- **"Don't hedge" had a purpose.** It existed to stop mushy notes, not to license
  invention. The doctrine now requires committing to a view — if an answer was weak, say so
  — and forbids only inventing a fact to support it.
- **BS detection was deliberate.** Calling out non-answers, evasions, redirections,
  rehearsed talking points and self-contradiction is the job of an assessment. Restored and
  required, with the wording quoted as evidence. The line is drawn at **interior state**:
  feelings, anxiety, morale, private belief and motive stay out, because a joke or a hedge
  is not evidence of them. That still prevents the observed failure, which was reading "I
  lose lots of sleep for lots of reasons" as concern about the CVS renewal.
- **Rating credibility is legitimate.** An unanchored "8.5 out of 10" is not: it implies
  precision that does not exist and cannot be compared between notes. The rating is restored
  on a defined 1–5 scale with stated anchors, and must name what drives it.
- **The baseline rule was wrong, not merely strict.** When a ticker has a registered thesis,
  `thesis_addendum` injects it and asks for per-pillar CONFIRMED / WEAKENED / NO MENTION
  verdicts — and T94 placed "no prior thesis is supplied, do not claim thesis confirmation"
  immediately before that block in the same prompt. The rule is now conditional: compare
  against a baseline where one is supplied, invent one where it is not.

**Lesson:** a rule written to prevent an observed failure removed three capabilities that
were not causing it. Before constraining a prompt, check what each instruction was for — the
"UNFILTERED" framing, the disingenuousness question and the credibility rating were all
deliberate, and only the unanchored scale and the interior-state inference were defects.

**Where the user wants this to go.** The stated goal is that over time the model judges what
is new, what contradicts a previous statement or thesis, and what differs from consensus.
The thesis half already exists via `thesis_addendum`. The missing half is prior-meeting
context: `company_history_recall.py`, `company_memory.py` and `thesis_history.py` hold the
material, but no summary prompt is given the last N notes for the ticker. Supplying them
would let the assessment say "this contradicts June" without inventing the comparison. That
is the natural next step for this workstream.

Validation: 614 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no summary generated under the revised doctrine.

### T94 — convergence decided: improve the original, retire Improved

Commit: `26c8c37` — `Port the Improved pipeline's evidence discipline into the original Summary`

**This settles priority 3.** Direct comparison of the two prompts showed the original already
carries the stronger fidelity apparatus, and Improved carries almost none of it:

| Mechanism | Original | Improved |
| --- | --- | --- |
| Source-type auto-classification with distinct lenses | yes | no |
| Transcript-correction confidence taxonomy + corrections log | yes | no |
| No quantitative tightening ("teens" ≠ "low teens") | yes | no |
| Segment attribution discipline | yes | no |
| Clarifying follow-ups kept as separate Q&A entries | yes | no |
| Quarterly vs all-time scope discipline | yes | no |
| Quote-length rules, never paraphrase inside quotes | yes | no |
| Nine-point silent self-check | yes | no |
| Named-entity hallucination guard pass | yes | no |
| Known Unknowns with deflection phrase and next check | yes | no |

What Improved had was restraint in the **assessment** step. The original's assessment prompt
asked for a "CANDID, UNFILTERED" take, told the model not to hedge, asked whether anyone
"seemed disingenuous", and requested "Rate overall credibility of key claims" — which is
where the invented "8.5/10" and the psychology read of a deflecting joke came from. The
original's *summary* prompt was never the problem.

So the cheaper path to the user's goal was the reverse of the previous eight releases: port
Improved's four epistemic rules into the original rather than rebuild a dozen fidelity
mechanisms inside Improved.

- `RESEARCH_DOCTRINE`: separate statement from judgment and name the support and the limit;
  never infer psychology or motive; never assign a numerical credibility score; claim no
  novelty, consensus difference or thesis confirmation without a supplied baseline. It
  explicitly outranks the instruction to be candid.
- `ASSESSMENT_INSTRUCTION` replaces **five** near-identical assessment prompts that had
  drifted apart across the audio, document, podcast, meeting and manual paths. It stays
  candid and specific but evidences evasion by quoting wording rather than inferring intent.
- Both summary prompts carry the doctrine; their fidelity apparatus is untouched.

**The earlier instruction not to rewrite the original Summary's prompts no longer applies.**
It existed to protect the comparison baseline; the user has now decided the comparison's
outcome and directed this change.

**Improved is deliberately still running.** Retire it only after the user confirms the
improved original on a real source. Retirement means removing the `summary_comparison`
workflow, its UI and its automatic fan-out — not deleting saved notes.

Validation: 613 backend tests including a new doctrine suite, 46 frontend tests, production
build, Render revision and Cloudflare marker verified. **Unproven:** no summary has been
generated under the new doctrine.

### T93 — every Improved section is written from the source

Commit: `e4907ae` — `Write every Improved section from the source, not a summary of it`

T92 fixed the Q&A log by reading it from the source. The user pointed out the obvious
consequence: **every** section should be, because the goal is for Improved to replace the
original Summary. They were right, and the architecture was worse than it looked.

Improved split the source into 24,000-character parts, summarised each into a
topic-organised record, and wrote every section from those records. The source never reached
the sections at all. The original Summary, by contrast, passes the full transcript to each
section prompt — so Improved was structurally **less faithful than the thing it is meant to
replace**.

The map-reduce exists for sources too large for one prompt. Measured against the last 60
real sources, that case does not occur:

| | |
| --- | --- |
| CAH transcript | 37,453 characters |
| Median source | 39,618 |
| Largest of the last 60 | 213,720 (~53k tokens) |
| Sources above 300,000 characters | 0 |
| Model context | ~200k tokens (~800k characters) |

- Below `DIRECT_SOURCE_LIMIT` (400,000 characters) every section is written from the
  complete original source, marked authoritative, with the evidence records alongside as a
  navigation aid that is explicitly not a substitute.
- Above it the pipeline falls back to records and says so in both the prompt and the
  workspace, so a degraded basis is visible rather than silent.
- Record consolidation is skipped when the source is read directly; it only ever existed to
  shrink records for synthesis.
- `state.synthesisBasis` records which path ran and the method panel reports it.

Cost: each section prompt now carries the full source, roughly 9k extra input tokens per
section on a median transcript.

**Lesson:** the map-reduce was carried over from a context budget that no longer binds, and
it silently degraded every section. Before summarising a source for a model, check whether
the source fits — on this workload it always does.

Validation: 598 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no note generated under `source-first-v7`.

### T92 — the Q&A log is read from the source, not from a summary of it

Commit: `c0da9ef` — `Build the Q&A log from the source, not from a summary of it`

The user compared the Improved Q&A log against the Original's and supplied the full
transcript. The Original captures essentially every exchange; Improved captured 10, labelled
most "(Implied)", and stated the source was management commentary rather than a
question-and-answer transcript. Both claims are false — the questions are verbatim in the
transcript from Speakers 2 and 3.

**This was a data-flow bug, not a prompt problem.** Every section including the Q&A log was
drafted from the evidence records, which are organised by topic and discard exchanges.
Measured on the saved note: the source has roughly 17 question turns, while the 15,265
characters of records retained **4 question marks and zero `Q:` markers**. The section was
asked to reproduce exchanges it was never shown, and — correctly forbidden from inventing
them — reported them as implied.

- The Q&A log is now generated per raw source part, checkpointed like the evidence records
  and joined in order. Parts containing no exchange are omitted.
- The instruction forbids describing a present question as implied or claiming the source
  lacks a Q&A structure.
- `qa_findings()` flags both a collapsed log and one that disclaims questions in a
  question-rich source.

Cost: one extra model call per source part, only for the Q&A section.

**Lesson:** a section whose job is fidelity to the source cannot be built from a summary of
the source. `brief`, `takeaways`, `assessment` and `questions` are syntheses and are
correctly record-derived; `qa` is a record and is not. Any future section should be
classified that way before it is wired up.

Validation: 593 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no note generated under `qa-from-source-v6`. The
check is whether the Q&A log reaches roughly the Original's exchange count with no "implied"
labels.

### T91 — assessment stops transcribing, and dropped figures are caught

Commit: `7cec6ce` — `Stop the assessment restating the record, and catch dropped figures`

`readable-v4` was generated on the real CAH note. Measured against every predecessor:

| | conservative-v1 | readable-v2 | readable-v3 | readable-v4 |
| --- | --- | --- | --- | --- |
| Quotes per 1k chars | 4.5 | 3.7 | 0.6 | **1.3** |
| Total characters | 15,762 | 21,084 | 13,778 | **31,689** |
| Q&A exchanges | 0 | 9 | 3 | **14** |

v4 is the best note produced so far and resolves the original complaint: 71% fewer quotes
per 1k than the version first flagged, with 50% more content than v2 and the richest Q&A
log of any version. The checks behaved as designed — takeaways (4.4/1k, worst block 6),
assessment (3.5/1k) and questions (2.0/1k) were each repaired once and each *grew*, so the
shrink guard never had to discard a repair.

Two defects remained, both fixed in `readable-v5`:

- The assessment reached 11,233 characters against 6,767 of takeaways and opened with
  "MANAGEMENT STATEMENTS", restating the record instead of assessing it — a consequence of
  T90 removing the cross-section rule with nothing in its place. The instruction now says so
  explicitly and `assessment_findings()` flags a section disproportionate to the takeaways.
  The shrink guard is bypassed for that one repair, since a shorter assessment is its point.
- The 12–14% long-term algorithm was dropped from every section *and* from the management
  record, while 3.5% was asserted as the long-term target. It was caught only by grepping
  for that string. `figure_coverage()` now extracts distinctive figures from the evidence
  records, normalising spacing and dash style, and reports any the note does not carry.

`figure_coverage` **reports, it does not rewrite.** Another automatic edit risks the damage
v3 caused, and a missing figure is not always an error.

Validation: 587 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified.

**`readable-v5` was then generated on the real CAH note and both defects are fixed:**

| | v1 | v2 | v3 | v4 | v5 |
| --- | --- | --- | --- | --- | --- |
| Quotes per 1k chars | 4.5 | 3.7 | 0.6 | 1.3 | 1.7 |
| Total characters | 15,762 | 21,084 | 13,778 | 31,689 | 26,459 |
| Q&A exchanges | 0 | 9 | 3 | 14 | 10 |
| Assessment ÷ takeaways | 0.67 | 0.53 | 0.39 | 1.66 | 0.68 |
| 12–14% figure present | yes | yes | yes | **no** | yes |

The assessment fell from 11,233 to 5,257 characters and its ratio from 1.66 to 0.68, in line
with v1 and v2. `assessment_findings()` did not fire — the instruction alone was sufficient,
and the check stands as an unused backstop. `figure_coverage` reported 12 distinct figures in
the records with none missing from the note. Against the note originally complained about:
62% fewer quotes per 1k, 68% more content, and a Q&A log where there was none.

Open judgment call for the user, not a defect: v5's Q&A log labels most questions "(Implied)"
and closes with a note that the source is management commentary with moderator prompts rather
than a formal Q&A transcript. That is the anti-fabrication rule working, and it is why v5 has
10 exchanges where v4 asserted 14. Whether the hedging helps or reads as noise is a product
preference.

**Stop single-note tuning here.** Five regenerations against one CAH transcript produced two
regressions caused by generalising from a sample of one (T89 and T90 both document a case).
Further prompt work should wait for priority 6's frozen real-source packs so a change can be
scored across several meetings. The machinery to do that now exists: `quote_findings`,
`qa_findings`, `assessment_findings` and `figure_coverage` are all deterministic and can be
run over any note without a model call.

### T90 — the quota must not be met by deleting evidence

Commit: `055df48` — `Stop the quota from being met by deleting evidence`

`readable-v3` was generated on the real CAH note and measured against its predecessors:

| | conservative-v1 | readable-v2 | readable-v3 |
| --- | --- | --- | --- |
| Quotes per 1k chars | 4.5 | 3.7 | **0.6** |
| Total characters | 15,762 | 21,084 | **13,778** |
| Q&A exchanges | — | 9 | **3** |

The quoting complaint was solved. The note also lost **35% of its content**, and picked up
four defects worse than the one it fixed:

- The Q&A log fell to three exchanges and asserted that no others were identifiable, from a
  source containing dozens of questions. One answer was openly reconstructed from elsewhere.
- Sections began deferring to each other. The executive brief — the first thing read —
  contained "As noted in the takeaways", "per the takeaways", "in the assessment section".
- The FY27 metric regressed from correctly flagged as unstated (v2) to asserted as a 3.5%
  long-term algorithm alongside a contradictory 12–14% figure in the same section, which is
  the Original's error.

**The repair pass was not the cause** — it fired on one section only. Both prompt changes
were. T90 (`readable-v4`) keeps the code enforcement and reverts them:

- The numeric quota is out of the prompt. Stating it made the model aim for zero and reach
  it by dropping evidence. `quote_findings()` still enforces the ceiling, and the rules now
  state that converting a quote never means dropping the fact, number, qualification or
  attribution it carried.
- Sections are no longer shown each other. Each must stand on its own.
- A repair returning a materially shorter section is discarded and the original kept.
- `qa_findings()` flags a Q&A log that collapsed against a question-rich source, and the Q&A
  instruction forbids reconstructing an answer.
- The method panel reports what each check found and which draft was kept.

**Lesson worth carrying:** on this pipeline, a prompt instruction that competes with the
fidelity rules is either ignored or over-complied with by deletion. Both failure modes were
observed on the same note. Constraints of this kind belong in code, with a guard against the
model satisfying them destructively.

Validation: 580 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. **Unproven:** no note has been generated under `readable-v4`.
The open question is whether it restores v2's content at v3's quote discipline.

### T89 — the quoting quota is enforced, not requested

Commit: `1b7ad81` — `Enforce the quoting quota instead of asking for it`

The user regenerated the CAH note under `readable-v2` and still saw heavy quotation.
Measured on the two saved notes rather than judged by impression:

| Section | conservative-v1 | readable-v2 |
| --- | --- | --- |
| Key takeaways | 6.3 quotes/1k chars | 4.6 |
| Executive brief | 4.2 | 3.6 |
| Follow-up questions | 3.4 | 1.2 |
| Investment assessment | 3.2 | 1.9 |
| Q&A log | — | 5.5 (new) |
| Overall | 4.5 | 3.7 |

`readable-v2` did work: density fell in every comparable section and broken numbered lists
went 21 to 0. But the note is 34% longer, so absolute quotes rose 71 to 79, and the rule
"at most one short quoted phrase per point" was ignored in **8 of 9 tagged takeaways**, one
carrying nine quoted fragments. The phrasing was ambiguous and a style preference loses to
the fidelity rules above it in the same prompt.

T89 (`readable-v3`) stops asking:

- Quoting is governed per section. The Q&A log and the per-part management records are
  exempt because verbatim is their job. The four analysis sections carry a stated quota.
- `quote_findings()` checks each draft in code and names the breach; a section over quota is
  redrafted once, instructed to keep every fact, number, qualification and attribution while
  converting the least informative quotes to reported speech.
- Sections already written are passed to later ones so caveats are referred to, not restated.
- The method panel reports which sections were redrafted.

Calibrated against the real note: brief (9 quotes, 3.55/1k) and takeaways (4.6/1k, worst
block 9) breach and would be repaired; assessment (1.88/1k) and questions (1.16/1k) already
comply, so no model call is wasted.

Validation: 575 backend tests including the enforcement and repair-pass tests, 46 frontend
tests, production build, Render revision and Cloudflare marker verified. **Unproven:** no
note has yet been generated under `readable-v3`. Whether the repair pass produces readable
prose without losing content is the next real-source check, and it is a paid run.

### T88 — regenerate an older Improved note in the current format

Commit: `15fa32b` — `Let an older Improved note be regenerated in the current format`

T87 added the Q&A log but left no way to obtain it. Existing notes carry their own prompt
version, and the workspace could only report **Not in this version**. The listing never
returned the pipeline version, the Generate button appeared only when no note existed at
all, and `start()` always passed `resumeId`, which resumes the old row under its own
version.

- The comparisons listing now returns `currentVersion`.
- When every saved note predates it, the workspace offers to generate the current version,
  naming both versions, stating that it re-reads the saved transcript and uses research API
  credits, and that existing notes are kept.
- That run omits `resumeId`, so it creates a note in the current version. Retry still
  resumes in place.
- The saved-version picker marks older notes as "older format".

Nothing is regenerated automatically: upgrading an existing summary is a paid run and stays
user-initiated. New summaries get the current format on their own.

Validation: 567 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified, and the panel confirmed in the browser against the real
`conservative-v1` CAH note. The generate button was not pressed.

### T87 — Improved notes: readable prose, topic tags, and a Q&A log

Commit: `5cbd098` — `Give Improved notes the readability the Original had`

Prompted by a real side-by-side review of a CAH management meeting. Findings from that
comparison, recorded because they bear on priority 3:

- The **Original** asserted "FY27 guidance of 3.5–4% EPS growth exceeds the long-term 3.5%
  target". The transcript never says EPS and never states a 3.5% long-term target; it
  contains Speaker 3's "3.5 to four is above 3.5" and, separately, Speaker 2's "12 to 14%
  long-term growth algorithm". The Original supplied the unit and dropped the conflicting
  12–14% figure. The Improved version flagged the ambiguity and preserved both figures.
- The **Original** also reported "High credibility (8.5/10)" — an invented numerical score
  the Improved rules already forbid — and inferred psychology, reading the CEO's joke "I
  lose lots of sleep for lots of reasons" as evidence of concern about the CVS renewal.
- The **Improved** version was materially less readable: quote-stuffed (eight quoted
  fragments in one CVS paragraph), repeated the same guidance caveat six times across
  sections, rendered broken numbered lists, and carried a far thinner Q&A log.

Neither version is objectively better overall, and no benchmark exists that could settle
it. T87 changes only the Improved pipeline, leaving the original Summary prompts untouched:

- `summary_comparison.py` RULES now make reported speech the default and reserve quotation
  marks for wording that is itself the evidence, at most one short phrase per point. It
  bans numbered lists, and requires each fact to be stated once rather than restated in
  every later section.
- Key takeaways open with a scannable bracketed topic tag, adopted from the Original.
- A new **Q&A log** section reproduces the substantive exchanges in order, preserving
  numbers, hedges, refusals and non-answers, and must say so rather than invent an exchange
  when the source has no real Q&A structure.
- `VERSION` moves to `readable-v2`. `summary_comparisons` is unique on
  (summary_id, source_hash, version), so existing notes are preserved and readable rather
  than mixed with output from a different prompt. Older complete notes now label a missing
  section **Not in this version** instead of a permanent "Waiting…".

Validation: 565 backend tests including a new cross-file contract test that every generated
section is rendered by the workspace, 46 frontend tests, production build, and the section
list verified in the browser against the real `conservative-v1` CAH note.

**No output-quality claim is being made.** The prompt now forbids the specific defects
observed; whether the result reads better on real sources is priority 2/3 work and needs
the user's own comparison on a new run.

### T86 — Summary Lab gives the page to whichever job is in front

Commit: `942407d` — `Let Summary Lab size itself to the task at hand`

The layout was fixed at `330px / 1fr` whatever the state. Measured at 1600px with no
experiment open: the composer had 330px (26%) while the empty "Independent source review"
panel held 962px (74%) and 1460px of height for 120px of placeholder text.

- With no experiment open the page is a single centred 900px column. The composer spans
  it: five intake buttons in one row, source dropdown 850px instead of 280px, and
  Experiment name beside Optional emphasis rather than stacked. The placeholder panel is
  gone; its orientation copy already exists in the page intro.
- While reading an experiment the two-column split returns, because a long note needs the
  width, and the composer collapses to a **New experiment** button. It went from 654px to
  192px, giving the Experiments list 786px.
- Mobile is unchanged in shape: single column, intake two across, fields stacked.

Verified in the browser against the built bundle by measuring the live DOM in all four
states — idle at 1600px and 375px, reading at both — including that the composer expands
on demand and nothing overflows horizontally.

### T84 / T85 — stop control and duplicate Summary Lab runs

Commits: `4065477` — `Let Summary Lab experiments be stopped, and stop duplicating them`,
`3d42157` — `Stop experiments a restart left running`

One real YouTube submission produced **three identical `korean_bilingual` experiments**
over the same 20k-character transcript. Investigated from the live rows, not reproduced.

- Root cause: the pending transcription job lives in `localStorage`, so every tab and
  every reload resumes monitoring it, and each resumed monitor called `startLab` on
  completion. The manual enqueue path had no duplicate guard at all, unlike the automatic
  one, so each POST bought another Opus multi-pass run.
- A Lab run triggered by a completed transcription now carries that job id, unique
  server-side, so every monitor converges on the first experiment. A deliberate Generate
  click sends no job id and still creates its own experiment, which is what
  `docs/summary-lab.md` specifies.
- New `POST /api/summary-lab/<id>/stop` and a **Stop this experiment** button. Cancellation
  is cooperative at the existing checkpoints: `save()` writes `cancelled` instead of
  `running` when a stop is pending and raises, so a stop cannot be papered over between
  stages, and a queued experiment never reaches the model.
- T85: a row left at `running` by a backend restart has no worker to reach a checkpoint,
  so the stop route tries the experiment's advisory lock. A free lock means nothing is
  working on it and the row is cancelled immediately.

Validation: 555 backend tests, 46 frontend tests, production build, Render revision and
Cloudflare marker verified. The duplicate rows were read from production through the
signed-in app; no experiment was started, cancelled or approved during the investigation.

**Known related design issue, not yet changed.** Summary Lab's YouTube and audio intakes
post to the shared `/api/youtube-summarize` and `/api/auto-process-audio` endpoints, which
always write a full original Summary (five sections plus Korean) to `meeting_summaries`
before Summary Lab starts its own experiment. One Lab submission therefore pays for both
an original Summary and a Lab run, and the original lands in the Summary tab rather than
where the user started. That is the current design — Lab compares against that baseline —
but it surprises the user and deserves an explicit decision.

### T83 — Catalyst synthesis becomes a named destination

Commit: `8ea85ab` — `Promote Catalyst notes to a top-level destination`

Catalyst synthesis was reachable only through Automations → Research agents → an
unrouted `Catalysts` pill, under a page heading that read `TradingAgents`. Three names
for one path, and the workspace had no URL at all.

- **Create → Catalyst notes** is now a top-level destination at `#view=catalysts`. It is
  linkable, bookmarkable and restored by the back button; `agentView` was local state, so
  none of that was previously possible.
- Arriving there shows a `Catalyst notes` heading and no TradingAgents framing or sub-tab
  row. The `Catalysts` pill was removed from the Research agents tab, leaving one path in.
- The agents sub-views (`research`, `new`, `batch`, `dashboard`, `history`) are now scoped
  to `activeTab === 'agents'`, so the last-opened agents panel cannot bleed into the
  routed Catalyst destination.
- Catalyst history is fetched on arrival. It used to depend on the pill's click handler,
  which a URL, a sidebar click or an alert action never invoked.
- Saving a synthesis previously ended in a blocking `alert('Saved to Research tab')` that
  named a destination without going there. It now shows an in-page confirmation naming the
  saved document, with a button that opens Library → Research documents.
- Fixed a pre-existing mobile defect on that screen: at 375px the Auto-Pilot `Auto-fire`
  select overflowed the card and overlapped the paragraph. The row now stacks below `sm`.

Validation completed for T83: 46 frontend tests (two new: `catalysts` in `VIEWS`, its
Create-group membership, its label, and `readRoute`/`routeHash` round-trip with ticker),
549 backend tests, production build, and a browser pass against the built bundle at
desktop and 375px. Verified in the browser: the sidebar entry and breadcrumb, the routed
heading, back-button restore, the Research agents tab still intact with no Catalysts pill
and no sub-view leakage, catalyst history loading on arrival, and the Auto-Pilot row
measured as non-overflowing at both widths. No synthesis was run, no proposal approved,
no note saved and no email sent during verification.

### T82 — correct the automatic dual Summary fan-out

Commit: `1dfaf72` — `Scope SUMMARIES fan-out and harden Summary Lab recovery`

T81 fanned every `/api/auto-process-audio` job into Summary Lab. That endpoint serves
two callers, so three real defects shipped with it. T82 fixes all three.

- **Duplicate paid Summary Lab runs.** Summary Lab's own Audio intake posts to the same
  endpoint and then starts its own experiment from the saved Summary. Since T81 each
  such upload produced two experiments over the identical transcript — two complete
  Opus multi-pass source reviews. Audio uploads now carry an `origin` marker, and only
  `origin=summaries-folder` (sent by `charlie_local_agent.py`) fans out. Unidentified
  callers do not fan out, so every failure mode is a missing experiment rather than
  silent duplicate model spend.
- **Summary Lab now adopts an automatic experiment instead of starting a second one.**
  If a completed transcription already carries `summaryLabId`, the workspace opens that
  experiment and says so. This holds even if a stale frontend bundle or an un-restarted
  agent disagrees about the origin marker.
- **Restart durability regression.** T81 moved `_mirror_transcription_state` after the
  two follow-on queue calls. A backend restart in that window left the mirror at
  `summarizing`, so the folder watcher kept the file in the `SUMMARIES` root and the
  next tick re-uploaded and re-transcribed audio that was already saved. Completion is
  now mirrored before the queues run, and the Summary Lab id is mirrored after.
- **False "Summary Lab did not start" alerts.** The watcher polls every 10 seconds and
  decided Lab state from the same response that first reported completion, before the
  fan-out had returned an id. The status route now reports an explicit
  `summaryLabState` (`pending`, `started`, `failed`, `not_requested`, `unknown`), the
  watcher waits out a bounded 60-second `pending` window, and it stays silent about Lab
  for uploads that never requested a fan-out. The database fallback never reports
  `pending`, because after a restart only a recorded Lab id proves the fan-out started.
- **Summary Lab recovery thread pile-up.** Only two experiments run at once, so a
  recovery thread can block on the semaphore for a long time. The 30-second sweep
  queued another blocked thread for the same experiment on every pass. Recovery now
  tracks in-flight experiment ids and starts at most one thread each.

Validation completed for T82: 549 backend unittests, 45 frontend tests, the research
quality self-test, `py_compile` on `app_v3.py`, `summary_lab.py` and
`charlie_local_agent.py`, and the production build. Render `/health` reported revision
`1dfaf726498be5ddde89bac233d7ea712f2fcbe2`, the Cloudflare worker reported
`2026-09-20T82` and served `service-worker.js` at `20260920-82`, and
`com.charlie.local-agent` restarted to `state = running` with a clean startup log. The
two new structural regression guards were confirmed to fail against the T81 code and to
pass now. No paid audio run and no model-backed workflow was launched for validation, so
the deployed fan-out behavior itself is not yet proven against a real recording.

Not changed in T82: automatic folder fan-out still uses Summary Lab's English mode, and
historical files under `SUMMARIES/Processed` are still not mass-reprocessed.

### T81 — automatic dual Summary processing

Commit: `d8c608d` — `Fan out SUMMARIES audio into Summary Lab`

New audio placed at the root of the iCloud `SUMMARIES` folder now follows the established original Summary workflow and also launches an independent Summary Lab experiment from the complete saved transcript.

- The original Summary remains the primary saved output.
- The older Improved comparison continues unchanged.
- Summary Lab receives a separate durable experiment using its own current prompt and settings.
- Automatic Summary Lab rows display `Auto from SUMMARIES`.
- Automatic experiments are idempotent by saved Summary, source hash, prompt version, and output mode.
- Interrupted automatic Lab jobs resume after a backend restart with bounded recovery.
- Lab completion/failure produces a Charlie alert.
- One branch can fail without preventing the other saved output.
- The transcription job records the linked Summary Lab ID, and the Mac completion notification states whether Lab started.
- Automatic folder jobs currently use Summary Lab's English mode.
- Historical files already under `SUMMARIES/Processed` were intentionally not mass-reprocessed because that would create large unrequested model usage.
- At deployment time the `SUMMARIES` root contained no waiting audio, so the first real new file remains the live end-to-end proof.
- T82 corrected this release's fan-out scope, restart durability, watcher notification and recovery sweep. Read the T82 section above first.

Validation completed for T81: 43 frontend tests, 18 focused backend tests, Python compilation, production build, Render health revision, Cloudflare T81 asset, and launch-agent restart. No paid audio run was launched solely for validation.

### T80 — Summary Lab language and email polish

Commit: `14784c4` — `Add Korean Summary Lab modes and clean email styling`

- Summary Lab YouTube intake mirrors original Summary language choices: English, English plus Korean interpretation, and Korean-only output where appropriate.
- Summary and Summary Lab email rendering was consolidated into a clean professional format.
- Removed the teal/blue title banner and `Generated by TDL Equity Analyzer` footer.
- Email content uses black Calibri 11-point styling with bold section headings.
- Existing Summary content prompts were not intentionally changed by this formatting work.

### Recent Summary Lab usability work

- `516be90`: fixed accordion state so sections expand/collapse reliably.
- `360d0b4`: added document, audio, and YouTube intake; converted raw Markdown-looking output into sanitized, readable HTML; improved section controls and mobile readability.
- `212208f`: added professional document formatting, individual section sharing, and previewed email-all-sections behavior.
- `8ea4e4c`: introduced Summary Lab as an isolated parallel experiment using connected iCloud documents, checkpointed long-source review, and five familiar sections: Executive Brief, Key Takeaways, Meeting Summary, Follow-up Questions, and Overall Assessment.

Summary Lab deliberately remains separate from the original Summary workflow so the user can compare output quality before deciding whether to replace anything.

### T77 — catalyst synthesis upgrade

Commit: `7b67700` — `Ship catalyst investor notes, event synthesis and Q&A sharing in T77`

Catalyst synthesis now distinguishes source shapes:

- Transcript/fireside-chat workflow: concise brief, detailed note, Q&A when a real question-and-answer structure exists, and shareable output.
- The Q&A view can hide/show asker and answerer identities; email/save follows the visible choice.
- Multi-document event workflow: reviews the folder's primary and broker materials, then writes one investor-facing event note in the user's voice. It does not narrate which broker said what and does not add BUY/HOLD/SELL labels.
- The event note centers on: My takeaway, What happened, Why it matters, What remains unproven, and What I'm watching next.
- Generated review assets and local comparison tools exist, but some remain untracked local development material; preserve them unless the user decides to formalize or remove them.

### Other important recent capabilities

- Editable earnings recap decks and guided research starters (`73382a3`).
- Versioned investment-case signals, evidence half-life doctrine, falsification/variant/position-divergence views (`4a7805d`).
- Evidence proposal repair with source re-reading, retained checkpoints, concise proposals, and acceptance gates.
- Command Charlie favorites and guided multi-company meeting preparation.
- Managed AlphaSense refresh policies, source preferences by stock/subsector/analyst, explicit restricted-source handling, durable collection ledgers, and bounded recovery.
- UI release audit across primary routes, themes, and responsive widths, followed by readability and pool-capacity fixes. See `docs/ui-release-audit.md` for proof boundaries.

## Current workflows and boundaries

### Original Summary

The established Summary workflow is still active and remains the comparison baseline. It supports saved documents/audio/YouTube, Brief, Key Takeaways, Meeting Summary, Follow-up Questions, Assessment, transcript access, saving/export, and email. Earlier character clipping was removed. Its prompts were deliberately changed in T94: see that entry. The earlier instruction not to touch them has been superseded by the user's convergence decision.

### Summary Lab

Summary Lab performs a more rigorous, checkpointed full-source review and produces the same recognizable backbone in a separate workspace. It supports saved Summary sources, connected documents, audio, YouTube, output-language modes, collapsible HTML sections, copy/save/email controls, and automatic dual routing from the `SUMMARIES` folder.

Automatic fan-out is scoped to the folder watcher only. Audio uploaded inside Summary Lab
starts exactly one experiment, using the title, emphasis and language the user chose. If a
recording was already fanned out automatically, the workspace opens that experiment rather
than paying for a second review of the same transcript.

Open question: after several real comparisons, decide which Lab prompt/format improvements should migrate into original Summary. Preserve the original until the user explicitly makes that decision.

### Catalyst synthesis

The upgraded code is deployed. The user has reviewed MMM and ELV transcript examples plus a multi-document MRK event example. Further work should focus on output quality, document-shape classification, analyst voice, and repeatable live comparisons rather than merging it with the unrelated Summary workflow.

### Investment thesis lifecycle

Charlie can store cases, compare new evidence, propose source-backed revisions, repair unsupported proposals, discuss interpretations, show case signals/evolution, and retain guarded revision history. The investor remains the approval authority. Evidence checks reduce risk but do not independently certify management claims or complete factual accuracy.

### AlphaSense automation

The scheduled collection worker is active, but a saved policy or queued request is not proof of completed downloads. The September 26 inspection found queued work and several requests paused for browser control; see the T111 operational notes. Full unattended coverage across all tickers has not been proven end to end.

Collection requires:

- This Mac to be awake and connected.
- A signed-in Chrome AlphaSense session.
- The scheduled browser worker.
- Explicit verification of company, date range, source type, usage restriction, downloaded originals, hashes, destination, and handoff.

Do not invent observed URLs, counts, downloads, or completion. Never pass provider-restricted originals into model workflows.

## Known proof gaps and next priorities

1. **Run a real dual-summary audio comparison.** Add one new representative audio file to the root `SUMMARIES` folder and confirm that original Summary and `Auto from SUMMARIES` Lab outputs both complete, are readable, and can be emailed/saved. This incurs real model usage and should be user-driven, not launched merely for QA. T82 changed the code paths this exercises, so it is still the live end-to-end proof: confirm exactly one Lab experiment per recording, that the Telegram message reports the correct Lab state, and that the file moves to `SUMMARIES/Processed` once.
2. **Evaluate Summary Lab quality across several source types.** Compare earnings calls, investor meetings, noisy audio, long YouTube transcripts, and non-earnings documents. Capture which sections are materially better or worse than original Summary.
3. **Validate Lab v4 and decide the long-term pipeline shape.** The original and Lab now
share one research doctrine, including the anchored 1–5 credibility scale and the ban on
unsupported interior-state claims. Lab v4 adds the newer quotation and transcript-repair
discipline. On the next user-driven real source, compare Original and Lab for factual
coverage, useful judgment, quotation density, transcript-repair noise, speaker attribution,
date handling, and PM readability. Then decide whether to promote specific proven Lab
rules into Original, retain both pipelines, or retire the older `summary_comparison`
workflow. Do not remove saved notes or merge workflows before the user makes that choice.
4. **Validate catalyst synthesis on more real folders.** Include single transcript, transcript plus presentation, and multi-broker event folders; score concision, factual attribution, analyst voice, unresolved issues, and PM usefulness.
5. **Prove a complete managed AlphaSense assignment.** Demonstrate browser discovery, source restrictions, original download, iCloud handoff, recap, thesis proposal, and recovery for a real user-selected ticker without overstating unattended coverage.
6. **Improve real-source quality benchmarks.** Current automated checks are useful regressions, not expert certification. Add frozen real-source packs and investor-scored outputs without committing licensed source bodies.
7. **Continue UI simplification.** Navigation and complex evidence workflows have improved but remain dense. Any redesign must be inspected at desktop and mobile widths with real long content. T83 did this for Catalyst synthesis. The same pattern is worth auditing elsewhere: sub-views held in unrouted local state have no URL, no back-button behavior and no way for an alert or sidebar entry to link into them. `agentView`'s remaining panels and the Research agents heading are the obvious next candidates.
8. **Broaden recovery cautiously.** Long-running Summary Lab and several research jobs have bounded recovery; audit remaining model-backed jobs for durable identity, checkpointing, ownership fencing, duplicate prevention, and visible failure. T82 fixed the Summary Lab recovery sweep and the audio completion mirror. T111 closes the saved-Summary-to-Lab creation gap with an atomic save for new opted-in recordings; the live restart scenario remains unproven. T125 resolves the server key or existing saved Settings key for Summary Lab and Improved-note recovery; remaining workers still need a credential/recovery audit.

## Safe commands

Never run `pytest`.

```sh
.venv/bin/python -m unittest discover -s tests/unit
npm run test:frontend
.venv/bin/python scripts/evaluate-research-quality.py --self-test
npm run build
```

For focused Summary Lab work, the most recent safe subset was:

```sh
.venv/bin/python -m unittest \
  tests.unit.test_summary_lab \
  tests.unit.test_summary_lab_recovery \
  tests.unit.test_summary_lab_enqueue \
  tests.unit.test_summary_lab_email \
  tests.unit.test_summary_lab_fanout \
  tests.unit.test_summary_bulk \
  tests.unit.test_summary_job_status \
  tests.unit.test_research_email_format
```

Use `py_compile` for every touched Python module. Do not start paid research, send emails, or approve research to prove a code path.

## Deployment and operational checks

Current release markers must stay synchronized:

- `worker.js`: `2026-10-09T131`
- `service-worker.js`: `20261009-131`
- `src/app.jsx`: `2026-10-09T131`

After an application change:

```sh
npm run build
git push origin main
npx wrangler deploy
curl -fsS https://equity-analyzer-backend.onrender.com/health
```

If the local agent changes:

```sh
launchctl kickstart -k gui/$(id -u)/com.charlie.local-agent
launchctl print gui/$(id -u)/com.charlie.local-agent
tail -80 /tmp/charlie-agent.stderr.log
```

Render may return transient 502 responses while rolling forward. Wait for `/health` to report the exact new commit before treating deployment as complete.

## Repository state warning

At this handoff, T107 is the intended release. The checkout also contains unrelated
local/runtime state. Preserve it. In particular, do not blanket-stage, stash, restore,
or delete:

- `.claude/settings.local.json`
- `.omc/**`
- `nohup.out`
- `.DS_Store`
- untracked generated files under `build/`
- `charlie_investment_research_handoff/`
- local catalyst review scripts/assets unless the assigned task explicitly owns them

Always inspect `git status --short`, stage an explicit allowlist, and review `git diff --cached` before committing.

## Suggested first Claude Code instruction

> Continue Charlie from release T107. Read `AGENTS.md`, `CLAUDE.md`, and `docs/AI_HANDOFF.md` before acting, and use `git log -1` for the exact release commit. Preserve every unrelated dirty or untracked file; do not reset, clean, stash, restore broadly, or stage by directory. T107 broadens catalyst discovery to eleven event families and holds likely duplicate headlines for review. The post-T106 backend changes cover Summary Lab model/cost tracking, Opus 5 token headroom, quotation and transcript-repair discipline, transcription benchmarking, and failed-call usage accounting. Do not replace Gemini transcription with the rejected OpenAI diarization candidate. First inspect `git status --short` and recent history without changing anything, then tell me what you believe is live, what remains unproven, and the exact next item you recommend. Do not launch a paid model run, send email, approve research, modify iCloud originals, or deploy merely for validation. Once we agree on the task, implement it end to end, use only the safe test commands in `AGENTS.md`, stage an explicit file allowlist, update this handoff, and deploy only when the change is complete and no audio job is active.

## Relevant deeper documentation

- `docs/summary-lab.md`
- `docs/catalyst-workflow.md`
- `docs/catalyst-release-T77.md`
- `docs/alphasense-refresh-worker.md`
- `docs/alphasense-collector.md`
- `docs/charlie-roadmap.md`
- `docs/charlie-command-build.md`
- `docs/evidence-workspace.md`
- `docs/earnings-workspace.md`
- `docs/ui-release-audit.md`

Some roadmap/status documents describe earlier releases. Treat this handoff and Git history as the current production baseline, then use older documents for design rationale and proof boundaries.
