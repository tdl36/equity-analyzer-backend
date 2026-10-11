# Investor workflow build session

Authorized October 10, 2026, approximately 9:16–11:16 p.m. Eastern
(2026-10-11 01:16–03:16 UTC). Finite continuation automation:
`continue-charlie-overnight`. Stop starting feature work at the cutoff and delete
that automation after recording a safe checkpoint.

Baseline: T146, source d458c5c; verification handoff fd00e27. Existing unrelated
runtime changes must remain untouched. Never pytest or paid research for QA.

## Current increment

Build a read-only change brief in the investment-case workspace from existing
case history, recorded proposal reviews and evidence proposals. No model calls.
Use an actual recorded review as a cutoff when available; otherwise explicitly
select a saved revision. Distinguish saved changes, new accepted evidence and
unapproved proposals. Preserve source excerpts and display bounded coverage.
Do not imply that a saved version or an opened page is investor approval.

T147 implemented; 947 backend and 103 frontend tests pass, Python compile and
production build pass. Deployed and verified backend 3893c89, hosted bytes, authenticated reads and
desktop/mobile layout. No paid
research or review mutations were used. After this increment, prioritize reusable coverage policy/recovery
improvements and stronger issuer/event validation without duplicating existing
engines. Record exact tests, release revision, verification and remaining gaps.

## Next increment: T148

Implemented read-only intake wait explanations plus the existing daily cap control.
Focused tests (22), all 952 backend tests, 103 frontend tests, compile and build
pass. Backend ebc195a and hosted bytes verified; live wait reasons confirmed. The
T147 change brief needs a saved investment-case baseline; UNH has a separate legacy
thesis but no saved investment-case revisions, so do not fabricate a comparison.
T148 improves that empty state with a Current thesis link. Do not click controls
that enable/rebudget live paid automation for QA.

## T149 in progress

Conservative saved-company-name/ticker attribution for catalyst candidates.
23 focused tests, 956 full backend tests, 103 frontend tests, compilation and
build pass. Backend d2ccf67 and hosted T149 bytes verified. No paid QA.

## T150 in progress

Direct workflow sections from Activity; URL/reload/back-forward support for
Research desk and Stock Analysis. Company navigation preserves ticker context.
No job submission/review side effects. 106 frontend tests/build pass. Backend 095a20f and hosted bytes verified. Section links/reload/Back and Activity
to SYK collection inspected.

## T151 in progress

Deere/Stryker profiles, art and verified segment mappings; generic renderer no
longer inherits UNH copy. Real public financial retrieval succeeded. DE/SYK public
data-only and saved UNH PDFs each one page, Calibri/black text, no clipping.
See visual-business-profile-validation.md. 960 backend tests, 106 frontend tests, compile/build pass. Backend 35e2e1f and hosted T151 bytes verified; Mac healthy.

## T152 in progress

Collection exceptions first and company-scoped; whole-claim infographic content
budgets prevent tested overflow, preserve paired arguments and linked monitors.
Actual saved UNH plus DE/SYK public-data-only PDFs each one page; stress fixtures
200/400/800 characters pass. 963 backend tests, 109 frontend tests, compile/build pass. Deployment pending.
