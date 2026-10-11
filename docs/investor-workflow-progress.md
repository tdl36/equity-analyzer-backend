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
No job submission/review side effects. 106 frontend tests/build pass. Release/live navigation verification pending.
