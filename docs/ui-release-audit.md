# Charlie UI release audit — T67

## Findings and repairs

| Finding | Repair |
| --- | --- |
| White broker/underweight fields with pale text in dark themes | Shared theme-aware input defaults; valid color tokens; native dark controls |
| New research features used raw inline controls and cramped paragraphs | Shared research-panel spacing, readable form labels, action buttons and prose |
| Narrow forms could exceed a fieldset's intrinsic width | Zero minimum fieldset width, bounded controls, stacked mobile filters |
| Mobile meeting heading and question actions crowded together | Wrapped heading, full-width title, separate action row |
| Meeting sources were truncated and showed stray zero values | Wrapping filenames, explicit positive-count checks, named remove controls |
| Research desk ignored the current company | Pass the route company into case/evidence workspaces |
| Research chat could return focus to its removed close button | Capture the opener before setting modal focus; restore it on close |
| Mobile navigation could cover a research overlay | Explicit overlay stacking above bottom navigation |
| Local preview retained a September 8 stylesheet | Restarted preview; Tailwind watcher now survives closed stdin |
| Intermittent live API 500s | Captured `connection pool exhausted`; bounded acquisition wait, synchronized initialization, cursor cleanup, a 20-connection pool matching 16 HTTP threads plus background work, and four concurrent API reads per browser tab |

The pool has a fixed cap of 20 connections per process (one configured worker),
replacing the inconsistent cap of 10 for 16 HTTP threads. Waiting does not
dynamically enlarge that cap and does not retry transactions. It
allows up to eight seconds for a returned connection. Sustained saturation can
still time out and must remain visible. This is not a claim that all backend
failure modes have been eliminated.

## Coverage and evidence

- 26 main routes × 1440, 768, 390 and 320 CSS pixels: 104 production-build browser
  checks. No JavaScript page exceptions or document-level horizontal overflow.
  The scan correctly failed on intermittent API errors; captured error bodies
  identified pool exhaustion. A hosted T66 follow-up still hit the bounded wait;
  this led to the capacity-alignment and browser-read scheduling fixes in T67. Initial stale-preview results were superseded by
  tests against `.worker-assets`, the exact staged deployment assets.
- Eleven research-desk sections opened at desktop and phone widths, with live
  read-only data and screenshots. These are screen-level checks, not proof that
  every generation branch in each section succeeds.
- Saved ABT summary and meeting pack opened; case proposal, decisions and evolution
  tabs checked; workspace search navigation exercised at 1440 and 390 pixels.
- Actual saved ABT proposal fixture: 11 changes, three review-ready, eight flagged;
  verified source-selection limits, preserved acceptance payload and blocked
  acceptance of flagged drafts with mocked callbacks.
- Deterministic theme test: seven themes × four widths. Control text contrast at
  least 4.5:1, mobile form text at least 16px, no overflowing preference field,
  dialog focus placement, Escape and focus restoration.
- 34 frontend model tests; 21 focused Python unit tests for pool contention and
  cleanup, proposal checks, bulk Summary sections and manual meeting profiles.
- Existing mocked browser checks for bulk Summary controls, source selection,
  and thesis evolution/underweight condition entry passed.

Screenshots and live response data stay in `/tmp/charlie-ui-audit`, not in Git.
No production POST/PUT/DELETE request was submitted by the browser audits. No
research was accepted, regenerated, deleted or emailed during verification.

## Repeatable release checks

Run from the repository root after `npm run build`. Chrome and the Node
`playwright` package are prerequisites. Set `PLAYWRIGHT_MODULE` to its absolute
module directory if it is supplied outside this repository's `node_modules`.

```sh
npm run test:frontend
node scripts/check-ui-foundations.cjs
python3 -m http.server 8788 --bind 127.0.0.1 --directory .worker-assets
```

In another terminal, supply a private JSON file containing authorized API request
headers through `AUDIT_HEADERS`; never commit that file. The audit blocks all
non-GET API requests, proxies preview API reads to the production API, and exits
nonzero for HTTP errors, JavaScript errors or page overflow.

```sh
AUDIT_HEADERS=/private/path/headers.json node scripts/audit-ui.cjs
```

Options: `AUDIT_URL`, `AUDIT_API`, `AUDIT_WIDTHS` (comma-separated CSS widths),
`AUDIT_OUTPUT` (default `/tmp/charlie-ui-audit`). Screenshots contain private
research; keep that output local. A failed run is a release investigation, not a
reason to weaken the checks. Inspect screenshots and scrolled readers as well:
absence of document overflow cannot prove absence of clipping inside a panel.

## Remaining proof boundaries

This audit uses Chrome desktop and responsive emulation. Physical iPhone Safari,
Android browser/keyboard behavior, Windows-specific rendering, actual email and
file delivery, long-running model generation, OCR, AlphaSense authentication and
recovery after Mac sleep require separate end-to-end runs. The audit does not
certify research accuracy or establish that every application state is error-free.
