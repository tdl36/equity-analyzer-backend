# Charlie workspace redesign

The interface now groups Charlie's existing tools around the work an investment researcher is doing. Research engines, database schemas, generated content, and deployment configuration are unchanged.

## Implemented

1. Five primary workspaces: Today, Companies, Library, Create, Automations. All 20 existing destinations remain available, including Investment Review on mobile.
2. A compact desktop sidebar, tablet/phone navigation, searchable workspace dialog, keyboard focus management, and browser zoom support.
3. Today shows real saved coverage, recent summaries, alerts, and upcoming meetings.
4. Companies combines saved theses and overviews and carries ticker context into research workflows. URL hashes preserve workspace and company through refresh.
5. Library searches saved summaries by company, title, and source; links lead directly into reading.
6. Focused summary reading hides library filters. Overview prose has a bounded reading width and more generous type and leading.
7. Create describes outputs and their source requirements before entering the existing generation tools. Studio uses restrained, theme-aware selection tiles.
8. Server-rendered investment reviews live in script-free, automatically sized iframes so report CSS cannot override the application.
9. Presentation previews load with an explicit project ID and ignore stale image responses. Preview precedes editing, and phones use a collapsible slide navigator.
10. Categorized Settings, readable paper-surface inputs, correct RFC/SQL date parsing, configured media API URLs, retryable feed errors, and distinct loading/empty/error states for ticker activity. Local-agent status is available in the header instead of a permanent warning banner.

## Source organization

- `src/workspace.jsx`: workspace shell, hubs, isolated report renderer.
- `src/workspace-model.mjs`: shared navigation, URL parsing, date normalization, coverage merge.
- `src/workspace.css`: theme-aware workspace layout and focused legacy-screen improvements.
- `src/app.jsx`: integration with existing state and APIs; slide/feed fixes.
- `tests/frontend/workspace.test.mjs`: navigation coverage, route, date, and merged coverage regressions.

## Validation

- `npm run test:frontend`: five passing test groups; no database access.
- `npm run build`: Babel, esbuild, and Tailwind production bundle.
- Browser checks against the local backend: existing research and automation destinations at 390px; tablet at 768px; desktop at 1280/1440px; all four themes.
- Opened a saved DE summary and overview, RVW3 review, and UnitedHealth presentation. Confirmed review CSS stays isolated, first slide preview loads, and company context survives overview refresh.
- No generation jobs, sends, deletes, database migrations, or deployments were performed as part of validation. External service failures and paid generation paths were not exercised end to end.

## Rollback checkpoint

A full verified repository backup predates these changes:
`/Users/tonydlee/Projects/charlie-backups 09062026/before-redesign_2026-09-06_13-45-58_-0400`

Use its `RESTORE.md` to restore into a separate directory first. The archive includes ignored/untracked files and Git history. External PostgreSQL data, iCloud, Keychain, and deployed state are outside that repository backup.
