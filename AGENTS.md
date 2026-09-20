# Charlie engineering contract

This file is the shared operating contract for any coding agent working on Charlie. Read `docs/AI_HANDOFF.md` next for the current production state and active priorities. Provider-specific files may add guidance, but they must not weaken these rules.

## Product standard

Charlie is a decision-support platform for professional equity analysts and portfolio managers. Preserve source fidelity, distinguish fact from interpretation, keep uncertainty visible, and require the investor to approve research or thesis changes. Never fabricate sources, completed collection, portfolio data, or successful validation. Never place or recommend trades automatically.

## Architecture

- `app_v3.py`: principal Flask API and older workflows.
- Extracted backend modules such as `summary_lab.py`, `catalyst_comparison.py`, `earnings_decks.py`, `investment_case.py`, and `case_signals.py`: newer bounded workflows.
- `src/app.jsx`: principal React application and routing. Larger newer workspaces live in separate `src/*.jsx` modules.
- `build-frontend.sh`: required frontend build. It compiles React with Babel block-scoping transformation before bundling with esbuild.
- `worker.js`: Cloudflare Worker, static frontend host, and production `/api/*` proxy.
- `charlie_local_agent.py`: Mac agent for iCloud, audio, catalyst, collection, and local processing. It normally talks to production.
- Production backend: Render/PostgreSQL. Production frontend: Cloudflare Workers.

## Protect user data and local state

- Preserve originals in iCloud `STOCKS`, `CATALYSTS`, and `SUMMARIES` folders.
- Do not delete, rename, redirect, or mass reprocess originals unless the user explicitly requests it.
- Do not apply thesis proposals, accept notes, approve recaps, send email, or launch paid research merely for testing.
- AlphaSense authentication and MFA are completed directly in AlphaSense. Never request credentials in chat or bypass authentication.
- The repository may contain unrelated dirty and untracked runtime state. Never run `git reset --hard`, `git clean`, blanket checkout/restore, or broad staging. Stage only files intentionally changed for the assigned task.
- Never expose `.env`, API keys, app passwords, cookies, private headers, licensed source bodies, or private research in commits or logs.

## Safe development and testing

- Never run `pytest`. Repository fixtures can truncate the local database.
- Backend tests: `.venv/bin/python -m unittest discover -s tests/unit`
- Frontend tests: `npm run test:frontend`
- Compile touched Python files with `.venv/bin/python -m py_compile ...`.
- Build production frontend with `npm run build` or `./build-frontend.sh`.
- Use local preview at `http://127.0.0.1:3000/?local=1`; `?local=0` uses production APIs.
- Agent-driven iCloud, audio, catalyst, and collection features normally execute against production even when the UI is local.
- Avoid paid/model-backed production runs solely for QA. Use existing safe unit tests and synthetic fixtures; label mock validation accurately.

## Release procedure

1. Inspect `git status` and preserve unrelated files.
2. Run focused tests, then the relevant broader safe suites.
3. Build the frontend.
4. Bump all three release markers together:
   - `worker.js`: `BUILD_VERSION`
   - `service-worker.js`: `BUILD_VERSION`
   - `src/app.jsx`: `BUILD_VERSION`
5. Commit source plus generated `build/` and `dist/` artifacts required by the change.
6. Push `main`; Render deploys the backend automatically.
7. Deploy the frontend with `npx wrangler deploy`.
8. Wait until `https://equity-analyzer-backend.onrender.com/health` reports the committed revision.
9. If `charlie_local_agent.py` changed, restart it with `launchctl kickstart -k gui/$(id -u)/com.charlie.local-agent` and inspect `/tmp/charlie-agent.stderr.log`.
10. Verify the hosted release and report what was actually tested.

Documentation-only commits do not require a release bump or deployment.

## Definition of done

- The requested behavior is implemented end to end.
- Failures and partial progress remain visible and recoverable.
- Duplicate submission and restart behavior are considered for long-running jobs.
- Desktop and mobile layouts remain readable for affected screens.
- Tests and build pass without destructive fixtures.
- Only intended files are committed.
- `docs/AI_HANDOFF.md` is updated when production state, major behavior, known risks, or the next priority changes.
