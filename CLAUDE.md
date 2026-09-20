# Claude Code entry point — Charlie

Read these files before editing:

1. `AGENTS.md` — shared engineering, safety, testing, and release rules.
2. `docs/AI_HANDOFF.md` — current production release, latest changes, proof gaps, and priorities.

## Essential context

Charlie is an institutional equity-research and decision-support application. The current production baseline is recorded in `docs/AI_HANDOFF.md`. Preserve source provenance, explicit uncertainty, investor approval, iCloud originals, and all unrelated local state.

The repository is intentionally not clean because local agent and development artifacts coexist with the application. Never use destructive Git cleanup or broad staging. Start every task with `git status --short`; stage only an explicit file allowlist.

Never run `pytest`; its repository fixtures can truncate the local database. Use the safe unittest and frontend commands in `AGENTS.md`.

Before implementation, state the concrete outcome you will deliver. Persist through testing and deployment when authorized. Do not describe a queued job, saved policy, mocked test, or HTTP submission as proof that a real research workflow completed.

For historical architecture notes, the user's local Claude memory may contain additional context, but tracked repository documentation and current code are authoritative when they differ.
