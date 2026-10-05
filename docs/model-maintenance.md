# Charlie model maintenance

The investor authorized routine, cost-conscious model upgrades on October 5, 2026.
`model_registry.json` is the reviewed source of truth for primary workflow defaults,
model choices, request settings, text prices, lifecycle dates and release history.
`model_registry.py` is shared by Render and the Mac agent. The browser obtains its
recap default and direct-analysis request settings from `/api/models`. A deployed
registry is immutable for that process; catalog discovery never rewrites defaults.
Environment overrides and explicit saved selections remain pins.

## Current approved decision

Sonnet 5.5 replaces routine Sonnet defaults, including Summary Lab source checks,
transcript cleanup, Mac notes/recaps, analysis and outline generation. It explicitly
uses `between_tools` at high effort, without up-front thinking. GPT-6 Luna replaces
OpenAI mini validation and fast fallback with `reasoning_effort=none` and unchanged
output caps. Opus 4.6 remains the research/composition default; Opus 5.5 and GPT-6.1
Sol are optional selections, not promises of a cheaper report. Haiku and Gemini
transcription remain unchanged. Explicit older saved preferences are preserved.

A lower token price does not guarantee a lower bill. Tokenizer expansion, hidden
reasoning, retries, cache rates, long-context premiums and tools all matter.
Provider SDK request compatibility has synthetic coverage; new-model output
quality has not been established by paid production QA. The initial changes were
specifically accepted by the investor after the model audit.

## Recurring maintenance procedure

The Codex heartbeat "Charlie model maintenance" performs the following daily.
It requires this Mac and Codex to be available, network access, the repository,
provider catalog credentials and existing Render/Cloudflare deployment access.
It does not run in Render. If the Mac is unavailable, no new release is shipped;
Charlie continues using its deployed registry. The UI shows the last completed
check in Settings → AI models, so an old check is visible.

1. Read AGENTS.md, docs/AI_HANDOFF.md and this document. Inspect git status and
   preserve all unrelated work. Inspect previous automation outcomes and any
   pending model-maintenance changes before starting; resume an incomplete
   deployment rather than starting another. Do not modify a dirty file already
   being edited by another task. Wait for active model jobs to finish before a
   deploy or Mac agent restart; never interrupt research to refresh a model.
2. Use read-only provider catalog APIs to establish account availability. Browse
   the official model, migration, pricing and retirement documentation linked in
   the registry. Check OpenAI and Anthropic; do not mistake a "not sooner than"
   support commitment for an announced retirement. Provider catalogs alone are
   insufficient. Never choose a successor by sorting model names. Unknown facts
   stay unknown. No paid model calls solely for testing.
3. Record verified prices, request options and exact announced retirement dates.
   Treat optional image/tool models separately from text prices. The remaining
   October 23 GPT Image 1 and o4-mini retirements need a compatible replacement
   with defensible cost evidence; do not silently substitute a dearer model.
4. For a routine default migration, store evidence in
   `docs/model-migrations/YYYY-MM-DD.json`, keyed by role. Each entry needs
   `officialSources` (official URLs), `accountAvailable`, `compatibilityTested`,
   `qualityEvidence` (specific source-backed rationale, clearly distinguishing
   published evidence from Charlie testing), `tokenMultiplier` (conservative,
   >=1), `noAdditionalThinking`, and `sameOrLowerTokenCaps`. Uncertain token or
   image/tool costs are not an automatic approval. Prices after that multiplier
   must not exceed the previous input, output or cache prices. Preserve native
   PDF/image support, structured output, streaming and completion checks.
5. Run `.venv/bin/python scripts/check_model_policy.py --baseline <old-commit>
   --evidence <evidence-file>`. Protected research/composition, premium OpenAI and
   image defaults cannot be changed by this unattended procedure, even if a
   replacement has lower published token rates. Changes to protected request
   settings also need investor review. Do not weaken the gate to make a proposed
   upgrade pass. Adding optional expensive picker choices also requires review.
   Alert the investor with the concrete trade-off when these rules block a
   necessary retirement migration. Existing per-job pins are never remapped.
6. Add synthetic migration/SDK tests for the exact affected APIs. Keep original
   sources and research untouched. Run focused tests, safe unittest, frontend
   tests, Python compile and production build. Never pytest. Update registry
   history and AI_HANDOFF. Ship a coherent release following AGENTS.md, staging
   only intended files. Verify Render's exact revision, hosted bundle and all
   three versions. Restart/check the Mac agent if registry or agent code changed.
   If deployment fails, record exact commit/stage and resume next wake. Never
   claim deployment or real-source quality without observing it.
7. POST `/api/models/maintenance` using the existing authenticated app client,
   with JSON `{"status":"checked|deployed|attention", "summary":"..."}`.
   The server stamps the time and deployed registry revision durably in Postgres.
   Only mark checked after official-source review succeeds; failed review must
   say attention. Record no secrets, originals or research bodies. No registry
   change means no commit/deploy is needed. Stay quiet on unchanged state; notify
   for a verified upgrade, meaningful failure or required investor decision.

## Runtime and recovery boundaries

Known retired selections fail with an actionable message rather than acquiring a
new identity. New selections hide known retired picker entries. The existing
provider catalog remains discovery only; unreviewed models are not auto-promoted.
Summary Lab now saves its checker ID and registry revision alongside checkpoints;
pre-existing partially checked runs retain their previous checker. A retired
checker leaves an actionable failure rather than combining another checker into
a resumed run. Other existing saved job model IDs remain unchanged. Older
workflows that only store defaults still use the process's deployed registry;
this is why maintenance waits for active work before deploying.

Rollback is a reviewed Git revert of the model release, followed by the same
coherent deployment and Mac agent restart. Never reset unrelated working files.
No universal retirement fallback can guarantee equal quality and cost; when the
provider removes the last eligible model, that workflow requires investor review.

The usage ledger stores the registry revision and normalized cache tokens on new
shared-adapter/Lab calls. Anthropic cache reads/writes are additional to ordinary
input; OpenAI cached reads are included in prompt tokens. GPT-6 long-context
premiums are included. Unknown prices are NULL and counted separately, never free.
Historical ledger rows are not rewritten. Tool fees, image generation, browser-
direct calls, some older helpers and Mac-only calls are not a complete invoice.
