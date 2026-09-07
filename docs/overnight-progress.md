# Charlie overnight work — September6–7,2026

User authorized continued implementation, commits, and production deployment while away for a few hours. Preserve unrelated local files and existing research content.

## Completed and verified
- Pilot `f287ec99ad89`, dates2026-08-07 through2026-09-06: all6 searches reconciled. DE45 brokers +1 transcript; ABT34 brokers +0 transcripts; AMT8 brokers +1 transcript. Total89 unique originals.
-59 eligible originals in iCloud `STOCKS/<ticker>/` (DE29, ABT24, AMT6). All hashes match and all59 appear as `folder: main` in the production local-agent manifest, verified2026-09-07T01:56UTC. This preserves both manifest browsing and the existing root-folder new-file alerts. It does not imply automatic thesis regeneration.
-30 originals have AlphaSense GenAI usage flags and remain reference-only outside AI ingestion folders. Do not hand these into STOCKS/CATALYSTS.
- Private ledger/downloads: `~/Library/Application Support/Charlie/AlphaSense/`. Browser inventory JSON and pilot-manifest-verification.json are there, never in Git/public assets.
- AllZIPs exported with max20 selection. DE batches20+20+5; ABT20+14; AMT8. Company IDs DE TK872754, ABT TK903037, AMT TK575464. Company changes reset source/date filters. Do not repeat the finished pilot.
- One DE JPMorgan alternate export differed only by the per-page download timestamp. Both original artifacts retained privately; redundant iCloud handoff removed after content verification. Collector now matches verified AlphaSense document IDs and recognizes that timestamp-only difference; changed content requires review. Search URL alone never proves document identity.
- `com.charlie.collector-monitor` LaunchAgent installed/running, read-only loopback8766, no restart of production file agent. Installer script in scripts/.
- Release0a0a315 pushed to main, Render /api/buildinfo confirmed commit0a0a3157f8ac and agentBatchConcurrency=true. Cloudflare version2026-09-06T02 deployed and assets matched local SHA256.
- Live Chrome phone-width390px QA: research desk, library377 saved summaries, workspace menu, and saved Abbott thesis loaded. Existing thesis content preserved.
- Follow-up release target2026-09-06T03 adds a collapsible mobile thesis action panel with44px touch targets, avoiding clipped tiny action buttons. Build and26 focused tests passed (9 frontend+17 standalone Python). Finish deployment verification if still pending.

## Continue overnight (bounded enhancements)
1. Read private deployment-receipt.json when present; verify latest release before doing new work.
2. Improve monitor clarity: show STOCKS/<ticker>/ destination alongside each handoff; separate reference-only held originals from actionable pending staging; add ticker/status filters for89-document runs. Show last verified production-manifest counts without claiming backend file bytes/thesis generation. Keep read-only/loopback-only and protect private paths.
3. Harden resumability: persist verified per-document identities/batch observations; test interrupted ZIP validation, stale search evidence, duplicate exports, and unavailable authentication. Do not repeat completed downloads or launch paid research. Do not process restricted originals with AI.
4. Inspect responsive research pipeline and theme selection; fix concrete layout regressions within existing data flows. Do not redesign unrelated workflows or mutate saved theses.
5. Commit/deploy completed app fixes with matching worker/frontend/service-worker versions. Explicit staged website assets only (.worker-assets), never repository root. Preserve unrelated .omc/.claude/nohup/research handoff files. Run focused tests, never pytest because global conftest truncates testDB.

Finite Codex heartbeat `continue-charlie-overnight` has four hourly follow-ups; stop no later than2026-09-07T05:45UTC or earlier if scoped work is complete. Mac must remain awake and Codex available for local browser work. Authentication/MFA must be completed by the user in AlphaSense; no credentials in chat. No permanent daily collection schedule exists.

## Overnight pass1 —02:45UTC
- Release262f1cb fully verified in private deployment receipt: Cloudflare2026-09-06T03, matching assets, Render262f1cbcd3b1. Mobile thesis action panel verified at390px.
- Monitor clarity work complete: ticker-relative destinations, held/pending separation, combined company/status/text filters,25-row pagination, visible verification timestamps.
- Added `charlie_collector.py verify RUN_ID`: read-only production manifests plus local hash check; ledger-bound fingerprints prevent old checks appearing current after a handoff changes. Errors retain previous timestamped evidence and do not leak credentials.
- Production verification at02:43UTC passed all59 originals: ABT24, AMT6, DE29. No new downloads or research jobs. Updated stale Deere search note to reflect completed45-report reconciliation.
-30 focused tests pass (19Python +11frontend). Native Chrome filter QA: Deere46 originals, Deere held17, Deere held/Wolfe7. Read-only endpoint rejects foreign origins/hosts and POST; JSON contains no /Users paths.
- Local monitor LaunchAgent restarted to serve upgrade. These are local collector changes; public app version remains2026-09-06T03.
- Remaining: authentication pause/resume while partially collected, stricter completion evidence and interrupted-archive tests, final mobile theme inspection. Do not redo completed pilot or monitor filters.

- Phone-width390px monitor QA complete: filters fit within the viewport; originals use cards with visible destination/status. Pagination confirmed26–50 of89.
