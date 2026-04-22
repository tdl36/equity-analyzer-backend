# Charlie Phase 2 — Media Trackers: Podcast Pilot (Design Spec)

**Date:** 2026-04-22
**Status:** Approved — ready for implementation plan
**Author:** Tony Lee + Claude
**Supersedes:** None (first spec for Phase 2 media trackers)
**Scope tag:** Phase 2a (Podcast). Phases 2b–2d (YouTube, X, LinkedIn) are separate specs that reuse the shared infra built here.

---

## 1. Context & Motivation

Charlie today is a manual research tool: user uploads docs, runs syntheses, maintains thesis notes by hand. The agentic-platform vision (see `charlie_agentic_vision.md`) calls for autonomous agents that work 24/7 — surfacing insights, tracking theses, monitoring media. This spec covers the first block of that vision: **Media Trackers**, starting with **Podcasts**.

Inspiration: Oliver Cox's "Kabuten" tool at JPM APAC Tech (podcast tracker + X heatmap + 30+ sector analyst agents). Screenshots of his Podcasts tab and Social Heatmap informed the UI direction here.

**Why podcasts first:** highest signal-to-noise in Tony's buy-side workflow (CEO interviews, sell-side strategists, sector specialists). Many top finance pods publish transcripts, keeping transcription cost low. Pattern-transfers cleanly to YouTube, X, LinkedIn in follow-on phases.

---

## 2. Scope

### 2.1 In Scope (this spec)
- **Podcast tracker** (ingestion → transcription → extraction → alerts)
- **Shared infrastructure** reused by future trackers: APScheduler on backend, `media_feeds` / `media_episodes` / `media_digest_points` / `signals_watchlist` / `media_theme_clusters` / `notification_prefs` tables, source-type column for future expansion
- **Feed tab** (new) — firehose view of all ingested podcast content
- **Alerts tab enhancements** — material-only stream with source-type filters
- **Settings — Media Trackers sub-panel** — feeds CRUD, watchlist, mutes, notification preferences, cost gauge
- **Notifications across 4 channels** — in-app tab, web push, Telegram, 7am email digest
- **Theme clustering** — weekly Sunday 5am Sonnet pass over material points
- **Soft cost guardrail** at $15/week (configurable) + hardcoded $50/week backstop

### 2.2 Out of Scope (deferred to later specs)
- YouTube, X/Twitter, LinkedIn trackers (Phase 2b–2d)
- Social Heatmap UI (Phase 2c, tied to X tracker)
- Agentic analyst team, Earnings Autopilot, "Ask Charlie" (Phase 3+)
- Scheduled Monitor-style user-defined recurring prompts (separate feature)
- Vector-DB / semantic search over transcripts
- Multi-user / team features
- OPML import, blog-post crawling for audio-less pods, share-to-Slack buttons, sponsored-segment stripping — intentionally deferred; revisit after 4–6 weeks of real usage

---

## 3. Design Decisions (locked during brainstorming)

| # | Decision | Chosen | Rationale |
|---|---|---|---|
| 1 | Approach | Shared infra + one pilot tracker | De-risks hard parts (transcription, dedup) before stamping out 3 more |
| 2 | Pilot source | Podcasts | Best signal-to-noise for buy-side; highest pre-made transcript availability |
| 3 | Feed management | Hybrid — 25 seed feeds + full CRUD UI | Ship value day 1, maintain ergonomically |
| 4 | Transcription | Scrape publisher transcript → Gemini 2.0 Flash fallback | Best signal/cost ratio; ~60% free, ~40% at $0.10/ep |
| 5 | Signal extraction | Two-tier: firehose digest + material alerts | Matches Material/Incremental/No Change taxonomy, lower friction than filter-only |
| 6 | Scheduler | APScheduler on Render backend, persistent SQLAlchemy jobstore | Runs 24/7 independent of Tony's Mac; persists across redeploys |
| 7 | Coverage universe | 36 saved tickers + manually curated signals watchlist | Captures competitor/exec/theme signals ticker-only matching misses |
| 8 | Surfacing | Alerts tab (material only) + new Feed tab (firehose) | Clear mental model; nav consolidation TBD later |
| 9 | Dedup | Both per-episode (7-day window) AND weekly theme clustering | Shipping both, A is cheap, B is the step-up in synthesis |
| 10 | Cadence / backfill | 30-min poll / 7-day backfill on feed add | Standard newsroom interval; meaningful day-one history without runaway cost |
| 11 | Seed list | Claude drafts defaults, Tony redlines before deploy | Fastest path to tailored day-1 coverage |
| 12 | Notifications | All 4 channels (tab + push + Telegram + email), each toggleable | User has multiple surfaces; throttling per-channel handles noise |
| 13 | Cost guardrail | Soft warning only at $15/wk (configurable), hard $50/wk backstop in code | Auto-shutoff risks missing signal in earnings weeks; warn + backstop balances safety |
| 14 | Per-item mute | Individual podcasts, tickers, themes each mutable without delete | Lets Tony dial noise without losing history |

---

## 4. Architecture

```
                    ┌──────────────────────────────────────────┐
                    │  Backend (Render, Flask + APScheduler)   │
                    │                                          │
  RSS feeds ──────▶ │  feed_poller   (every 30 min)            │
                    │       │                                  │
                    │       ▼                                  │
                    │  media_episodes   (new rows, status=new) │
                    │       │                                  │
                    │       ▼                                  │
                    │  transcription_worker (every 2 min)      │
                    │   ├─ scrape publisher transcript         │
                    │   └─ fallback: Gemini 2.0 Flash audio    │
                    │       │                                  │
                    │       ▼                                  │
                    │  extraction_worker (every 2 min)         │
                    │   ├─ Haiku: bullet points + tags/tickers │
                    │   ├─ Sonnet: material gate (2nd pass)    │
                    │   └─ dedup: (ticker,guest,theme,7d)      │
                    │       │                                  │
                    │       ▼                                  │
                    │  agent_alerts (material) + media_digests │
                    │       │                                  │
                    │       ▼                                  │
                    │  theme_clusterer (Sunday 5am, Sonnet)    │
                    │       │                                  │
                    │       ▼                                  │
                    │  notifier  →  push / Telegram / email    │
                    │                                          │
                    └──────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  Frontend (Cloudflare, index.html)       │
                    │  • Alerts tab   — material only          │
                    │  • Feed tab     — firehose digest (new)  │
                    │  • Settings     — feeds, watchlist,      │
                    │                   mutes, notif channels  │
                    └──────────────────────────────────────────┘
```

**Reuse of existing infrastructure:**
- `agent_alerts` table already exists — material podcast points write rows with `alert_type='podcast_material'`
- Alerts tab UI at `activeTab === 'alerts'` already renders alerts; no changes to reads, just richer `detail` JSONB + filter pills
- Tavily already integrated (may be used by future trackers, not Phase 2a)
- Local agent `charlie_local_agent.py` is *not* modified for Phase 2a — backend does all the work
- `CHARLIE_API_KEY` auth via `require_auth()` gates all new endpoints
- Cloudflare Worker proxy (`worker.js`) requires no routing changes — backend origin handles `/api/media/*` automatically

---

## 5. Data Model

All new tables added to `init_db()` in `app_v3.py`.

```sql
-- 1. Feed registry (seeded + user-managed)
CREATE TABLE media_feeds (
    id              VARCHAR(100) PRIMARY KEY,
    source_type     VARCHAR(20) NOT NULL,   -- 'podcast' (future: 'youtube','x','linkedin')
    name            TEXT NOT NULL,
    feed_url        TEXT NOT NULL,
    sector_tags     TEXT[] DEFAULT '{}',
    muted           BOOLEAN DEFAULT FALSE,
    last_polled_at  TIMESTAMP,
    last_episode_at TIMESTAMP,
    poll_interval_min INT DEFAULT 30,
    error_count     INT DEFAULT 0,
    last_error      TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 2. Ingested episodes
CREATE TABLE media_episodes (
    id              VARCHAR(100) PRIMARY KEY,
    feed_id         VARCHAR(100) REFERENCES media_feeds(id) ON DELETE CASCADE,
    guid            TEXT NOT NULL,
    title           TEXT NOT NULL,
    published_at    TIMESTAMP,
    audio_url       TEXT,
    source_url      TEXT,
    show_notes      TEXT,
    duration_sec    INT,
    transcript      TEXT,
    transcript_source VARCHAR(20),           -- 'publisher' | 'gemini' | 'show_notes_only' | NULL
    status          VARCHAR(20) DEFAULT 'new',
    -- Status lifecycle: 'new' → 'transcribing' → 'extracting' → 'done' | 'failed' | 'skipped'
    error_message   TEXT,
    cost_usd        NUMERIC(10,4) DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(feed_id, guid)
);

-- 3. Bullet-level extraction points (Oliver-style tagging)
CREATE TABLE media_digest_points (
    id              VARCHAR(100) PRIMARY KEY,
    episode_id      VARCHAR(100) REFERENCES media_episodes(id) ON DELETE CASCADE,
    point_order     INT NOT NULL,
    text            TEXT NOT NULL,
    tickers         TEXT[] DEFAULT '{}',
    sector_tags     TEXT[] DEFAULT '{}',
    theme_tags      TEXT[] DEFAULT '{}',
    timestamp_sec   INT,                     -- position in episode (optional)
    material        BOOLEAN DEFAULT FALSE,
    cluster_id      VARCHAR(100),            -- populated weekly
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 4. Signals watchlist (ticker, keyword, exec — beyond the 36 covered names)
CREATE TABLE signals_watchlist (
    id              VARCHAR(100) PRIMARY KEY,
    kind            VARCHAR(20) NOT NULL,    -- 'ticker' | 'keyword' | 'exec'
    value           TEXT NOT NULL,
    associated_ticker VARCHAR(20),           -- which covered name this signal relates to
    muted           BOOLEAN DEFAULT FALSE,
    note            TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(kind, value)
);

-- 5. Weekly theme clusters
CREATE TABLE media_theme_clusters (
    id              VARCHAR(100) PRIMARY KEY,
    theme           TEXT NOT NULL,
    summary         TEXT,
    point_ids       TEXT[] DEFAULT '{}',
    primary_tickers TEXT[] DEFAULT '{}',
    week_start      DATE NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 6. Notification / cost preferences (key/value)
CREATE TABLE notification_prefs (
    key             VARCHAR(50) PRIMARY KEY,
    value           JSONB
);
-- Seed rows:
--   'channels'            → {tab:true, push:true, telegram:true, email:true}
--   'email_digest_time'   → '07:00'
--   'email_digest_to'     → 'tonydlee@gmail.com'
--   'telegram_chat_id'    → '...'
--   'cost_weekly_warn_usd'→ 15.00
--   'push_subscriptions'  → []                (filled as browsers opt in)

-- Indexes
CREATE INDEX idx_episodes_status ON media_episodes(status);
CREATE INDEX idx_episodes_feed_published ON media_episodes(feed_id, published_at DESC);
CREATE INDEX idx_points_episode ON media_digest_points(episode_id, point_order);
CREATE INDEX idx_points_tickers_gin ON media_digest_points USING GIN(tickers);
CREATE INDEX idx_points_material ON media_digest_points(material, created_at DESC);
CREATE INDEX idx_points_cluster ON media_digest_points(cluster_id);
```

**Design properties:**
- `source_type` column on `media_feeds` makes future YouTube/X/LinkedIn additions a table-extension, not a new table
- `UNIQUE(feed_id, guid)` makes RSS re-polls idempotent
- Bullet-level `material` flag (not episode-level) matches Oliver's UX and lets the material gate fire on any single point without dragging the whole episode
- `cluster_id` nullable → firehose works day one, clustering is purely additive
- `muted` column on `media_feeds` and `signals_watchlist` supports per-item mute UX
- `agent_alerts` table NOT modified — material points write into it using existing schema

---

## 6. Pipeline

### 6.1 Scheduler setup

New file `scheduler.py`, initialized from `app_v3.py` at app boot:

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

scheduler = BackgroundScheduler(
    jobstores={'default': SQLAlchemyJobStore(url=DATABASE_URL)},
    job_defaults={'coalesce': True, 'max_instances': 1, 'misfire_grace_time': 300},
)
scheduler.add_job(poll_all_feeds,      'interval', minutes=30, id='feed_poller')
scheduler.add_job(process_transcribe,  'interval', minutes=2,  id='transcribe_worker')
scheduler.add_job(process_extract,     'interval', minutes=2,  id='extract_worker')
scheduler.add_job(cluster_weekly,      'cron', day_of_week='sun', hour=5, id='clusterer')
scheduler.add_job(send_email_digest,   'cron', hour=7, minute=0, id='email_digest')
scheduler.add_job(check_cost_warning,  'cron', hour=23, minute=0, id='cost_watch')
scheduler.start()
```

**Why persistent jobstore:** Render restarts mid-deploy; without persistence we'd double-fire or miss ticks. Also lets us inspect next-fire times for observability.

**Kill switch:** `app_settings.media_scheduler_enabled=false` skips all job bodies without unregistering them. Instant pause without a deploy.

### 6.2 Stage 1 — `poll_all_feeds()`

Every 30 min:
```
for feed in media_feeds WHERE muted=false AND
        (last_polled_at IS NULL OR last_polled_at < NOW() - poll_interval_min*'1 min'):
    try: entries = feedparser.parse(feed.feed_url)
    for entry in entries:
        upsert into media_episodes (feed_id, guid) ON CONFLICT DO NOTHING
        if published_at < (first_run ? NOW()-7d : feed.last_episode_at): skip (backfill guard)
    UPDATE feed SET last_polled_at=NOW(), last_episode_at=max(entry.published), error_count=0
    on error: error_count++, last_error=str(e), exponential backoff (30m→1h→2h→4h),
              after 5 errors: auto-mute + create_alert(type='system', ticker='', title='Feed X muted due to errors')
```

7-day backfill on feed-add: `backfill_feed(feed_id, days=7)` runs once immediately after INSERT, populating episodes synchronously.

### 6.3 Stage 2 — `process_transcribe()`

Every 2 min, pick up to 3 episodes WHERE `status='new'` ORDER BY `published_at DESC`. Claim with `UPDATE ... WHERE status='new' RETURNING *` for atomicity:

```
SET status='transcribing'
transcript = try_scrape_publisher_transcript(episode.source_url)
if transcript: transcript_source='publisher', cost=0
else:
    if duration_sec > 7200: skip (too long)
    audio = download(audio_url, timeout=120, max_size=150MB)
    transcript = gemini_audio_transcribe(audio, model='gemini-2.0-flash')
    transcript_source='gemini'; cost = gemini_audio_rate * duration_sec/3600
if all paths fail: transcript=None, transcript_source='show_notes_only'
SET status='extracting'
```

**Publisher scrapers** (v1 set; each ~30 LOC):
- Invest Like the Best (joincolossus.com)
- Odd Lots (Bloomberg transcript pages)
- Acquired (acquired.fm)
- Capital Allocators (capitalallocators.com)

Additional scrapers added as we identify high-value feeds with transcripts; fall-through to Gemini is always safe.

### 6.4 Stage 3 — `process_extract()`

Every 2 min, claim up to 3 episodes WHERE `status='extracting'`:

```
points = call_llm(
    tier='fast',                           # Haiku
    system=EXTRACTION_PROMPT,
    input=transcript or show_notes,
    response_format=JSON_SCHEMA
)
# Returns list of {text, tickers[], sector_tags[], theme_tags[], timestamp_sec}

covered  = {row.ticker for row in saved_analyses}
wl       = signals_watchlist WHERE muted=false

for p in points:
    candidate_material = (
        any(t in covered or t in wl.tickers for t in p.tickers)
        or any(k in (wl.keywords + wl.exec_names) for k in (p.theme_tags + [p.text]))
    )
    if candidate_material:
        verdict = call_llm(tier='standard', system=MATERIAL_JUDGE_PROMPT, input=p)
        p.material = verdict.is_material

    if p.material:
        # 7-day dedup scoring: an existing material alert within 7 days is a duplicate if
        #   (ticker_overlap >= 1 ticker)
        #   AND (theme_overlap >= 1 tag  OR  guest_match_from_title)
        # guest_match is a cheap regex check for "with <Name>" / "feat. <Name>" in both titles.
        # primary_ticker(p) := p.tickers[0] if non-empty else NULL.
        dup = find_duplicate_alert(p, window='7d')
        if dup:
            update agent_alerts SET detail = jsonb_set(detail, '{related_episodes}',
                   detail->'related_episodes' || episode_link)
        else:
            insert agent_alerts(alert_type='podcast_material', ticker=primary_ticker(p),
                                title=p.text[:120], detail={...})

    insert media_digest_points(...)

SET episode.status='done', cost_usd = accumulated
enqueue_notifications(new_material_alerts)  # see Section 7
```

Two-pass material gate: Haiku is fast/cheap for structured extraction but over-flags; Sonnet second-pass rejects boilerplate ("NVIDIA continues to dominate AI"). Saves reader attention.

### 6.5 Stage 4 — `cluster_weekly()` (Sunday 5am)

```
points = media_digest_points WHERE created_at >= NOW() - 7d AND material=true
clusters = call_llm(tier='standard', system=CLUSTERING_PROMPT, input=points_summary)
# Returns [{theme, point_ids[], summary, primary_tickers[]}, ...]
upsert media_theme_clusters for week_start = most_recent_monday()
UPDATE media_digest_points SET cluster_id=... for each clustered point
```

Clusters only material points (firehose would be too noisy/expensive). Weekly cadence lets themes congeal.

### 6.6 Stage 5 — `check_cost_warning()` (daily 11pm)

```
weekly_cost = SUM(cost_usd) FROM media_episodes WHERE created_at >= NOW() - 7d
soft_limit  = notification_prefs['cost_weekly_warn_usd']  (default 15.00)
hard_limit  = 50.00  # hard-coded in code
if weekly_cost > hard_limit:
    app_settings.media_scheduler_enabled = false  # auto-pause transcription
    create_alert(type='system', title=f"Hard $50/wk cap hit — trackers paused")
elif weekly_cost > soft_limit:
    create_alert(type='system', title=f"Media tracker spent ${weekly_cost:.2f} this week")
```

Hard limit is last-resort belt-and-suspenders; the intended operating mode is soft warning only.

### 6.7 Manual scanner

`POST /api/media/run-scanner` — fires `poll_all_feeds()` in a thread, returns `202 {scan_id}`. UI polls status via `GET /api/media/scan/:id` (reuses existing job-status pattern).

### 6.8 Concurrency

All three stages claim rows with atomic `UPDATE ... WHERE status=EXPECTED RETURNING *` (or `SELECT ... FOR UPDATE SKIP LOCKED` if we prefer SELECT-then-UPDATE). Re-entrant scheduler ticks cannot double-process an episode.

### 6.9 Failure modes

| Failure | Handling |
|---|---|
| RSS 4xx/5xx | `error_count++`, exponential backoff, auto-mute after 5 |
| Audio download fails | `status='failed'`, log; manual retry via `POST /api/media/episode/:id/retry` |
| Gemini audio error | Fall through to show-notes extraction with `transcript_source='show_notes_only'` |
| Haiku JSON invalid | Use existing `_repair_truncated_json()` in `app_v3.py`, retry once, else fail episode |
| Dyno restart mid-stage | Episode status is single source of truth; next tick picks up wherever it was |
| APScheduler lost ticks after restart | `misfire_grace_time=300` runs up to 5 min late; otherwise skipped (acceptable — 30 min cycle) |

---

## 7. Notifications

Four channels, each independently toggleable via `notification_prefs.channels`.

### 7.1 In-app tab
- Alerts tab already exists; material points create rows that appear naturally
- Alert-count badge (`/api/alerts/count`) auto-updates existing badge UI

### 7.2 Web push
- Service-worker already registered in `service-worker.js`; add `push` event listener showing notification
- Backend `send_push(alert)` uses `pywebpush` library, iterates `notification_prefs.push_subscriptions[]`
- Subscribe UI in Settings — "Enable push on this device" prompts browser permission, captures `PushSubscription` JSON, POSTs to `/api/media/notification-prefs/push-subscribe`
- Throttled: max 1 push every 10 min (batched); pushes include count + top ticker

### 7.3 Telegram
- Uses existing Telegram bot (plugin already configured)
- `send_telegram(alert)` posts to Telegram Bot API `sendMessage` with `chat_id` from `notification_prefs.telegram_chat_id`
- Same batching — max 1 message per 30 min; batches list ticker counts + first 3 titles

### 7.4 Email digest
- Daily 7am cron (`send_email_digest()`) aggregates last-24h material alerts
- SMTP via existing backend mail config (Catalyst feature already uses it)
- HTML template with alert cards grouped by ticker, deep-link to Alerts tab
- Skipped on days with zero material alerts

### 7.5 Per-channel test buttons
`POST /api/media/notification-prefs/test` with `{channel: 'telegram'}` sends a test ping to verify configuration without waiting for real content.

---

## 8. Frontend

### 8.1 New Feed tab

- Added to `activeTab` enum, entry in bottom-nav "More" overflow
- Layout per Section 4.1 of brainstorming:
  - Header: name + "Run Scanner" CTA + stats (feeds tracked, last scan, new today)
  - Filter row: source pills, sector/ticker/date selects, search, muted toggle
  - Card list: one card per episode, each bullet = one `media_digest_points` row with sector/theme tag chips and `[★]` for material
  - Overflow menu per card: mute feed/ticker/theme, mark material, dismiss
  - Linkify tickers to Portfolio tab
  - `SOURCE ↗` opens `episode.source_url` in new tab
  - Polls `GET /api/media/feed?...` every 60s while active

### 8.2 Alerts tab (light edits)

- Filter pill row adds `[Material] [Podcasts] [Thesis] [System]`
- Podcast-material alert cards include: ticker badge, episode source link, "View in Feed" deep-link
- Additional actions: "Mute this ticker/theme" (hits watchlist or feeds endpoints)

### 8.3 Settings — Media Trackers accordion

Four collapsible sub-sections:
- **Feeds** — table with mute toggle, name, URL, sector tags, last episode, errors; `+ Add feed` modal (validates RSS on save via backend)
- **Signals Watchlist** — table with mute, kind, value, associated ticker, note; `+ Add signal` modal
- **Coverage mute overrides** — list of 36 saved tickers each with per-tracker mute toggle
- **Notifications & cost** — 4 channel toggles, email digest time picker, email recipient, Telegram chat ID + Test ping button, cost-warn slider, current weekly cost gauge (`$3.84 / $15.00 ████░░░░░░░░░░░░░░ 26%`)

### 8.4 Backend endpoints

All require `require_auth()`:

```
GET    /api/media/feeds                       list
POST   /api/media/feeds                       add (validates RSS)
PATCH  /api/media/feeds/:id                   update
DELETE /api/media/feeds/:id                   remove

GET    /api/media/watchlist                   list
POST   /api/media/watchlist                   add
PATCH  /api/media/watchlist/:id               update
DELETE /api/media/watchlist/:id               remove

GET    /api/media/feed?source=&ticker=&days=  paginated firehose
GET    /api/media/episode/:id                 full episode + points + transcript preview
POST   /api/media/run-scanner                 manual trigger
GET    /api/media/scan/:id                    scan status
POST   /api/media/episode/:id/retry           manual retry

GET    /api/media/notification-prefs          current config
PUT    /api/media/notification-prefs          update
POST   /api/media/notification-prefs/push-subscribe
POST   /api/media/notification-prefs/test     test ping

GET    /api/media/clusters?week=              current/historical clusters
GET    /api/media/cost-usage?days=            cost breakdown
```

### 8.5 Deploy checklist (enforced in README)

1. Bump `index.html` `BUILD_VERSION`
2. Bump `worker.js` `BUILD_VERSION`
3. Bump `service-worker.js` `BUILD_VERSION` (driving CACHE_NAME)
4. Commit frontend files to git (`index.html`, `worker.js`, `service-worker.js`) — NEVER deploy to Cloudflare without a git commit
5. `npx wrangler deploy` (NOT `wrangler pages deploy` — custom domain binds to Worker)
6. `git push origin main` for backend (Render auto-deploys)
7. Verify `GET /api/media/feeds` returns seed
8. Check APScheduler startup log in Render for `Adding job tentatively -- it will be properly scheduled when the scheduler starts`

---

## 9. Seed Podcast List (v1 — for Tony to redline before deploy)

Twenty-five feeds targeted at Tony's 36-stock coverage (healthcare + tech + macro weighting).

**Macro / general finance**
- Odd Lots (Bloomberg)
- Bloomberg Surveillance
- Invest Like the Best (Joincolossus)
- Capital Allocators
- Grant's Current Yield Podcast
- Animal Spirits
- The Transcript (earnings-transcript-driven)
- Market Huddle

**Tech / AI / semis**
- Acquired
- Stratechery / Sharp Tech (Ben Thompson)
- a16z Podcast
- All-In Podcast
- BG2 Pod (Gurley / Gerstner)
- Big Technology Podcast (Alex Kantrowitz)
- No Priors (Guo / Friedman)
- Dwarkesh Podcast

**Healthcare / biotech**
- STAT First Opinion
- Biotech Hangout
- Endpoints Weekly
- The Long Run (Timmerman)
- MedTech Talk

**Sector / specialty**
- Business Breakdowns
- Value Hive
- Dealcast (M&A / banking)
- Hard Fork (NYT tech news)

Tony deletes/adds before M1 deploy. RSS URLs resolved at seed-insert time via a helper script.

---

## 10. Implementation Sequence

| # | Milestone | Shippable output | Verification |
|---|---|---|---|
| M1 | DB migration + feed registry + seed | 6 tables created, 25 feeds seeded, feed CRUD endpoints | `curl /api/media/feeds` returns seed |
| M2 | RSS poller (no transcription yet) | `poll_all_feeds()` populates `media_episodes` w/ titles + show notes, 7-day backfill | New rows after one tick |
| M3 | Show-notes-only extraction | Haiku extracts points from `show_notes`, `media_digest_points` populated | Points visible with sensible tags |
| M4 | Feed tab UI (read-only) | In-app firehose visible before transcription wired | Feed tab renders Oliver-style cards |
| M5 | Transcription: 4 publisher scrapers + Gemini fallback + cost tracking | `transcript` populated, cost accurate | 10 real episodes, 4 scrapers + Gemini all pass |
| M6 | Material gating + dedup + alerts | Material points fire `agent_alerts`, 7d dedup, Alerts tab shows them | 1-week live run, manual precision audit ≥70% |
| M7 | Settings UI (feeds CRUD, watchlist, mutes, notif prefs, cost gauge) | Full end-to-end management without DB access | Walkthrough |
| M8 | Clustering + notifications + cost watch | Sunday clustering, 7am email, push, Telegram, $15 warn, $50 hard cap | Staging test each channel + 1 weekly cycle |

M1–M4 ship as backend/frontend-paired PRs. After M4, a working firehose exists in-app without transcription. M5 turns it from "show notes" quality to "transcript" quality. M6–M8 complete the production system.

---

## 11. Testing Strategy

**Unit / integration** (added under backend `tests/`):
- `test_rss_poller.py` — fixtures for 3 real feeds, assert GUID dedup idempotency, 7d backfill boundary
- `test_transcription.py` — mock publisher pages per scraper, mock Gemini client, assert fallback path, assert cost accounting
- `test_extraction.py` — golden transcripts, assert point extraction within expected tag sets, material-gate correctness on coverage fixture
- `test_dedup.py` — 3 similar points across 3 episodes, assert 1 alert + `related_episodes[]` append
- `test_clustering.py` — 20 seeded points, assert sensible cluster aggregation
- `test_scheduler.py` — APScheduler jobs register, no double-fire, persist across restart (in-memory Postgres)

**E2E dogfooding** (after M6 and M8):
- Run against real feeds in staging for 7 days before enabling Tony's prod account
- Manual precision audit on last 50 material alerts — target ≥70% "would flag to PM"
- False-negative audit: 5 known-material episodes, confirm system caught them

**Prompt-quality regression**: M3/M6 include a prompt-eval harness — set of 10 transcripts + expected tags/tickers/material flag, re-run on every extraction-prompt change.

---

## 12. Rollout Plan

1. **Stealth mode** — M1–M3 deployed with tables + poller + show-notes extraction. No UI exposed. Collect 1 week of real data to tune prompts.
2. **Tony-only UI gate** — Feed tab hidden behind `app_settings.media_tab_enabled=true`. Turn on for Tony's account only after M4, before M5.
3. **Notifications off by default** — even once UI is visible, push/Telegram/email are opt-in from Settings (M7 ships them off by default).
4. **Cost alarm** — soft $15 warn + hard $50 auto-pause. Both live at M5 (first real spend).
5. **Kill switch** — `app_settings.media_scheduler_enabled=false` instantly pauses all jobs without a redeploy.

---

## 13. Cost Baseline

| Item | Weekly cost |
|---|---|
| Transcription (40% of 75 eps via Gemini @ $0.10) | ~$3 |
| Haiku extraction (75 eps) | ~$0.75 |
| Sonnet material-gate 2nd pass (~15% of points) | ~$0.50 |
| Sonnet weekly clustering | ~$1 |
| **Expected weekly** | **~$5** |
| **Soft warning threshold** | **$15** (configurable) |
| **Hard auto-pause** | **$50** (hard-coded) |

Monthly envelope ~$25 for podcast tracker alone. Scales linearly as YouTube/X/LinkedIn add.

---

## 14. Open Questions — Deferred Decisions

All of the below are intentionally punted. Flagged for revisit after 4–6 weeks of live usage:

1. **OPML import** in Settings — skip v1; `+ Add feed` is fine for 25 feeds
2. **Podcasts without audio** (Substack-style post-only) — edge case, ~5% of feeds; defer
3. **Sponsored-segment stripping** — defer until real false positives observed
4. **Share-to-Slack / Copy-summary buttons** on Feed cards — defer to polish PR
5. **Transcript purge policy** — keep forever for now (Postgres fine at 10k-episode scale)
6. **Tab consolidation** (Feed + Alerts merge) — defer pending post-launch UX feedback

---

## 15. Next Steps

1. Tony reviews this spec.
2. On approval, transition to `writing-plans` skill → implementation plan broken by M1–M8 milestones with per-file tasks.
3. Execute plan with review checkpoints at end of each milestone.
