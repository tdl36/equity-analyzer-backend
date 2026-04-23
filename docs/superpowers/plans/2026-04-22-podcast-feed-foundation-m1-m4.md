# Podcast Feed Foundation (M1–M4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working Podcast Feed tab in Charlie with RSS polling, show-notes-quality extraction, and an Oliver-style bullet-tagged firehose UI. At the end of this plan, Tony can open the Feed tab and see fresh podcast summaries from 25 tracked shows.

**Architecture:** Flask backend on Render with APScheduler persistent jobstore polling RSS feeds every 30 min and Haiku extracting bullet-tagged points from show notes. Postgres holds feeds, episodes, digest points, signals watchlist, and theme clusters. Cloudflare-hosted React-in-Babel frontend renders a new Feed tab grouped by episode with per-bullet tag chips.

**Tech Stack:** Python 3 · Flask · psycopg2 · Postgres · APScheduler (SQLAlchemy jobstore) · feedparser · Anthropic Claude Haiku · React (Babel standalone) · Tailwind · Cloudflare Worker · pytest

**Spec:** `docs/superpowers/specs/2026-04-22-charlie-media-trackers-podcast-design.md`

**Scope boundary:** This plan covers M1–M4 only. M5 (transcription), M6 (material gating), M7 (Settings UI), and M8 (clustering + notifications) are a separate plan.

---

## File Structure

**New files (backend):**
- `media_trackers/__init__.py` — package marker
- `media_trackers/poller.py` — `poll_feed()`, `poll_all_feeds()`, `backfill_feed()`
- `media_trackers/extractor.py` — `extract_from_episode()` using Haiku on show_notes
- `media_trackers/seed_feeds.py` — one-shot seeder for 25 starter feeds
- `media_trackers/prompts.py` — `EXTRACTION_PROMPT` + JSON schema
- `scheduler.py` — APScheduler setup + kill-switch
- `tests/__init__.py`
- `tests/conftest.py` — pytest fixtures (test DB, app client)
- `tests/test_db_migration.py`
- `tests/test_feeds_api.py`
- `tests/test_watchlist_api.py`
- `tests/test_poller.py`
- `tests/test_extractor.py`
- `tests/test_scheduler.py`
- `tests/test_feed_api.py`
- `tests/fixtures/odd_lots_sample.xml` — RSS fixture
- `tests/fixtures/shownotes_sample.txt` — show notes fixture
- `pytest.ini`

**Modified files (backend):**
- `requirements.txt` — add `feedparser`, `apscheduler`, `sqlalchemy`, `pytest`, `pytest-flask`, `responses`
- `app_v3.py` — add 6 tables to `init_db()`, register scheduler on boot, add 9 new endpoints

**Modified files (frontend):**
- `/Users/tonydlee/Library/CloudStorage/GoogleDrive-tonydlee@gmail.com/My Drive/ea v58 NEW/charlie-deployment/index.html` — add Feed tab, bump `BUILD_VERSION`
- `/Users/tonydlee/Library/CloudStorage/GoogleDrive-tonydlee@gmail.com/My Drive/ea v58 NEW/charlie-deployment/worker.js` — bump `BUILD_VERSION`
- `/Users/tonydlee/Library/CloudStorage/GoogleDrive-tonydlee@gmail.com/My Drive/ea v58 NEW/charlie-deployment/service-worker.js` — bump `BUILD_VERSION`

**Boundaries & responsibilities:**
- `scheduler.py` knows only about APScheduler wiring — it imports job bodies from `media_trackers/*` but contains no domain logic.
- `media_trackers/poller.py` handles RSS ingestion only — does not touch LLMs.
- `media_trackers/extractor.py` handles LLM extraction only — does not touch RSS.
- `media_trackers/prompts.py` is the single home for the extraction prompt and JSON schema.
- `app_v3.py` routes are thin — they call into `media_trackers/*` and serialize to JSON.

---

## Conventions for every task

- **Branch:** all work on `feature/media-trackers-m1-m4` (create at Task 1 if not exists).
- **Test DB:** tests use `TEST_DATABASE_URL` pointing at a local Postgres (e.g. `postgres://localhost:5432/charlie_test`). Fixtures reset tables between tests.
- **Anthropic key in tests:** never call the real API — use `responses` library to mock HTTP, or monkeypatch `call_llm`.
- **Commit message format:** `<prefix>(<scope>): <subject>` where prefix ∈ {feat, fix, test, chore, docs}. Include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer.
- **Run tests after each task:** `pytest -x -v` before committing.

---

## Task 1: Bootstrap test infrastructure

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies to `requirements.txt`**

Append the following lines to `/Users/tonydlee/Projects/equity-analyzer-backend/requirements.txt`:

```
feedparser>=6.0,<7
apscheduler>=3.10,<4
sqlalchemy>=2.0,<3
pytest>=8.0,<9
pytest-flask>=1.3,<2
responses>=0.25,<1
```

- [ ] **Step 2: Install in a venv**

```bash
cd /Users/tonydlee/Projects/equity-analyzer-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install cleanly. If `psycopg2-binary` fails on macOS, `brew install postgresql` first.

- [ ] **Step 3: Create `pytest.ini`**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -ra --strict-markers
markers =
    integration: requires live Postgres via TEST_DATABASE_URL
```

- [ ] **Step 4: Create `tests/__init__.py`** (empty file)

```bash
mkdir -p tests/fixtures
touch tests/__init__.py
```

- [ ] **Step 5: Create `tests/conftest.py`**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/conftest.py`

```python
"""Shared pytest fixtures.

Tests require TEST_DATABASE_URL set to a Postgres instance that the test
process can freely create/drop tables in. All tables are truncated between
tests via the `clean_db` fixture.
"""
import os
import pytest

# Point init_db() at the test database for the duration of this process.
os.environ.setdefault('DATABASE_URL', os.environ.get('TEST_DATABASE_URL', 'postgres://localhost:5432/charlie_test'))

import app_v3  # noqa: E402  — import after env set


@pytest.fixture(scope='session', autouse=True)
def _init_schema():
    app_v3.init_db()
    yield


@pytest.fixture
def clean_db():
    """Truncate all tracker-related tables between tests."""
    with app_v3.get_db(commit=True) as (_conn, cur):
        cur.execute("""
            TRUNCATE media_digest_points, media_episodes, media_feeds,
                     signals_watchlist, media_theme_clusters, notification_prefs,
                     agent_alerts RESTART IDENTITY CASCADE
        """)
    yield


@pytest.fixture
def client(clean_db):
    app_v3.app.config['TESTING'] = True
    # Auth: the global @app.before_request hook require_auth() checks
    # CHARLIE_PASSWORD and CHARLIE_API_KEY env vars. When both are empty,
    # it falls into dev-mode "allow all" (app_v3.py:119-120). conftest sets
    # these to '' in the test env so all requests bypass auth naturally.
    with app_v3.app.test_client() as c:
        yield c
```

Also add to the top of `conftest.py` (before `import app_v3`):

```python
os.environ.setdefault('CHARLIE_PASSWORD', '')
os.environ.setdefault('CHARLIE_API_KEY', '')
```

- [ ] **Step 6: Create a local test DB**

```bash
createdb charlie_test 2>&1 || echo "already exists"
psql charlie_test -c "SELECT version();"
```

Expected: psql returns a PostgreSQL version string.

- [ ] **Step 7: Run pytest to verify empty-suite success**

```bash
cd /Users/tonydlee/Projects/equity-analyzer-backend && .venv/bin/pytest -v
```

Expected: `no tests ran in 0.XXs` (success — confirms conftest imports without errors).

- [ ] **Step 8: Commit**

```bash
git checkout -b feature/media-trackers-m1-m4
git add requirements.txt pytest.ini tests/
git commit -m "chore(media-trackers): bootstrap pytest + test fixtures

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: DB migration — add 6 tracker tables

**Files:**
- Modify: `app_v3.py` (extend `init_db()` function ~line 680–1433)
- Create: `tests/test_db_migration.py`

- [ ] **Step 1: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_db_migration.py`

```python
"""Verify all media-tracker tables created by init_db()."""
import app_v3

EXPECTED_TABLES = [
    'media_feeds',
    'media_episodes',
    'media_digest_points',
    'signals_watchlist',
    'media_theme_clusters',
    'notification_prefs',
]


def test_media_tables_exist(clean_db):
    with app_v3.get_db() as (_conn, cur):
        cur.execute("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename = ANY(%s)
        """, (EXPECTED_TABLES,))
        found = {r['tablename'] for r in cur.fetchall()}
    missing = set(EXPECTED_TABLES) - found
    assert not missing, f"Missing tables: {missing}"


def test_media_feeds_has_required_columns(clean_db):
    with app_v3.get_db() as (_conn, cur):
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'media_feeds'
        """)
        cols = {r['column_name'] for r in cur.fetchall()}
    required = {'id', 'source_type', 'name', 'feed_url', 'sector_tags',
                'muted', 'last_polled_at', 'last_episode_at',
                'poll_interval_min', 'error_count', 'last_error', 'created_at'}
    assert required <= cols, f"Missing columns: {required - cols}"
```

- [ ] **Step 2: Run the test — expect failure**

```bash
.venv/bin/pytest tests/test_db_migration.py -v
```

Expected: both tests fail with `Missing tables`.

- [ ] **Step 3: Extend `init_db()` with the 6 new tables**

Open `/Users/tonydlee/Projects/equity-analyzer-backend/app_v3.py`. Find the existing `agent_alerts` CREATE TABLE at line ~1420. **After** the existing `CREATE INDEX idx_agent_alerts_created` line (~1431), **before** `print("Database tables initialized")`, insert:

```python
            cur.execute('''
                CREATE TABLE IF NOT EXISTS media_feeds (
                    id                VARCHAR(100) PRIMARY KEY,
                    source_type       VARCHAR(20) NOT NULL,
                    name              TEXT NOT NULL,
                    feed_url          TEXT NOT NULL,
                    sector_tags       TEXT[] DEFAULT '{}',
                    muted             BOOLEAN DEFAULT FALSE,
                    last_polled_at    TIMESTAMP,
                    last_episode_at   TIMESTAMP,
                    poll_interval_min INT DEFAULT 30,
                    error_count       INT DEFAULT 0,
                    last_error        TEXT,
                    created_at        TIMESTAMP DEFAULT NOW()
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS media_episodes (
                    id                VARCHAR(100) PRIMARY KEY,
                    feed_id           VARCHAR(100) REFERENCES media_feeds(id) ON DELETE CASCADE,
                    guid              TEXT NOT NULL,
                    title             TEXT NOT NULL,
                    published_at      TIMESTAMP,
                    audio_url         TEXT,
                    source_url        TEXT,
                    show_notes        TEXT,
                    duration_sec      INT,
                    transcript        TEXT,
                    transcript_source VARCHAR(20),
                    status            VARCHAR(20) DEFAULT 'new',
                    error_message     TEXT,
                    cost_usd          NUMERIC(10,4) DEFAULT 0,
                    created_at        TIMESTAMP DEFAULT NOW(),
                    UNIQUE(feed_id, guid)
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS media_digest_points (
                    id            VARCHAR(100) PRIMARY KEY,
                    episode_id    VARCHAR(100) REFERENCES media_episodes(id) ON DELETE CASCADE,
                    point_order   INT NOT NULL,
                    text          TEXT NOT NULL,
                    tickers       TEXT[] DEFAULT '{}',
                    sector_tags   TEXT[] DEFAULT '{}',
                    theme_tags    TEXT[] DEFAULT '{}',
                    timestamp_sec INT,
                    material      BOOLEAN DEFAULT FALSE,
                    cluster_id    VARCHAR(100),
                    created_at    TIMESTAMP DEFAULT NOW()
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS signals_watchlist (
                    id                VARCHAR(100) PRIMARY KEY,
                    kind              VARCHAR(20) NOT NULL,
                    value             TEXT NOT NULL,
                    associated_ticker VARCHAR(20),
                    muted             BOOLEAN DEFAULT FALSE,
                    note              TEXT,
                    created_at        TIMESTAMP DEFAULT NOW(),
                    UNIQUE(kind, value)
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS media_theme_clusters (
                    id              VARCHAR(100) PRIMARY KEY,
                    theme           TEXT NOT NULL,
                    summary         TEXT,
                    point_ids       TEXT[] DEFAULT '{}',
                    primary_tickers TEXT[] DEFAULT '{}',
                    week_start      DATE NOT NULL,
                    created_at      TIMESTAMP DEFAULT NOW()
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS notification_prefs (
                    key   VARCHAR(50) PRIMARY KEY,
                    value JSONB
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_episodes_status ON media_episodes(status)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_episodes_feed_published ON media_episodes(feed_id, published_at DESC)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_points_episode ON media_digest_points(episode_id, point_order)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_points_tickers_gin ON media_digest_points USING GIN(tickers)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_points_material ON media_digest_points(material, created_at DESC)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_points_cluster ON media_digest_points(cluster_id)')
```

- [ ] **Step 4: Run the test — expect pass**

```bash
.venv/bin/pytest tests/test_db_migration.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add app_v3.py tests/test_db_migration.py
git commit -m "feat(media-trackers): add 6 tracker tables to init_db

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Seed 25 podcast feeds

**Files:**
- Create: `media_trackers/__init__.py`
- Create: `media_trackers/seed_feeds.py`
- Create: `tests/test_seed_feeds.py`

- [ ] **Step 1: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_seed_feeds.py`

```python
"""Verify seed script inserts 25 feeds idempotently."""
import app_v3
from media_trackers.seed_feeds import seed, SEED_FEEDS


def test_seed_inserts_25_feeds(clean_db):
    seed()
    with app_v3.get_db() as (_conn, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_feeds WHERE source_type='podcast'")
        assert cur.fetchone()['n'] == len(SEED_FEEDS)


def test_seed_is_idempotent(clean_db):
    seed()
    seed()  # second run should not duplicate
    with app_v3.get_db() as (_conn, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_feeds WHERE source_type='podcast'")
        assert cur.fetchone()['n'] == len(SEED_FEEDS)


def test_all_feeds_have_rss_urls():
    for f in SEED_FEEDS:
        assert f['feed_url'].startswith(('http://', 'https://')), f
        assert f['name'], f
        assert 'sector_tags' in f
```

- [ ] **Step 2: Run the test — expect ImportError**

```bash
.venv/bin/pytest tests/test_seed_feeds.py -v
```

Expected: `ModuleNotFoundError: No module named 'media_trackers'`.

- [ ] **Step 3: Create package marker**

```bash
mkdir -p /Users/tonydlee/Projects/equity-analyzer-backend/media_trackers
touch /Users/tonydlee/Projects/equity-analyzer-backend/media_trackers/__init__.py
```

- [ ] **Step 4: Create seed script**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/media_trackers/seed_feeds.py`

```python
"""Seed the media_feeds table with Tony's default 25 podcasts.

Idempotent via ON CONFLICT DO NOTHING keyed on feed_url.
Run: python -m media_trackers.seed_feeds
"""
import uuid
import app_v3


SEED_FEEDS = [
    # Macro / general finance
    {'name': 'Odd Lots', 'feed_url': 'https://feeds.bloomberg.fm/BLM7293307739', 'sector_tags': ['macro', 'markets']},
    {'name': 'Bloomberg Surveillance', 'feed_url': 'https://feeds.bloomberg.fm/BLM4689043924', 'sector_tags': ['macro', 'markets']},
    {'name': 'Invest Like the Best', 'feed_url': 'https://feeds.simplecast.com/JGE3yC0V', 'sector_tags': ['investing']},
    {'name': 'Capital Allocators', 'feed_url': 'https://capitalallocators.libsyn.com/rss', 'sector_tags': ['investing']},
    {'name': "Grant's Current Yield Podcast", 'feed_url': 'https://feeds.megaphone.fm/grantscurrentyield', 'sector_tags': ['macro', 'credit']},
    {'name': 'Animal Spirits', 'feed_url': 'https://feeds.megaphone.fm/animalspirits', 'sector_tags': ['investing', 'macro']},
    {'name': 'The Transcript', 'feed_url': 'https://feeds.simplecast.com/UVgAJtOV', 'sector_tags': ['earnings']},
    {'name': 'Market Huddle', 'feed_url': 'https://feeds.megaphone.fm/markethuddle', 'sector_tags': ['macro', 'markets']},
    # Tech / AI / semis
    {'name': 'Acquired', 'feed_url': 'https://feeds.transistor.fm/acquired', 'sector_tags': ['tech', 'investing']},
    {'name': 'Sharp Tech with Ben Thompson', 'feed_url': 'https://sharptech.fm/feed/podcast', 'sector_tags': ['tech']},
    {'name': 'a16z Podcast', 'feed_url': 'https://feeds.simplecast.com/JGE3yC0V', 'sector_tags': ['tech', 'vc']},
    {'name': 'All-In Podcast', 'feed_url': 'https://allinchamathjason.libsyn.com/rss', 'sector_tags': ['tech', 'macro']},
    {'name': 'BG2 Pod', 'feed_url': 'https://feeds.megaphone.fm/bg2pod', 'sector_tags': ['tech', 'ai']},
    {'name': 'Big Technology Podcast', 'feed_url': 'https://feeds.megaphone.fm/bigtechnology', 'sector_tags': ['tech']},
    {'name': 'No Priors', 'feed_url': 'https://feeds.megaphone.fm/nopriors', 'sector_tags': ['tech', 'ai']},
    {'name': 'Dwarkesh Podcast', 'feed_url': 'https://feeds.transistor.fm/dwarkesh', 'sector_tags': ['tech', 'ai']},
    # Healthcare / biotech
    {'name': 'STAT First Opinion', 'feed_url': 'https://feeds.simplecast.com/5OGUyVKf', 'sector_tags': ['healthcare']},
    {'name': 'Biotech Hangout', 'feed_url': 'https://biotechhangout.libsyn.com/rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'Endpoints Weekly', 'feed_url': 'https://feeds.buzzsprout.com/1791687.rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'The Long Run', 'feed_url': 'https://timmermanreport.libsyn.com/rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'MedTech Talk', 'feed_url': 'https://feeds.buzzsprout.com/1955770.rss', 'sector_tags': ['healthcare', 'medtech']},
    # Sector / specialty
    {'name': 'Business Breakdowns', 'feed_url': 'https://feeds.simplecast.com/IZe51ENa', 'sector_tags': ['investing', 'sector']},
    {'name': 'Value Hive', 'feed_url': 'https://feeds.simplecast.com/ynrv7Y_H', 'sector_tags': ['investing']},
    {'name': 'Dealcast', 'feed_url': 'https://feeds.megaphone.fm/dealcast', 'sector_tags': ['banking', 'ma']},
    {'name': 'Hard Fork', 'feed_url': 'https://feeds.simplecast.com/l2i9YnTd', 'sector_tags': ['tech']},
]


def seed():
    """Insert default feeds. Idempotent: uses ON CONFLICT on feed_url.

    Note: feed_url uniqueness is enforced by a partial unique index created
    here on first run (not in init_db — optional seed-only constraint).
    """
    with app_v3.get_db(commit=True) as (_conn, cur):
        cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_media_feeds_url_unique ON media_feeds(feed_url)')
        for f in SEED_FEEDS:
            cur.execute('''
                INSERT INTO media_feeds (id, source_type, name, feed_url, sector_tags, poll_interval_min)
                VALUES (%s, 'podcast', %s, %s, %s, 30)
                ON CONFLICT (feed_url) DO NOTHING
            ''', (str(uuid.uuid4()), f['name'], f['feed_url'], f['sector_tags']))


if __name__ == '__main__':
    seed()
    print(f"Seeded up to {len(SEED_FEEDS)} feeds.")
```

- [ ] **Step 5: Run the test — expect pass**

```bash
.venv/bin/pytest tests/test_seed_feeds.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add media_trackers/ tests/test_seed_feeds.py
git commit -m "feat(media-trackers): seed 25 default podcast feeds

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

> **Note to executor:** The RSS URLs are best-effort from public pages. Tony will redline before first prod deploy. If any URL 404s during the Task 6 poller run, note it and `UPDATE media_feeds SET feed_url = '...' WHERE name = '...'` in a follow-up commit.

---

## Task 4: Feeds CRUD endpoints

**Files:**
- Modify: `app_v3.py` (add endpoints near the existing `/api/alerts` section, ~line 265)
- Create: `tests/test_feeds_api.py`

- [ ] **Step 1: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_feeds_api.py`

```python
"""Tests for /api/media/feeds CRUD endpoints."""
import json


def test_get_feeds_empty(client):
    resp = client.get('/api/media/feeds')
    assert resp.status_code == 200
    assert resp.get_json() == {'feeds': [], 'total': 0}


def test_create_feed(client):
    payload = {
        'name': 'Odd Lots',
        'feedUrl': 'https://feeds.bloomberg.fm/BLM7293307739',
        'sourceType': 'podcast',
        'sectorTags': ['macro'],
    }
    resp = client.post('/api/media/feeds', json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['feed']['name'] == 'Odd Lots'
    assert body['feed']['sectorTags'] == ['macro']
    assert body['feed']['muted'] is False


def test_create_feed_requires_name_and_url(client):
    resp = client.post('/api/media/feeds', json={'name': 'X'})
    assert resp.status_code == 400


def test_patch_feed_toggles_mute(client):
    resp = client.post('/api/media/feeds', json={
        'name': 'Odd Lots', 'feedUrl': 'https://example.com/rss', 'sourceType': 'podcast',
    })
    feed_id = resp.get_json()['feed']['id']
    resp = client.patch(f'/api/media/feeds/{feed_id}', json={'muted': True})
    assert resp.status_code == 200
    assert resp.get_json()['feed']['muted'] is True


def test_delete_feed(client):
    resp = client.post('/api/media/feeds', json={
        'name': 'Odd Lots', 'feedUrl': 'https://example.com/rss', 'sourceType': 'podcast',
    })
    feed_id = resp.get_json()['feed']['id']
    resp = client.delete(f'/api/media/feeds/{feed_id}')
    assert resp.status_code == 200
    resp = client.get('/api/media/feeds')
    assert resp.get_json()['total'] == 0
```

- [ ] **Step 2: Run the test — expect 404s**

```bash
.venv/bin/pytest tests/test_feeds_api.py -v
```

Expected: all 5 tests fail with 404 (routes don't exist yet).

- [ ] **Step 3: Add endpoints to `app_v3.py`**

Open `/Users/tonydlee/Projects/equity-analyzer-backend/app_v3.py`. After the last alerts endpoint (`def alert_count()` at line ~264), **before** the `# MULTI-MODEL LLM FALLBACK` comment block at line ~267, insert:

```python
# ============================================
# MEDIA TRACKER FEEDS — CRUD
# ============================================

def _row_to_feed(r):
    return {
        'id': r['id'],
        'sourceType': r['source_type'],
        'name': r['name'],
        'feedUrl': r['feed_url'],
        'sectorTags': r['sector_tags'] or [],
        'muted': r['muted'],
        'lastPolledAt': r['last_polled_at'].isoformat() if r['last_polled_at'] else None,
        'lastEpisodeAt': r['last_episode_at'].isoformat() if r['last_episode_at'] else None,
        'pollIntervalMin': r['poll_interval_min'],
        'errorCount': r['error_count'],
        'lastError': r['last_error'],
        'createdAt': r['created_at'].isoformat() if r['created_at'] else None,
    }


@app.route('/api/media/feeds', methods=['GET'])
def media_feeds_list():
with get_db() as (_c, cur):
        cur.execute("SELECT * FROM media_feeds ORDER BY name ASC")
        rows = cur.fetchall()
    feeds = [_row_to_feed(r) for r in rows]
    return jsonify({'feeds': feeds, 'total': len(feeds)})


@app.route('/api/media/feeds', methods=['POST'])
def media_feeds_create():
data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    feed_url = (data.get('feedUrl') or '').strip()
    source_type = (data.get('sourceType') or 'podcast').strip()
    sector_tags = data.get('sectorTags') or []
    poll_interval = int(data.get('pollIntervalMin') or 30)
    if not name or not feed_url:
        return jsonify({'error': 'name and feedUrl required'}), 400
    feed_id = str(uuid.uuid4())
    with get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url, sector_tags, poll_interval_min)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        ''', (feed_id, source_type, name, feed_url, sector_tags, poll_interval))
        row = cur.fetchone()
    return jsonify({'feed': _row_to_feed(row)})


@app.route('/api/media/feeds/<feed_id>', methods=['PATCH'])
def media_feeds_update(feed_id):
data = request.get_json() or {}
    fields, values = [], []
    for k, col in [('muted', 'muted'), ('name', 'name'), ('feedUrl', 'feed_url'),
                   ('sectorTags', 'sector_tags'), ('pollIntervalMin', 'poll_interval_min')]:
        if k in data:
            fields.append(f"{col} = %s")
            values.append(data[k])
    if not fields:
        return jsonify({'error': 'no updatable fields'}), 400
    values.append(feed_id)
    with get_db(commit=True) as (_c, cur):
        cur.execute(f"UPDATE media_feeds SET {', '.join(fields)} WHERE id = %s RETURNING *", values)
        row = cur.fetchone()
    if not row:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'feed': _row_to_feed(row)})


@app.route('/api/media/feeds/<feed_id>', methods=['DELETE'])
def media_feeds_delete(feed_id):
with get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM media_feeds WHERE id = %s", (feed_id,))
    return jsonify({'success': True})
```

- [ ] **Step 4: Run the test — expect pass**

```bash
.venv/bin/pytest tests/test_feeds_api.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app_v3.py tests/test_feeds_api.py
git commit -m "feat(media-trackers): /api/media/feeds CRUD endpoints

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Signals watchlist CRUD endpoints

**Files:**
- Modify: `app_v3.py` (add endpoints below the feeds block from Task 4)
- Create: `tests/test_watchlist_api.py`

- [ ] **Step 1: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_watchlist_api.py`

```python
def test_watchlist_empty(client):
    resp = client.get('/api/media/watchlist')
    assert resp.status_code == 200
    assert resp.get_json() == {'signals': [], 'total': 0}


def test_watchlist_create_ticker(client):
    resp = client.post('/api/media/watchlist', json={
        'kind': 'ticker', 'value': 'ISRG', 'associatedTicker': 'BSX', 'note': 'Competitor',
    })
    assert resp.status_code == 200
    sig = resp.get_json()['signal']
    assert sig['kind'] == 'ticker'
    assert sig['value'] == 'ISRG'
    assert sig['associatedTicker'] == 'BSX'


def test_watchlist_duplicate_returns_409(client):
    client.post('/api/media/watchlist', json={'kind': 'keyword', 'value': 'GLP-1'})
    resp = client.post('/api/media/watchlist', json={'kind': 'keyword', 'value': 'GLP-1'})
    assert resp.status_code == 409


def test_watchlist_mute(client):
    r = client.post('/api/media/watchlist', json={'kind': 'exec', 'value': 'Jensen Huang'})
    sid = r.get_json()['signal']['id']
    resp = client.patch(f'/api/media/watchlist/{sid}', json={'muted': True})
    assert resp.get_json()['signal']['muted'] is True
```

- [ ] **Step 2: Run the test — expect failure**

```bash
.venv/bin/pytest tests/test_watchlist_api.py -v
```

Expected: all fail with 404.

- [ ] **Step 3: Add endpoints after the feeds block in `app_v3.py`**

Insert after the last feeds endpoint from Task 4:

```python
# ============================================
# MEDIA TRACKER — SIGNALS WATCHLIST
# ============================================

def _row_to_signal(r):
    return {
        'id': r['id'],
        'kind': r['kind'],
        'value': r['value'],
        'associatedTicker': r['associated_ticker'],
        'muted': r['muted'],
        'note': r['note'],
        'createdAt': r['created_at'].isoformat() if r['created_at'] else None,
    }


@app.route('/api/media/watchlist', methods=['GET'])
def media_watchlist_list():
with get_db() as (_c, cur):
        cur.execute("SELECT * FROM signals_watchlist ORDER BY kind, value")
        rows = cur.fetchall()
    return jsonify({'signals': [_row_to_signal(r) for r in rows], 'total': len(rows)})


@app.route('/api/media/watchlist', methods=['POST'])
def media_watchlist_create():
data = request.get_json() or {}
    kind = (data.get('kind') or '').strip()
    value = (data.get('value') or '').strip()
    if kind not in ('ticker', 'keyword', 'exec') or not value:
        return jsonify({'error': 'kind must be ticker|keyword|exec; value required'}), 400
    sid = str(uuid.uuid4())
    try:
        with get_db(commit=True) as (_c, cur):
            cur.execute('''
                INSERT INTO signals_watchlist (id, kind, value, associated_ticker, note)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
            ''', (sid, kind, value, data.get('associatedTicker'), data.get('note')))
            row = cur.fetchone()
    except Exception as e:
        if 'duplicate key' in str(e).lower() or 'unique' in str(e).lower():
            return jsonify({'error': 'already exists'}), 409
        raise
    return jsonify({'signal': _row_to_signal(row)})


@app.route('/api/media/watchlist/<signal_id>', methods=['PATCH'])
def media_watchlist_update(signal_id):
data = request.get_json() or {}
    fields, values = [], []
    for k, col in [('muted', 'muted'), ('note', 'note'), ('associatedTicker', 'associated_ticker')]:
        if k in data:
            fields.append(f"{col} = %s")
            values.append(data[k])
    if not fields:
        return jsonify({'error': 'no updatable fields'}), 400
    values.append(signal_id)
    with get_db(commit=True) as (_c, cur):
        cur.execute(f"UPDATE signals_watchlist SET {', '.join(fields)} WHERE id = %s RETURNING *", values)
        row = cur.fetchone()
    if not row:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'signal': _row_to_signal(row)})


@app.route('/api/media/watchlist/<signal_id>', methods=['DELETE'])
def media_watchlist_delete(signal_id):
with get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM signals_watchlist WHERE id = %s", (signal_id,))
    return jsonify({'success': True})
```

- [ ] **Step 4: Run the test — expect pass**

```bash
.venv/bin/pytest tests/test_watchlist_api.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app_v3.py tests/test_watchlist_api.py
git commit -m "feat(media-trackers): /api/media/watchlist CRUD

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: RSS poller (M2)

**Files:**
- Create: `media_trackers/poller.py`
- Create: `tests/fixtures/odd_lots_sample.xml`
- Create: `tests/test_poller.py`

- [ ] **Step 1: Create RSS fixture**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/fixtures/odd_lots_sample.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Odd Lots</title>
  <item>
    <title>The GLP-1 Supply Crunch</title>
    <guid isPermaLink="false">odd-lots-2026-04-20</guid>
    <pubDate>Mon, 20 Apr 2026 10:00:00 GMT</pubDate>
    <link>https://www.bloomberg.com/podcasts/oddlots/glp1</link>
    <enclosure url="https://cdn.example.com/odd-lots-20260420.mp3" type="audio/mpeg"/>
    <description>Tracy and Joe talk to a Lilly exec about Zepbound supply.</description>
    <itunes:duration xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">3720</itunes:duration>
  </item>
  <item>
    <title>Fed Minutes Deep Dive</title>
    <guid>odd-lots-2026-04-18</guid>
    <pubDate>Fri, 18 Apr 2026 10:00:00 GMT</pubDate>
    <link>https://www.bloomberg.com/podcasts/oddlots/fed</link>
    <enclosure url="https://cdn.example.com/odd-lots-20260418.mp3" type="audio/mpeg"/>
    <description>Unpacking the April FOMC minutes.</description>
  </item>
</channel>
</rss>
```

- [ ] **Step 2: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_poller.py`

```python
"""Tests for the RSS poller."""
import os
from pathlib import Path
import pytest
import responses
import app_v3
from media_trackers import poller

FIXTURE = Path(__file__).parent / 'fixtures' / 'odd_lots_sample.xml'


def _create_feed(url='https://fake.example/rss', name='Odd Lots'):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('feed-1', 'podcast', %s, %s) RETURNING id
        ''', (name, url))
        return cur.fetchone()['id']


@responses.activate
def test_poll_feed_inserts_episodes(clean_db):
    feed_id = _create_feed()
    responses.add(responses.GET, 'https://fake.example/rss',
                  body=FIXTURE.read_text(), status=200,
                  content_type='application/rss+xml')
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT guid, title FROM media_episodes WHERE feed_id=%s ORDER BY guid", (feed_id,))
        rows = cur.fetchall()
    guids = {r['guid'] for r in rows}
    assert guids == {'odd-lots-2026-04-20', 'odd-lots-2026-04-18'}


@responses.activate
def test_poll_feed_is_idempotent(clean_db):
    feed_id = _create_feed()
    responses.add(responses.GET, 'https://fake.example/rss',
                  body=FIXTURE.read_text(), status=200)
    poller.poll_feed(feed_id)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_episodes WHERE feed_id=%s", (feed_id,))
        assert cur.fetchone()['n'] == 2


@responses.activate
def test_poll_feed_records_error_on_404(clean_db):
    feed_id = _create_feed()
    responses.add(responses.GET, 'https://fake.example/rss', status=404)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT error_count, last_error FROM media_feeds WHERE id=%s", (feed_id,))
        row = cur.fetchone()
    assert row['error_count'] == 1
    assert row['last_error']


@responses.activate
def test_poll_feed_auto_mutes_after_5_errors(clean_db):
    feed_id = _create_feed()
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET error_count=4 WHERE id=%s", (feed_id,))
    responses.add(responses.GET, 'https://fake.example/rss', status=500)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT muted FROM media_feeds WHERE id=%s", (feed_id,))
        assert cur.fetchone()['muted'] is True


@responses.activate
def test_backfill_feed_skips_episodes_older_than_7d(clean_db):
    feed_id = _create_feed()
    # Both fixture episodes are from 2026-04-18 and 2026-04-20 — in past relative to test run.
    # We simulate this by stamping last_episode_at to 2026-04-19 so poll only picks 04-20.
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET last_episode_at='2026-04-19'::timestamp WHERE id=%s", (feed_id,))
    responses.add(responses.GET, 'https://fake.example/rss', body=FIXTURE.read_text(), status=200)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT guid FROM media_episodes WHERE feed_id=%s", (feed_id,))
        guids = {r['guid'] for r in cur.fetchall()}
    assert guids == {'odd-lots-2026-04-20'}


def test_poll_all_feeds_iterates(clean_db, monkeypatch):
    _create_feed(url='https://a.example/rss', name='A')
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('feed-2', 'podcast', 'B', 'https://b.example/rss')
        ''')
    calls = []
    monkeypatch.setattr(poller, 'poll_feed', lambda fid: calls.append(fid))
    poller.poll_all_feeds()
    assert set(calls) == {'feed-1', 'feed-2'}


def test_poll_all_feeds_skips_muted(clean_db, monkeypatch):
    _create_feed(url='https://a.example/rss', name='A')
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET muted=TRUE WHERE id='feed-1'")
    calls = []
    monkeypatch.setattr(poller, 'poll_feed', lambda fid: calls.append(fid))
    poller.poll_all_feeds()
    assert calls == []
```

- [ ] **Step 3: Run the test — expect ImportError**

```bash
.venv/bin/pytest tests/test_poller.py -v
```

Expected: `cannot import name 'poller' from 'media_trackers'`.

- [ ] **Step 4: Create poller module**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/media_trackers/poller.py`

```python
"""RSS feed poller.

poll_feed(feed_id) — fetches one feed, upserts episodes, updates feed metadata.
poll_all_feeds()   — iterates non-muted feeds due for polling.
backfill_feed(id)  — first-time add helper; poll once ignoring backoff window.
"""
import uuid
import time
import requests
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import app_v3


MAX_ERRORS_BEFORE_MUTE = 5
BACKFILL_DAYS = 7
HTTP_TIMEOUT_SEC = 30


def _parse_published(entry):
    for attr in ('published', 'updated'):
        v = entry.get(attr)
        if v:
            try:
                return parsedate_to_datetime(v).astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                pass
    return None


def _audio_url(entry):
    for enc in entry.get('enclosures') or []:
        if enc.get('type', '').startswith('audio'):
            return enc.get('href') or enc.get('url')
    return None


def _duration_sec(entry):
    v = entry.get('itunes_duration')
    if not v: return None
    try:
        parts = str(v).split(':')
        if len(parts) == 1: return int(parts[0])
        if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def _record_error(feed_id, message):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            UPDATE media_feeds
               SET error_count = error_count + 1,
                   last_error  = %s,
                   last_polled_at = NOW(),
                   muted = CASE WHEN error_count + 1 >= %s THEN TRUE ELSE muted END
             WHERE id = %s
        ''', (message, MAX_ERRORS_BEFORE_MUTE, feed_id))


def _reset_error(feed_id, max_pub):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            UPDATE media_feeds
               SET error_count = 0, last_error = NULL,
                   last_polled_at = NOW(),
                   last_episode_at = GREATEST(COALESCE(last_episode_at, '1970-01-01'::timestamp), %s)
             WHERE id = %s
        ''', (max_pub, feed_id))


def poll_feed(feed_id: str) -> None:
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT * FROM media_feeds WHERE id=%s", (feed_id,))
        feed = cur.fetchone()
    if not feed:
        return

    try:
        resp = requests.get(feed['feed_url'], timeout=HTTP_TIMEOUT_SEC,
                            headers={'User-Agent': 'Charlie/1.0 (+https://charlie.tonydlee.com)'})
        resp.raise_for_status()
    except Exception as e:
        _record_error(feed_id, f"HTTP: {e}")
        return

    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        _record_error(feed_id, f"parse: {parsed.bozo_exception}")
        return

    # Backfill cutoff: new feed → 7 days ago; existing → last_episode_at.
    if feed['last_episode_at'] is None:
        cutoff = datetime.utcnow() - _timedelta_days(BACKFILL_DAYS)
    else:
        cutoff = feed['last_episode_at']

    max_pub = cutoff
    inserted = 0
    with app_v3.get_db(commit=True) as (_c, cur):
        for entry in parsed.entries:
            guid = entry.get('id') or entry.get('guid') or entry.get('link')
            if not guid:
                continue
            pub = _parse_published(entry)
            if pub and pub <= cutoff:
                continue
            if pub and pub > max_pub:
                max_pub = pub
            cur.execute('''
                INSERT INTO media_episodes
                    (id, feed_id, guid, title, published_at, audio_url, source_url, show_notes, duration_sec, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'new')
                ON CONFLICT (feed_id, guid) DO NOTHING
            ''', (
                str(uuid.uuid4()), feed_id, guid, entry.get('title') or '(untitled)',
                pub, _audio_url(entry), entry.get('link'),
                entry.get('summary') or entry.get('description'),
                _duration_sec(entry),
            ))
            if cur.rowcount:
                inserted += 1

    _reset_error(feed_id, max_pub)


def _timedelta_days(d):
    from datetime import timedelta
    return timedelta(days=d)


def poll_all_feeds() -> None:
    """Iterate non-muted feeds that are due for polling."""
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM media_feeds
             WHERE muted = FALSE
               AND (last_polled_at IS NULL
                    OR last_polled_at < NOW() - (poll_interval_min || ' minutes')::interval)
        ''')
        ids = [r['id'] for r in cur.fetchall()]
    for feed_id in ids:
        try:
            poll_feed(feed_id)
        except Exception as e:
            _record_error(feed_id, f"poll: {e}")
        time.sleep(0.25)  # be nice to publishers


def backfill_feed(feed_id: str) -> None:
    """Force-poll a newly added feed, bypassing the normal backoff window."""
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET last_polled_at = NULL WHERE id = %s", (feed_id,))
    poll_feed(feed_id)
```

- [ ] **Step 5: Run the test — expect pass**

```bash
.venv/bin/pytest tests/test_poller.py -v
```

Expected: 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add media_trackers/poller.py tests/fixtures/odd_lots_sample.xml tests/test_poller.py
git commit -m "feat(media-trackers): RSS poller with backfill + error handling

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: APScheduler setup + kill switch

**Files:**
- Create: `scheduler.py`
- Modify: `app_v3.py` (import + boot registration)
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_scheduler.py`

```python
import app_v3
import scheduler


def test_scheduler_registers_expected_jobs(clean_db):
    sched = scheduler.build_scheduler(use_memory_jobstore=True)
    job_ids = {j.id for j in sched.get_jobs()}
    assert {'feed_poller', 'extract_worker'} <= job_ids


def test_kill_switch_skips_job_body(clean_db, monkeypatch):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO app_settings (key, value) VALUES ('media_scheduler_enabled', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        ''', ('false',))
    called = []
    monkeypatch.setattr('media_trackers.poller.poll_all_feeds', lambda: called.append(1))
    scheduler.run_feed_poller()
    assert called == []  # skipped because kill switch is off
```

- [ ] **Step 2: Run the test — expect ImportError**

```bash
.venv/bin/pytest tests/test_scheduler.py -v
```

Expected: `No module named 'scheduler'`.

- [ ] **Step 3: Create scheduler module**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/scheduler.py`

```python
"""APScheduler setup for media trackers.

build_scheduler() returns a configured BackgroundScheduler without starting it,
useful for tests. start() is called from app_v3.py at boot if
APSCHEDULER_ENABLED env var is truthy.
"""
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore

import app_v3
from media_trackers import poller, extractor


def _kill_switch_on() -> bool:
    """Return True if the scheduler should execute job bodies."""
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key = 'media_scheduler_enabled'")
            row = cur.fetchone()
        if not row:
            return True  # default on
        return str(row['value']).lower() not in ('false', '0', 'off', 'no')
    except Exception:
        return False  # fail closed


def run_feed_poller():
    if not _kill_switch_on():
        return
    poller.poll_all_feeds()


def run_extract_worker():
    if not _kill_switch_on():
        return
    extractor.process_extract_batch()


def build_scheduler(use_memory_jobstore: bool = False) -> BackgroundScheduler:
    if use_memory_jobstore or not os.environ.get('DATABASE_URL'):
        jobstore = MemoryJobStore()
    else:
        jobstore = SQLAlchemyJobStore(url=os.environ['DATABASE_URL'], tablename='apscheduler_jobs')
    sched = BackgroundScheduler(
        jobstores={'default': jobstore},
        job_defaults={'coalesce': True, 'max_instances': 1, 'misfire_grace_time': 300},
    )
    sched.add_job(run_feed_poller,   'interval', minutes=30, id='feed_poller', replace_existing=True)
    sched.add_job(run_extract_worker, 'interval', minutes=2,  id='extract_worker', replace_existing=True)
    return sched


_scheduler = None


def start():
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = build_scheduler()
    _scheduler.start()
    return _scheduler
```

- [ ] **Step 4: Register scheduler boot in `app_v3.py`**

Find the `try: init_db()` block near line 1438. Immediately after it, add:

```python
# Start APScheduler for media trackers (unless APSCHEDULER_DISABLED set)
if not os.environ.get('APSCHEDULER_DISABLED'):
    try:
        import scheduler as _media_scheduler
        _media_scheduler.start()
        print("Media tracker scheduler started")
    except Exception as e:
        print(f"Scheduler start failed (non-fatal): {e}")
```

- [ ] **Step 5: Create an extractor stub so `scheduler.py` imports**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/media_trackers/extractor.py` (stub — Task 8 fills it in)

```python
"""Extractor — Haiku-powered bullet extraction from show notes / transcripts.

This is a Task 7 stub; Task 8 implements process_extract_batch properly.
"""


def process_extract_batch():
    """Claim up to 3 'new' episodes and extract digest points. Filled in Task 8."""
    pass
```

- [ ] **Step 6: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_scheduler.py -v
```

Expected: 2 tests pass.

- [ ] **Step 7: Commit**

```bash
git add scheduler.py media_trackers/extractor.py app_v3.py tests/test_scheduler.py
git commit -m "feat(media-trackers): APScheduler with kill switch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Show-notes extraction (M3)

**Files:**
- Modify: `media_trackers/extractor.py` (replace stub with real impl)
- Create: `media_trackers/prompts.py`
- Create: `tests/fixtures/shownotes_sample.txt`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Create show-notes fixture**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/fixtures/shownotes_sample.txt`

```
On this episode of Odd Lots, Tracy Alloway and Joe Weisenthal talk to
Diogo Rau, Chief Information and Digital Officer at Eli Lilly (LLY),
about Zepbound's move from vials to autoinjector pens and how the company
is allocating capacity between the US and Ireland. They also cover NVDA's
H200 hyperscaler allocations and what TSM's latest comments imply for
the broader semis cycle.
```

- [ ] **Step 2: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_extractor.py`

```python
"""Tests for the show-notes extractor."""
import json
from pathlib import Path
from unittest.mock import patch
import app_v3
from media_trackers import extractor

FIXTURE = (Path(__file__).parent / 'fixtures' / 'shownotes_sample.txt').read_text()


def _seed_episode(show_notes=FIXTURE, status='new'):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('f1', 'podcast', 'Odd Lots', 'https://x/rss')
            ON CONFLICT (id) DO NOTHING
        ''')
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, show_notes, status)
            VALUES ('e1', 'f1', 'g1', 'Zepbound supply', %s, %s)
            ON CONFLICT (feed_id, guid) DO NOTHING
        ''', (show_notes, status))
    return 'e1'


FAKE_LLM_RESPONSE = {
    'points': [
        {'text': 'LLY is shifting Zepbound from vials to pen injectors.',
         'tickers': ['LLY'], 'sector_tags': ['PHARMA'], 'theme_tags': ['GLP-1']},
        {'text': 'NVDA H200 allocations going to hyperscalers first.',
         'tickers': ['NVDA'], 'sector_tags': ['SEMIS'], 'theme_tags': ['AI']},
        {'text': 'TSM comments signal broader semis cycle tightening.',
         'tickers': ['TSM'], 'sector_tags': ['SEMIS'], 'theme_tags': []},
    ]
}


def test_extract_from_episode_inserts_points(clean_db):
    _seed_episode()
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.extract_from_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT text, tickers, sector_tags FROM media_digest_points WHERE episode_id='e1' ORDER BY point_order")
        rows = cur.fetchall()
    assert len(rows) == 3
    assert rows[0]['tickers'] == ['LLY']
    assert rows[1]['sector_tags'] == ['SEMIS']


def test_extract_marks_episode_done(clean_db):
    _seed_episode()
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.extract_from_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status FROM media_episodes WHERE id='e1'")
        assert cur.fetchone()['status'] == 'done'


def test_process_extract_batch_picks_up_to_3(clean_db):
    for i in range(5):
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute('''
                INSERT INTO media_feeds (id, source_type, name, feed_url)
                VALUES ('f1','podcast','X','u') ON CONFLICT DO NOTHING
            ''')
            cur.execute('''
                INSERT INTO media_episodes (id, feed_id, guid, title, show_notes, status)
                VALUES (%s, 'f1', %s, 't', %s, 'new')
            ''', (f'e{i}', f'g{i}', FIXTURE))
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.process_extract_batch()
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_episodes WHERE status='done'")
        assert cur.fetchone()['n'] == 3


def test_extract_skips_episode_with_no_show_notes(clean_db):
    _seed_episode(show_notes=None)
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.extract_from_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status, error_message FROM media_episodes WHERE id='e1'")
        row = cur.fetchone()
    assert row['status'] == 'skipped'
    assert 'no content' in row['error_message'].lower()
```

- [ ] **Step 3: Run the test — expect failure**

```bash
.venv/bin/pytest tests/test_extractor.py -v
```

Expected: all 4 fail (stub does nothing).

- [ ] **Step 4: Create prompts module**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/media_trackers/prompts.py`

```python
"""Extraction prompts for media-tracker LLM calls."""

EXTRACTION_PROMPT = """You extract investment-relevant bullet points from podcast episode content (show notes or transcript).

RULES:
- 3-8 bullets per episode. Skip if no investment signal.
- Each bullet: one specific claim, fact, or observation. No boilerplate.
- Tag with tickers if a specific public company is mentioned (US listed; also LSE/HKEX/TSE for ADRs). Use exchange-clean symbols (NVDA, TSM, LLY).
- Tag with sector_tags (uppercase short codes: SEMIS, AI, PHARMA, BIOTECH, MEDTECH, MACRO, CREDIT, BANKS, CONSUMER, ENERGY, INDUSTRIALS, TECH, CRYPTO, M&A).
- Tag with theme_tags for recurring cross-episode topics (GLP-1, ARM servers, AI capex, rate cuts, inventory correction).
- NEVER reference sell-side brokers or analyst opinions. Extract what a guest or host said about fundamentals, numbers, guidance, behavior.
- Output JSON only.

SCHEMA:
{
  "points": [
    {
      "text": "<bullet — ≤30 words, specific>",
      "tickers": ["TICKER1", ...],
      "sector_tags": ["SEMIS", ...],
      "theme_tags": ["GLP-1 supply", ...]
    }
  ]
}

If no investment signal, return {"points": []}.
"""
```

- [ ] **Step 5: Replace `media_trackers/extractor.py` with real implementation**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/media_trackers/extractor.py`

```python
"""Extract bullet-tagged digest points from episode content via Haiku."""
import json
import os
import uuid

import app_v3
from media_trackers.prompts import EXTRACTION_PROMPT

MAX_EPISODES_PER_BATCH = 3


def _call_haiku(content: str) -> dict:
    """Call Claude Haiku with the extraction prompt. Returns parsed JSON."""
    raw = app_v3.call_llm(
        messages=[{'role': 'user', 'content': content[:30000]}],
        system=EXTRACTION_PROMPT,
        tier='fast',
        max_tokens=2000,
    )
    return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = app_v3._repair_truncated_json(raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {'points': []}


def extract_from_episode(episode_id: str) -> None:
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_episodes SET status='extracting' WHERE id=%s AND status='new' RETURNING *",
                    (episode_id,))
        ep = cur.fetchone()
    if not ep:
        return

    content = ep['transcript'] or ep['show_notes']
    if not content:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute("UPDATE media_episodes SET status='skipped', error_message='no content' WHERE id=%s",
                        (episode_id,))
        return

    try:
        result = _call_haiku(content)
    except Exception as e:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute("UPDATE media_episodes SET status='failed', error_message=%s WHERE id=%s",
                        (str(e)[:500], episode_id))
        return

    points = result.get('points') or []
    with app_v3.get_db(commit=True) as (_c, cur):
        for i, p in enumerate(points):
            cur.execute('''
                INSERT INTO media_digest_points
                    (id, episode_id, point_order, text, tickers, sector_tags, theme_tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                str(uuid.uuid4()), episode_id, i,
                (p.get('text') or '')[:2000],
                p.get('tickers') or [],
                p.get('sector_tags') or [],
                p.get('theme_tags') or [],
            ))
        cur.execute("UPDATE media_episodes SET status='done' WHERE id=%s", (episode_id,))


def process_extract_batch() -> None:
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM media_episodes
             WHERE status = 'new'
             ORDER BY published_at DESC NULLS LAST
             LIMIT %s
        ''', (MAX_EPISODES_PER_BATCH,))
        ids = [r['id'] for r in cur.fetchall()]
    for eid in ids:
        extract_from_episode(eid)
```

- [ ] **Step 6: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_extractor.py -v
```

Expected: 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add media_trackers/extractor.py media_trackers/prompts.py \
        tests/fixtures/shownotes_sample.txt tests/test_extractor.py
git commit -m "feat(media-trackers): Haiku show-notes extraction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Feed read + run-scanner endpoints

**Files:**
- Modify: `app_v3.py` (add endpoints below watchlist block)
- Create: `tests/test_feed_api.py`

- [ ] **Step 1: Write failing test**

Path: `/Users/tonydlee/Projects/equity-analyzer-backend/tests/test_feed_api.py`

```python
import time
import app_v3


def _seed_data():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("INSERT INTO media_feeds (id, source_type, name, feed_url) VALUES ('f1','podcast','Odd Lots','https://x')")
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, published_at, source_url, status)
            VALUES ('e1', 'f1', 'g1', 'GLP-1 supply crunch', NOW() - INTERVAL '1 day',
                    'https://example.com/ep1', 'done')
        ''')
        cur.execute('''
            INSERT INTO media_digest_points (id, episode_id, point_order, text, tickers, sector_tags, theme_tags)
            VALUES ('p1','e1', 0, 'LLY Zepbound pen conversion',      ARRAY['LLY'],  ARRAY['PHARMA'], ARRAY['GLP-1']),
                   ('p2','e1', 1, 'NVDA H200 hyperscaler allocation', ARRAY['NVDA'], ARRAY['SEMIS'],  ARRAY['AI'])
        ''')


def test_feed_returns_episodes_with_points(client):
    _seed_data()
    resp = client.get('/api/media/feed')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['total'] == 1
    ep = body['episodes'][0]
    assert ep['title'] == 'GLP-1 supply crunch'
    assert len(ep['points']) == 2
    assert ep['points'][0]['tickers'] == ['LLY']


def test_feed_filter_by_ticker(client):
    _seed_data()
    resp = client.get('/api/media/feed?ticker=NVDA')
    body = resp.get_json()
    assert body['total'] == 1
    # Only the NVDA point should come back for the episode
    points = body['episodes'][0]['points']
    assert all('NVDA' in p['tickers'] for p in points)


def test_run_scanner_returns_202(client, monkeypatch):
    monkeypatch.setattr('media_trackers.poller.poll_all_feeds', lambda: time.sleep(0.01))
    resp = client.post('/api/media/run-scanner')
    assert resp.status_code == 202
    assert 'scanId' in resp.get_json()
```

- [ ] **Step 2: Run the test — expect 404**

```bash
.venv/bin/pytest tests/test_feed_api.py -v
```

Expected: all 3 fail.

- [ ] **Step 3: Add endpoints to `app_v3.py`** (below watchlist block from Task 5)

```python
# ============================================
# MEDIA TRACKER — FEED (firehose) + SCANNER
# ============================================

_scanner_runs = {}  # in-memory scan status; ok to lose across deploys


@app.route('/api/media/feed', methods=['GET'])
def media_feed_read():
source  = request.args.get('source')
    ticker  = request.args.get('ticker')
    sector  = request.args.get('sector')
    days    = int(request.args.get('days', 7))
    q       = request.args.get('q', '').strip()
    material_only = request.args.get('material') == 'true'
    limit   = int(request.args.get('limit', 100))

    where = ["e.created_at > NOW() - %s::interval"]
    params = [f'{days} days']
    if source:
        where.append("f.source_type = %s"); params.append(source)
    if material_only:
        where.append("EXISTS (SELECT 1 FROM media_digest_points p2 WHERE p2.episode_id=e.id AND p2.material)")
    if q:
        where.append("(e.title ILIKE %s OR e.show_notes ILIKE %s)")
        params.extend([f'%{q}%', f'%{q}%'])

    with get_db() as (_c, cur):
        cur.execute(f'''
            SELECT e.*, f.name AS feed_name, f.sector_tags AS feed_sector_tags
              FROM media_episodes e
              JOIN media_feeds f ON f.id = e.feed_id
             WHERE {' AND '.join(where)}
               AND e.status = 'done'
             ORDER BY e.published_at DESC NULLS LAST
             LIMIT %s
        ''', params + [limit])
        episodes = cur.fetchall()

        episode_ids = [e['id'] for e in episodes]
        points = []
        if episode_ids:
            pq = "SELECT * FROM media_digest_points WHERE episode_id = ANY(%s)"
            pparams = [episode_ids]
            if ticker:
                pq += " AND %s = ANY(tickers)"; pparams.append(ticker)
            if sector:
                pq += " AND %s = ANY(sector_tags)"; pparams.append(sector)
            pq += " ORDER BY episode_id, point_order"
            cur.execute(pq, pparams)
            points = cur.fetchall()

    by_ep = {}
    for p in points:
        by_ep.setdefault(p['episode_id'], []).append({
            'id': p['id'], 'text': p['text'], 'tickers': p['tickers'] or [],
            'sectorTags': p['sector_tags'] or [], 'themeTags': p['theme_tags'] or [],
            'material': p['material'], 'timestampSec': p['timestamp_sec'],
        })

    result = []
    for e in episodes:
        pts = by_ep.get(e['id'], [])
        if (ticker or sector) and not pts:
            continue  # filtered all points out — hide the episode
        result.append({
            'id': e['id'], 'feedId': e['feed_id'], 'feedName': e['feed_name'],
            'title': e['title'],
            'publishedAt': e['published_at'].isoformat() if e['published_at'] else None,
            'sourceUrl': e['source_url'], 'points': pts,
        })
    return jsonify({'episodes': result, 'total': len(result)})


@app.route('/api/media/run-scanner', methods=['POST'])
def media_run_scanner():
scan_id = str(uuid.uuid4())
    _scanner_runs[scan_id] = {'status': 'running', 'started_at': datetime.utcnow().isoformat()}

    def _run():
        try:
            from media_trackers import poller
            poller.poll_all_feeds()
            _scanner_runs[scan_id]['status'] = 'done'
        except Exception as e:
            _scanner_runs[scan_id]['status'] = 'failed'
            _scanner_runs[scan_id]['error'] = str(e)
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'scanId': scan_id}), 202


@app.route('/api/media/scan/<scan_id>', methods=['GET'])
def media_scan_status(scan_id):
rec = _scanner_runs.get(scan_id)
    if not rec:
        return jsonify({'error': 'unknown scan id'}), 404
    return jsonify(rec)
```

**Note:** `datetime` is already imported at the top of `app_v3.py`; `threading` is too. Verify with `grep "^import threading" app_v3.py` — if missing, add it at the top.

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_feed_api.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Run the entire suite for a green bar**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass (25+).

- [ ] **Step 6: Commit**

```bash
git add app_v3.py tests/test_feed_api.py
git commit -m "feat(media-trackers): /api/media/feed + /run-scanner endpoints

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Frontend — Feed tab scaffold (M4 part 1)

**Files:**
- Modify: `/Users/tonydlee/Library/CloudStorage/GoogleDrive-tonydlee@gmail.com/My Drive/ea v58 NEW/charlie-deployment/index.html`
- Modify: same dir `worker.js`
- Modify: same dir `service-worker.js`

- [ ] **Step 1: Read current BUILD_VERSION values**

```bash
FE="/Users/tonydlee/Library/CloudStorage/GoogleDrive-tonydlee@gmail.com/My Drive/ea v58 NEW/charlie-deployment"
grep -n "BUILD_VERSION" "$FE/index.html" "$FE/worker.js" "$FE/service-worker.js"
```

Record the three current version strings. The new version will be that + `+feed-scaffold` (or bump minor/patch consistently — follow existing style).

- [ ] **Step 2: Bump BUILD_VERSION in all three files**

For each file, replace the version string. If current is `v2.14.3` then bump to `v2.15.0`.

`index.html` (line ~281):
```javascript
const BUILD_VERSION = 'v2.15.0';  // was v2.14.3
```

`worker.js` (line 2):
```javascript
const BUILD_VERSION = 'v2.15.0';
```

`service-worker.js` (line 3):
```javascript
const BUILD_VERSION = 'v2.15.0';
```

- [ ] **Step 3: Add 'feed' to the activeTab enum comment + nav**

In `index.html`, find line ~1570 where `activeTab` is declared. Update the comment listing valid tabs to include `'feed'`:

```javascript
const [activeTab, setActiveTab] = useState('chat'); // Bottom nav: 'portfolio', 'overview', 'chat', 'summary', 'research', 'settings', 'feed'
```

- [ ] **Step 4: Add nav entry for Feed in the "More" overflow menu**

Find the More overflow menu around line ~22175 (where existing 'alerts' / 'dashboard' / 'research' etc. are listed). Add an entry for 'feed'.

Pattern to match (approximate — adapt to exact surrounding code):

```javascript
// Look for the block rendering More-menu items; it's an array like:
// { id: 'alerts', label: 'Alerts', icon: <Bell/> }, ...
// Add:
{ id: 'feed', label: 'Feed', icon: <Rss/> },
```

If `Rss` isn't already imported, add it to the lucide imports block at top of the React component (search `from 'lucide-react'` or similar). If icons are local SVGs (lines ~700–750), add an `RssIcon` component using this SVG path:

```jsx
const RssIcon = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M4 11a9 9 0 0 1 9 9"/>
    <path d="M4 4a16 16 0 0 1 16 16"/>
    <circle cx="5" cy="19" r="1"/>
  </svg>
);
```

- [ ] **Step 5: Add an empty Feed tab render block**

Find the last `activeTab === '...'` block in `index.html` (around line 21703 for `alerts`). Add directly after it:

```jsx
{activeTab === 'feed' && (
  <div className="min-h-screen bg-slate-50 p-4">
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Feed</h1>
          <p className="text-sm text-slate-500">Podcasts — firehose</p>
        </div>
        <button className="px-4 py-2 bg-slate-900 text-white rounded-lg text-sm">
          Run Scanner
        </button>
      </div>
      <div className="text-slate-400 text-center py-12">
        Task 11 will populate this with episode cards.
      </div>
    </div>
  </div>
)}
```

- [ ] **Step 6: Smoke test locally**

Open `file://<FE>/index.html` in a browser, open dev console, and run:
```javascript
window.location.hash = '#feed'  // if your router uses hash, otherwise click More > Feed
```

Expected: Feed tab renders the heading and placeholder text without console errors.

- [ ] **Step 7: Commit**

```bash
cd "$FE" && git add index.html worker.js service-worker.js
git commit -m "feat(media-trackers): scaffold Feed tab shell + bump build versions

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

> **Note:** `$FE` is outside the backend repo. If Tony keeps frontend in a separate git repo, this commit lands there. If it's not yet a git repo, run `git init` + first commit — the memory rule is "must commit frontend alongside backend" so both need to be versioned.

---

## Task 11: Frontend — render feed cards (M4 part 2)

**Files:**
- Modify: `index.html` (Feed tab block from Task 10)

- [ ] **Step 1: Add state + fetch logic**

In the React component, near existing tab-specific state hooks (search `feedPoints`, `fetchAlerts`, etc. for the pattern), add:

```jsx
// Feed tab state
const [feedEpisodes, setFeedEpisodes] = useState([]);
const [feedLoading, setFeedLoading] = useState(false);
const [feedError, setFeedError] = useState(null);
const [feedFilters, setFeedFilters] = useState({
  source: 'podcast', ticker: '', sector: '', days: 7, q: '', material: false,
});

const fetchFeed = useCallback(async () => {
  setFeedLoading(true);
  setFeedError(null);
  try {
    const params = new URLSearchParams();
    if (feedFilters.source)   params.set('source', feedFilters.source);
    if (feedFilters.ticker)   params.set('ticker', feedFilters.ticker);
    if (feedFilters.sector)   params.set('sector', feedFilters.sector);
    if (feedFilters.days)     params.set('days', String(feedFilters.days));
    if (feedFilters.q)        params.set('q', feedFilters.q);
    if (feedFilters.material) params.set('material', 'true');
    // Note: window.fetch is globally wrapped (index.html ~line 306) to inject
    // Authorization: Bearer <token> on all /api/ URLs — no manual header needed.
    const resp = await fetch(`/api/media/feed?${params}`);
    const data = await resp.json();
    setFeedEpisodes(data.episodes || []);
  } catch (e) {
    setFeedError(String(e));
  } finally {
    setFeedLoading(false);
  }
}, [feedFilters]);
```

Find the existing `useEffect` that switches on `activeTab === 'alerts'` (around line 4502) and add a sibling line:
```jsx
if (activeTab === 'feed') fetchFeed();
```

Add a polling effect next to it:
```jsx
useEffect(() => {
  if (activeTab !== 'feed') return;
  const id = setInterval(fetchFeed, 60000);
  return () => clearInterval(id);
}, [activeTab, fetchFeed]);
```

- [ ] **Step 2: Replace the placeholder block with card rendering**

Replace the `"Task 11 will populate…"` placeholder from Task 10 with:

```jsx
<div className="space-y-4">
  {feedLoading && <div className="text-sm text-slate-500">Loading…</div>}
  {feedError && <div className="text-sm text-rose-600">Error: {feedError}</div>}
  {!feedLoading && feedEpisodes.length === 0 && (
    <div className="text-sm text-slate-400 text-center py-12">
      No episodes yet. Click "Run Scanner" to trigger a fresh poll.
    </div>
  )}
  {feedEpisodes.map(ep => (
    <div key={ep.id} className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
      <div className="flex items-baseline justify-between mb-1">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          {ep.feedName} · {ep.publishedAt ? new Date(ep.publishedAt).toLocaleDateString() : ''}
          {ep.sourceUrl && (
            <a href={ep.sourceUrl} target="_blank" rel="noopener" className="ml-2 text-indigo-600 hover:underline">
              SOURCE ↗
            </a>
          )}
        </div>
      </div>
      <h3 className="font-semibold text-slate-900 mb-3">{ep.title}</h3>
      <ul className="space-y-2">
        {ep.points.map(p => (
          <li key={p.id} className="flex items-start gap-2 text-sm text-slate-700">
            {p.sectorTags.map(t => (
              <span key={t} className="shrink-0 px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 text-[10px] font-semibold">
                {t}
              </span>
            ))}
            {p.material && <span className="shrink-0 text-amber-500">★</span>}
            <span>{p.text}</span>
          </li>
        ))}
      </ul>
    </div>
  ))}
</div>
```

- [ ] **Step 3: Wire the "Run Scanner" button**

Replace the existing button element with:

```jsx
<button
  onClick={async () => {
    // Auth header auto-injected by the global fetch wrapper.
    await fetch('/api/media/run-scanner', { method: 'POST' });
    setTimeout(fetchFeed, 3000);
  }}
  className="px-4 py-2 bg-slate-900 text-white rounded-lg text-sm hover:bg-slate-700"
>
  Run Scanner
</button>
```

- [ ] **Step 4: Bump BUILD_VERSION again** (now v2.15.1)

Same three files as Task 10, increment patch.

- [ ] **Step 5: Local smoke test**

- Start backend: `cd backend && .venv/bin/python app_v3.py` (or whatever the dev entry point is)
- Open frontend index.html in browser
- Click More > Feed — should fetch and render (empty if DB has no episodes)
- Click Run Scanner — after ~3 seconds should see episodes from any feeds whose RSS returns data

- [ ] **Step 6: Commit**

```bash
cd "$FE" && git add index.html worker.js service-worker.js
git commit -m "feat(media-trackers): render Feed cards with tag chips + polling

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Frontend — filters + search (M4 part 3)

**Files:**
- Modify: `index.html` (Feed tab)

- [ ] **Step 1: Add filter UI above the card list**

Insert between the header row and the card list:

```jsx
<div className="flex flex-wrap items-center gap-2 mb-4 text-sm">
  <input
    type="text"
    placeholder="Search episodes, tickers, topics…"
    value={feedFilters.q}
    onChange={e => setFeedFilters({...feedFilters, q: e.target.value})}
    className="flex-1 min-w-[200px] px-3 py-1.5 rounded-lg border border-slate-300"
  />
  <select
    value={feedFilters.days}
    onChange={e => setFeedFilters({...feedFilters, days: Number(e.target.value)})}
    className="px-2 py-1.5 rounded-lg border border-slate-300"
  >
    <option value={1}>Today</option>
    <option value={7}>7 days</option>
    <option value={30}>30 days</option>
  </select>
  <input
    type="text"
    placeholder="Ticker"
    value={feedFilters.ticker}
    onChange={e => setFeedFilters({...feedFilters, ticker: e.target.value.toUpperCase()})}
    className="w-24 px-2 py-1.5 rounded-lg border border-slate-300"
  />
  <label className="flex items-center gap-1">
    <input
      type="checkbox"
      checked={feedFilters.material}
      onChange={e => setFeedFilters({...feedFilters, material: e.target.checked})}
    />
    <span className="text-xs">Material only</span>
  </label>
</div>
```

- [ ] **Step 2: Refetch on filter change**

Add a useEffect that refetches when `feedFilters` changes:

```jsx
useEffect(() => {
  if (activeTab === 'feed') {
    const debounce = setTimeout(fetchFeed, 250);
    return () => clearTimeout(debounce);
  }
}, [feedFilters]);
```

- [ ] **Step 3: Bump BUILD_VERSION to v2.15.2** (same three files)

- [ ] **Step 4: Smoke test**

- Type in search box — results should filter after 250ms
- Switch "days" — results reload
- Enter ticker like "NVDA" — only NVDA-tagged points visible

- [ ] **Step 5: Commit**

```bash
cd "$FE" && git add index.html worker.js service-worker.js
git commit -m "feat(media-trackers): Feed tab filters (search, ticker, date, material)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Deploy & dogfood

**Files:**
- No code changes; deployment steps only.

- [ ] **Step 1: Verify all tests pass**

```bash
cd /Users/tonydlee/Projects/equity-analyzer-backend
.venv/bin/pytest -v
```

Expected: all tests pass (25+ across 7 test files).

- [ ] **Step 2: Verify frontend version bumps**

```bash
FE="/Users/tonydlee/Library/CloudStorage/GoogleDrive-tonydlee@gmail.com/My Drive/ea v58 NEW/charlie-deployment"
grep "BUILD_VERSION" "$FE/index.html" "$FE/worker.js" "$FE/service-worker.js"
```

Expected: all three show `v2.15.2` (or whatever you ended at).

- [ ] **Step 3: Merge feature branch to main**

```bash
cd /Users/tonydlee/Projects/equity-analyzer-backend
git checkout main
git merge --no-ff feature/media-trackers-m1-m4 -m "merge: podcast feed foundation M1–M4"
```

- [ ] **Step 4: Push backend (triggers Render auto-deploy)**

```bash
git push origin main
```

Watch the Render dashboard for a successful deploy.

- [ ] **Step 5: Seed production feeds**

Once Render shows green:

```bash
# One-time invocation from a local shell with prod DATABASE_URL
DATABASE_URL="$PROD_DATABASE_URL" .venv/bin/python -m media_trackers.seed_feeds
```

Expected: `Seeded up to 25 feeds.`

- [ ] **Step 6: Verify backend endpoint**

```bash
curl -H "Authorization: ApiKey $CHARLIE_API_KEY" https://<render-url>/api/media/feeds | jq '.total'
```

Expected: `25`

- [ ] **Step 7: Deploy frontend**

```bash
cd "$FE"
npx wrangler deploy  # NOT `wrangler pages deploy` — custom domain is on the Worker
```

Expected: `Success! Published to https://charlie.tonydlee.com`.

- [ ] **Step 8: End-to-end dogfood**

- Open `https://charlie.tonydlee.com` on desktop + mobile
- Navigate: More > Feed
- Click Run Scanner
- Wait ~2 min, hit the refresh button or let it auto-poll
- Verify episode cards appear with tag chips
- Test filters: search "AI", set days to "Today", set ticker to "NVDA"

- [ ] **Step 9: Tag the release**

```bash
cd /Users/tonydlee/Projects/equity-analyzer-backend
git tag -a media-trackers-m1m4-v1 -m "Podcast feed foundation — M1-M4 live"
git push origin media-trackers-m1m4-v1
```

- [ ] **Step 10: Update CHANGELOG (if exists)** — add entry under a new `### Added` section noting the Feed tab and Phase 2a scope landed.

---

## Verification Checklist (end-of-plan)

- [ ] All 25+ pytest tests pass in CI / locally
- [ ] `init_db()` creates all 6 new tables
- [ ] 25 seed feeds inserted in prod
- [ ] APScheduler log shows `Media tracker scheduler started` on Render boot
- [ ] `GET /api/media/feeds` returns 25 feeds
- [ ] After one poller tick, `GET /api/media/feed` returns episodes
- [ ] Feed tab renders cards with bullet points, tag chips, SOURCE links
- [ ] Filters work: search, days, ticker, material-only
- [ ] Run Scanner button triggers a fresh poll
- [ ] No errors in Render logs, no JS errors in browser console
- [ ] Three version strings all bumped to same value
- [ ] Backend + frontend committed and pushed

---

## Self-Review Notes (for the writer, not the executor)

- **Spec coverage:** M1 (DB + seed + feeds CRUD) → Tasks 2–4. M2 (poller) → Task 6. Scheduler → Task 7. M3 (show-notes extraction) → Task 8. M4 (Feed UI) → Tasks 10–12. Deploy → Task 13. Signals watchlist is included (Task 5) even though its UI is a Plan 2 item — worth having the backend ready so Plan 2 is purely UI.
- **Deferred to Plan 2 (M5–M8):** transcription (publisher scrapers + Gemini fallback), material gating + 2nd-pass Sonnet, agent_alerts creation, dedup logic, clustering, cost_watch, notifications (push/Telegram/email), Settings UI.
- **Known risks:** (1) some seeded RSS URLs may be out-of-date; Tony will redline. (2) RSS scraping rate limits — the `time.sleep(0.25)` between feeds in `poll_all_feeds()` should cover ~25 feeds in < 15 sec. (3) first `poll_all_feeds()` on empty DB will attempt a 7-day backfill on every feed — cost is show-notes only at this stage, so fine (~$0.75 of Haiku).
- **Deviation from spec:** the spec describes a separate `extract_worker` that picks up `status='extracting'` after a future transcribe stage. For M3 (show-notes only), `process_extract_batch` picks up `status='new'` directly and skips the `extracting` state. When M5 lands, the flow becomes `new → transcribing → extracting → done` and `process_extract_batch` will switch to `status='extracting'`. Noted in Task 8 — an M5-time follow-up.
