"""Shared pytest fixtures.

Tests require TEST_DATABASE_URL set to a Postgres instance that the test
process can freely create/drop tables in. All tables are truncated between
tests via the `clean_db` fixture.
"""
import os

os.environ.setdefault('CHARLIE_PASSWORD', '')
os.environ.setdefault('CHARLIE_API_KEY', '')
os.environ.setdefault('APSCHEDULER_DISABLED', '1')
os.environ.setdefault('DATABASE_URL', os.environ.get('TEST_DATABASE_URL', 'postgres://localhost:5432/charlie_test'))

import pytest
import app_v3  # noqa: E402


@pytest.fixture(scope='session', autouse=True)
def _init_schema():
    app_v3.init_db()
    yield


@pytest.fixture
def clean_db():
    """Truncate all tracker-related tables between tests.

    Note: Some of these tables don't exist yet (added in Task 2). Use
    the try/except to make this a no-op when tables are missing.
    """
    try:
        with app_v3.get_db(commit=True) as (_conn, cur):
            cur.execute("""
                TRUNCATE media_point_notes, media_digest_points, media_episodes, media_feeds,
                         signals_watchlist, media_theme_clusters, notification_prefs,
                         agent_alerts, mp_jobs, portfolio_analyses,
                         analyst_activities, analysts,
                         earnings_calendar, ticker_earnings_config,
                         research_pipeline_jobs,
                         research_documents
                RESTART IDENTITY CASCADE
            """)
            cur.execute("""
                DELETE FROM app_settings WHERE key IN (
                    'media_scheduler_enabled',
                    'media_notification_channels',
                    'media_email_digest_to',
                    'telegram_chat_id',
                    'media_cost_weekly_warn_usd',
                    'media_muted_coverage_tickers',
                    'finnhub_api_key',
                    'finnhub_last_sync'
                )
            """)
    except Exception:
        pass  # media_* tables don't exist until Task 2; agent_alerts does
    yield


@pytest.fixture
def client(clean_db):
    app_v3.app.config['TESTING'] = True
    with app_v3.app.test_client() as c:
        yield c
