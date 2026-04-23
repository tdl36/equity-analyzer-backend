"""Tests for media-tracker weekly cost watcher (M8)."""
import json

import app_v3
from media_trackers import cost_watch


def _seed_feed_and_episode(cost_usd, episode_id='c-ep', when_sql="NOW()"):
    """Insert a feed + episode with the given cost. Safe to call multiple times."""
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            """
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('cf1', 'podcast', 'Cost Test Feed', 'https://x')
            ON CONFLICT (id) DO NOTHING
            """
        )
        cur.execute(
            f"""
            INSERT INTO media_episodes (id, feed_id, guid, title, source_url, status, cost_usd, created_at)
            VALUES (%s, 'cf1', %s, 'episode', 'https://ep', 'done', %s, {when_sql})
            ON CONFLICT (feed_id, guid) DO NOTHING
            """,
            (episode_id, f'guid-{episode_id}', cost_usd),
        )


def test_cost_watch_no_warning_under_soft(clean_db):
    _seed_feed_and_episode(10.0, episode_id='c-ep-a')
    # Set soft cap to 15
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_cost_weekly_warn_usd', '15')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            '''
        )
    cost_watch.check_cost_warning()
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM agent_alerts WHERE alert_type='system'")
        n = cur.fetchone()['n']
    assert n == 0


def test_cost_watch_fires_soft_warning(clean_db):
    _seed_feed_and_episode(20.0, episode_id='c-ep-b')
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_cost_weekly_warn_usd', '15')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            '''
        )
    cost_watch.check_cost_warning()
    with app_v3.get_db() as (_c, cur):
        cur.execute(
            "SELECT title, detail FROM agent_alerts WHERE alert_type='system' ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        # Ensure scheduler was NOT disabled
        cur.execute("SELECT value FROM app_settings WHERE key='media_scheduler_enabled'")
        kill = cur.fetchone()
    assert row is not None
    assert '20.00' in row['title']
    det = row['detail'] if isinstance(row['detail'], dict) else json.loads(row['detail'])
    assert det['weeklyCostUsd'] == 20.0
    assert det.get('hardCapped') is not True
    assert kill is None or str(kill['value']).lower() not in ('false', '0', 'off')


def test_cost_watch_hard_cap_disables_scheduler(clean_db):
    _seed_feed_and_episode(60.0, episode_id='c-ep-c')
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_cost_weekly_warn_usd', '15')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            '''
        )
    cost_watch.check_cost_warning()
    with app_v3.get_db() as (_c, cur):
        cur.execute(
            "SELECT title, detail FROM agent_alerts WHERE alert_type='system' ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        cur.execute("SELECT value FROM app_settings WHERE key='media_scheduler_enabled'")
        kill = cur.fetchone()
    assert row is not None
    assert 'auto-paused' in row['title']
    det = row['detail'] if isinstance(row['detail'], dict) else json.loads(row['detail'])
    assert det.get('hardCapped') is True
    assert kill is not None
    assert str(kill['value']).lower() == 'false'


def test_cost_watch_ignores_episodes_outside_7d_window(clean_db):
    _seed_feed_and_episode(80.0, episode_id='c-ep-old', when_sql="NOW() - INTERVAL '10 days'")
    cost_watch.check_cost_warning()
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM agent_alerts WHERE alert_type='system'")
        n = cur.fetchone()['n']
    assert n == 0
