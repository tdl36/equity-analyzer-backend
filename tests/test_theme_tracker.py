"""Cross-stock theme tracker tests."""
import uuid
from datetime import datetime, timedelta

import app_v3
import theme_tracker


def _insert_feed_and_episode(cur, feed_id, feed_name, ep_id, pub_at):
    cur.execute('''
        INSERT INTO media_feeds (id, source_type, name, feed_url)
        VALUES (%s, 'podcast', %s, %s) ON CONFLICT DO NOTHING
    ''', (feed_id, feed_name, f'https://x/{feed_id}'))
    cur.execute('''
        INSERT INTO media_episodes (id, feed_id, guid, title, published_at, status)
        VALUES (%s, %s, %s, %s, %s, 'done') ON CONFLICT (feed_id, guid) DO NOTHING
    ''', (ep_id, feed_id, f'g-{ep_id}', f'ep-{ep_id}', pub_at))


def _insert_point(cur, ep_id, tickers, theme_tags, material=False, text='bullet'):
    cur.execute('''
        INSERT INTO media_digest_points
            (id, episode_id, point_order, text, tickers, sector_tags, theme_tags, material)
        VALUES (%s, %s, 0, %s, %s, %s, %s, %s)
    ''', (str(uuid.uuid4()), ep_id, text, tickers, [], theme_tags, material))


def _seed_cross_theme(coverage_tickers):
    """Drop 4 bullets in the last 24h tagged with 'ai_capex' across 3 covered tickers."""
    now = datetime.utcnow()
    with app_v3.get_db(commit=True) as (_c, cur):
        _insert_feed_and_episode(cur, 'f1', 'BG2 Pod', 'e1', now - timedelta(hours=12))
        _insert_feed_and_episode(cur, 'f2', 'Acquired', 'e2', now - timedelta(hours=5))
        _insert_point(cur, 'e1', [coverage_tickers[0]], ['ai_capex'], material=True, text=f'{coverage_tickers[0]} capex surge')
        _insert_point(cur, 'e1', [coverage_tickers[1]], ['ai_capex'], material=False, text=f'{coverage_tickers[1]} feeds AI capex')
        _insert_point(cur, 'e2', [coverage_tickers[2]], ['ai_capex'], material=True, text=f'{coverage_tickers[2]} AI-capex winner')
        _insert_point(cur, 'e2', [coverage_tickers[0]], ['ai_capex'], material=False, text='Follow-up mention')


def test_theme_scan_routes_when_threshold_hit(client, clean_db):
    aid = client.post('/api/analysts', json={
        'name': 'Tech', 'coverageTickers': ['NVDA', 'AMAT', 'TSM', 'LRCX'],
    }).get_json()['analyst']['id']
    _seed_cross_theme(['NVDA', 'AMAT', 'TSM'])
    stats = theme_tracker.run_theme_scan()
    assert stats['alerts_created'] == 1
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT activity_type, input FROM analyst_activities WHERE analyst_id=%s", (aid,))
        row = cur.fetchone()
    assert row['activity_type'] == 'theme_alert'
    import json as _j
    inp = row['input'] if isinstance(row['input'], dict) else _j.loads(row['input'])
    assert inp['bulletCount'] == 4
    assert set(inp['tickers']) == {'NVDA', 'AMAT', 'TSM'}
    assert inp['normalizedTheme'] == 'ai_capex'


def test_theme_scan_skips_below_ticker_threshold(client, clean_db):
    client.post('/api/analysts', json={
        'name': 'Tech', 'coverageTickers': ['NVDA', 'AMAT'],
    })
    # Only 1 covered ticker mentioned — below MIN_TICKERS=2
    now = datetime.utcnow()
    with app_v3.get_db(commit=True) as (_c, cur):
        _insert_feed_and_episode(cur, 'f1', 'BG2', 'e1', now - timedelta(hours=1))
        _insert_point(cur, 'e1', ['NVDA'], ['ai_capex'])
        _insert_point(cur, 'e1', ['NVDA'], ['ai_capex'])
        _insert_point(cur, 'e1', ['NVDA'], ['ai_capex'])
    stats = theme_tracker.run_theme_scan()
    assert stats['alerts_created'] == 0


def test_theme_scan_skips_below_bullet_threshold(client, clean_db):
    client.post('/api/analysts', json={
        'name': 'Tech', 'coverageTickers': ['NVDA', 'AMAT', 'TSM'],
    })
    now = datetime.utcnow()
    with app_v3.get_db(commit=True) as (_c, cur):
        _insert_feed_and_episode(cur, 'f1', 'BG2', 'e1', now - timedelta(hours=1))
        # Only 2 bullets — below MIN_BULLETS=3
        _insert_point(cur, 'e1', ['NVDA'], ['ai_capex'])
        _insert_point(cur, 'e1', ['AMAT'], ['ai_capex'])
    stats = theme_tracker.run_theme_scan()
    assert stats['alerts_created'] == 0


def test_theme_scan_dedups_within_24h(client, clean_db):
    aid = client.post('/api/analysts', json={
        'name': 'Tech', 'coverageTickers': ['NVDA', 'AMAT', 'TSM'],
    }).get_json()['analyst']['id']
    _seed_cross_theme(['NVDA', 'AMAT', 'TSM'])
    r1 = theme_tracker.run_theme_scan()
    r2 = theme_tracker.run_theme_scan()
    assert r1['alerts_created'] == 1
    assert r2['alerts_created'] == 0
    assert r2['deduped'] == 1


def test_theme_scan_endpoint(client, clean_db):
    client.post('/api/analysts', json={'name': 'X', 'coverageTickers': ['NVDA', 'AMAT', 'TSM']})
    _seed_cross_theme(['NVDA', 'AMAT', 'TSM'])
    r = client.post('/api/themes/scan')
    assert r.status_code == 200
    body = r.get_json()
    assert body['alerts_created'] == 1


def test_theme_scan_ignores_bullets_outside_window(client, clean_db):
    client.post('/api/analysts', json={'name': 'X', 'coverageTickers': ['NVDA', 'AMAT', 'TSM']})
    old = datetime.utcnow() - timedelta(hours=72)  # outside 48h window
    with app_v3.get_db(commit=True) as (_c, cur):
        _insert_feed_and_episode(cur, 'f1', 'BG2', 'e1', old)
        _insert_point(cur, 'e1', ['NVDA'], ['ai_capex'])
        _insert_point(cur, 'e1', ['AMAT'], ['ai_capex'])
        _insert_point(cur, 'e1', ['TSM'], ['ai_capex'])
    stats = theme_tracker.run_theme_scan()
    assert stats['alerts_created'] == 0
