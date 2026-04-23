"""Tests for /api/media/theme-clusters."""
from datetime import date, timedelta
import app_v3


def _seed_clusters():
    # Most recent Monday
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "INSERT INTO media_feeds (id, source_type, name, feed_url) "
            "VALUES ('ft','podcast','Macro Voices','https://x')"
        )
        cur.execute(
            """
            INSERT INTO media_episodes (id, feed_id, guid, title, published_at, source_url, status)
            VALUES ('et1','ft','gt1','AI buildout', NOW() - INTERVAL '2 days',
                    'https://example.com/et1', 'done')
            """
        )
        cur.execute(
            """
            INSERT INTO media_digest_points (id, episode_id, point_order, text, tickers)
            VALUES ('tp1','et1',0,'NVDA allocation', ARRAY['NVDA']),
                   ('tp2','et1',1,'MSFT capex up',   ARRAY['MSFT'])
            """
        )
        cur.execute(
            """
            INSERT INTO media_theme_clusters (id, theme, summary, point_ids, primary_tickers, week_start)
            VALUES ('c1','AI Capex','Hyperscaler spend accelerating',
                    ARRAY['tp1','tp2'], ARRAY['NVDA','MSFT'], %s)
            """,
            (monday,),
        )
    return monday


def test_theme_clusters_empty(client):
    resp = client.get('/api/media/theme-clusters')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['weekStart'] is None
    assert body['clusters'] == []


def test_theme_clusters_populated(client):
    monday = _seed_clusters()
    resp = client.get('/api/media/theme-clusters?week=current')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['weekStart'] == monday.isoformat()
    clusters = body['clusters']
    assert len(clusters) == 1
    c = clusters[0]
    assert c['theme'] == 'AI Capex'
    assert c['pointCount'] == 2
    assert set(c['primaryTickers']) == {'NVDA', 'MSFT'}
    # Associated points joined in with text + episode source
    assert len(c['points']) == 2
    texts = {p['text'] for p in c['points']}
    assert texts == {'NVDA allocation', 'MSFT capex up'}
    assert all(p['sourceUrl'] == 'https://example.com/et1' for p in c['points'])
