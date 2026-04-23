"""Tests for /api/media/ticker-heatmap."""
import app_v3


def _seed_heatmap():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "INSERT INTO media_feeds (id, source_type, name, feed_url) "
            "VALUES ('fh','podcast','Odd Lots','https://x')"
        )
        cur.execute(
            """
            INSERT INTO media_episodes (id, feed_id, guid, title, published_at, source_url, status)
            VALUES ('eh1','fh','gh1','AI capex ramp', NOW() - INTERVAL '1 day',
                    'https://example.com/eh1', 'done')
            """
        )
        cur.execute(
            """
            INSERT INTO media_digest_points (id, episode_id, point_order, text, tickers)
            VALUES
                ('hp1','eh1',0,'NVDA H200 allocation', ARRAY['NVDA']),
                ('hp2','eh1',1,'NVDA rev guide raise', ARRAY['NVDA']),
                ('hp3','eh1',2,'LLY Zepbound pen',     ARRAY['LLY']),
                ('hp4','eh1',3,'NVDA / AMD mix',       ARRAY['NVDA','AMD'])
            """
        )


def test_heatmap_empty(client):
    resp = client.get('/api/media/ticker-heatmap')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['heatmap'] == []
    assert body['days'] == 7


def test_heatmap_counts_and_ordering(client):
    _seed_heatmap()
    resp = client.get('/api/media/ticker-heatmap?days=7')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['days'] == 7
    heatmap = body['heatmap']
    assert len(heatmap) == 3
    # NVDA appears 3 times, LLY 1, AMD 1 — NVDA must be first
    assert heatmap[0] == {'ticker': 'NVDA', 'count': 3}
    by_ticker = {row['ticker']: row['count'] for row in heatmap}
    assert by_ticker['LLY'] == 1
    assert by_ticker['AMD'] == 1
