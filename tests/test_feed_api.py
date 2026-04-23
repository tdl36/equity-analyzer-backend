"""Tests for /api/media/feed and /api/media/run-scanner."""
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
    points = body['episodes'][0]['points']
    assert all('NVDA' in p['tickers'] for p in points)


def test_run_scanner_returns_202(client, monkeypatch):
    monkeypatch.setattr('media_trackers.poller.poll_all_feeds', lambda: time.sleep(0.01))
    resp = client.post('/api/media/run-scanner')
    assert resp.status_code == 202
    assert 'scanId' in resp.get_json()
