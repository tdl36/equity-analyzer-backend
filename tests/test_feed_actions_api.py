"""Tests for Feature 5-8 endpoints:
  - POST /api/media/attach-to-thesis
  - POST /api/media/mute-theme
  - POST /api/media/episode/<id>/full-summary
  - GET  /api/media/feed (hasTranscript flag)
"""
import json
import app_v3


def _seed_episode_with_transcript():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("INSERT INTO media_feeds (id, source_type, name, feed_url) VALUES ('f1','podcast','Odd Lots','https://x')")
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, published_at, source_url, transcript, status)
            VALUES ('e1', 'f1', 'g1', 'GLP-1 supply crunch',
                    NOW() - INTERVAL '1 day',
                    'https://example.com/ep1',
                    'This is a transcript. LLY continues to scale Zepbound.',
                    'done')
        ''')
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, published_at, source_url, transcript, status)
            VALUES ('e2', 'f1', 'g2', 'Silent episode',
                    NOW() - INTERVAL '1 day',
                    'https://example.com/ep2',
                    '',
                    'done')
        ''')
        cur.execute('''
            INSERT INTO media_digest_points (id, episode_id, point_order, text, tickers, theme_tags)
            VALUES ('p1','e1', 0, 'LLY pen conversion accelerating', ARRAY['LLY'], ARRAY['GLP-1'])
        ''')


def test_feed_returns_has_transcript(client):
    _seed_episode_with_transcript()
    resp = client.get('/api/media/feed')
    assert resp.status_code == 200
    body = resp.get_json()
    ep_map = {ep['id']: ep for ep in body['episodes']}
    assert ep_map['e1']['hasTranscript'] is True
    assert ep_map['e2']['hasTranscript'] is False


def test_attach_to_thesis_creates_and_appends(client):
    _seed_episode_with_transcript()
    # First attach creates row
    resp = client.post('/api/media/attach-to-thesis', json={
        'ticker': 'LLY',
        'bulletText': 'LLY pen conversion accelerating',
        'episodeTitle': 'GLP-1 supply crunch',
        'feedName': 'Odd Lots',
        'sourceUrl': 'https://example.com/ep1',
    })
    assert resp.status_code == 200
    assert resp.get_json()['success'] is True

    # Second attach appends
    resp = client.post('/api/media/attach-to-thesis', json={
        'ticker': 'LLY',
        'bulletText': 'Zepbound capacity ramp',
    })
    assert resp.status_code == 200

    # Verify persisted
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT media_notes FROM portfolio_analyses WHERE ticker = 'LLY'")
        row = cur.fetchone()
    notes = row['media_notes']
    if isinstance(notes, str):
        notes = json.loads(notes)
    assert len(notes) == 2
    assert notes[0]['bulletText'] == 'LLY pen conversion accelerating'
    assert notes[1]['bulletText'] == 'Zepbound capacity ramp'


def test_attach_to_thesis_requires_ticker_and_bullet(client):
    resp = client.post('/api/media/attach-to-thesis', json={'ticker': ''})
    assert resp.status_code == 400


def test_mute_theme_upserts_to_watchlist(client):
    resp = client.post('/api/media/mute-theme', json={'theme': 'GLP-1'})
    assert resp.status_code == 200
    assert resp.get_json()['success'] is True

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT * FROM signals_watchlist WHERE kind='keyword' AND value='GLP-1'")
        row = cur.fetchone()
    assert row is not None
    assert row['muted'] is True

    # Idempotent: calling again does not crash and keeps muted=true
    resp = client.post('/api/media/mute-theme', json={'theme': 'GLP-1'})
    assert resp.status_code == 200


def test_mute_theme_requires_theme(client):
    resp = client.post('/api/media/mute-theme', json={})
    assert resp.status_code == 400


def test_full_summary_rejects_episode_without_transcript(client, monkeypatch):
    _seed_episode_with_transcript()
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    resp = client.post('/api/media/episode/e2/full-summary', json={})
    assert resp.status_code == 400


def test_full_summary_404_for_missing_episode(client, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    resp = client.post('/api/media/episode/does-not-exist/full-summary', json={})
    assert resp.status_code == 404


def test_full_summary_queues_job(client, monkeypatch):
    _seed_episode_with_transcript()
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')

    # Stub out the thread-based runner so we don't make real LLM calls
    monkeypatch.setattr(app_v3, '_run_podcast_fullsummary_job', lambda *a, **kw: None)

    resp = client.post('/api/media/episode/e1/full-summary', json={})
    assert resp.status_code == 202
    body = resp.get_json()
    assert 'jobId' in body

    # mp_jobs row should exist with stage 'podcast-fullsummary'
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT * FROM mp_jobs WHERE id = %s", (body['jobId'],))
        row = cur.fetchone()
    assert row is not None
    assert row['stage'] == 'podcast-fullsummary'
