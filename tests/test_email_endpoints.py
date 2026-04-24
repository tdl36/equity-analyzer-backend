"""Tests for Feature 9 email-share endpoints.

Endpoints:
  - POST /api/media/email-bullet
  - POST /api/media/email-ticker-digest
  - POST /api/media/email-episode

All three gracefully return {sent: false, reason: ...} when SMTP env is not
configured instead of raising 500s.
"""
import app_v3


def _seed_episode_points():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "INSERT INTO media_feeds (id, source_type, name, feed_url) "
            "VALUES ('f1','podcast','Odd Lots','https://x')"
        )
        cur.execute(
            """
            INSERT INTO media_episodes (id, feed_id, guid, title, published_at, source_url, status)
            VALUES ('e1', 'f1', 'g1', 'GLP-1 crunch', NOW() - INTERVAL '1 day',
                    'https://example.com/ep1', 'done')
            """
        )
        cur.execute(
            """
            INSERT INTO media_digest_points (id, episode_id, point_order, text, tickers, material)
            VALUES ('p1','e1', 0, 'LLY Zepbound scale', ARRAY['LLY'], TRUE),
                   ('p2','e1', 1, 'Supply constraints', ARRAY['LLY','NVO'], FALSE)
            """
        )


def _clear_smtp(monkeypatch):
    monkeypatch.delenv('SMTP_HOST', raising=False)
    monkeypatch.delenv('SMTP_USER', raising=False)
    monkeypatch.delenv('SMTP_PASSWORD', raising=False)


def test_email_bullet_no_smtp_returns_sent_false(client, monkeypatch):
    _clear_smtp(monkeypatch)
    resp = client.post('/api/media/email-bullet', json={
        'ticker': 'LLY',
        'pointText': 'Zepbound pen conversion accelerating',
        'feedName': 'Odd Lots',
        'episodeTitle': 'GLP-1 crunch',
        'sourceUrl': 'https://example.com/ep1',
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['sent'] is False
    assert 'reason' in body


def test_email_bullet_requires_point_text(client, monkeypatch):
    _clear_smtp(monkeypatch)
    resp = client.post('/api/media/email-bullet', json={'ticker': 'LLY'})
    assert resp.status_code == 400
    assert resp.get_json()['sent'] is False


def test_email_ticker_digest_no_smtp(client, monkeypatch):
    _seed_episode_points()
    _clear_smtp(monkeypatch)
    resp = client.post('/api/media/email-ticker-digest', json={'ticker': 'LLY', 'days': 7})
    assert resp.status_code == 200
    assert resp.get_json()['sent'] is False


def test_email_ticker_digest_requires_ticker(client, monkeypatch):
    _clear_smtp(monkeypatch)
    resp = client.post('/api/media/email-ticker-digest', json={})
    assert resp.status_code == 400


def test_email_episode_no_smtp(client, monkeypatch):
    _seed_episode_points()
    _clear_smtp(monkeypatch)
    resp = client.post('/api/media/email-episode', json={'episodeId': 'e1'})
    assert resp.status_code == 200
    assert resp.get_json()['sent'] is False


def test_email_episode_requires_episode_id(client, monkeypatch):
    _clear_smtp(monkeypatch)
    resp = client.post('/api/media/email-episode', json={})
    assert resp.status_code == 400


def test_email_bullet_sends_when_smtp_configured(client, monkeypatch):
    """With SMTP env set and _email_send stubbed, returns sent=true."""
    monkeypatch.setenv('SMTP_HOST', 'smtp.test.com')
    monkeypatch.setenv('SMTP_USER', 'user@test.com')
    monkeypatch.setenv('SMTP_PASSWORD', 'pw')

    sent_records = []

    def fake_send(subject, html, to=''):
        sent_records.append({'subject': subject, 'html': html, 'to': to})

    # Patch the import target used inside app_v3
    import media_trackers.notifications as notif
    monkeypatch.setattr(notif, '_email_send', fake_send)

    resp = client.post('/api/media/email-bullet', json={
        'ticker': 'LLY',
        'pointText': 'test bullet',
    })
    assert resp.status_code == 200
    assert resp.get_json()['sent'] is True
    assert len(sent_records) == 1
    assert 'LLY' in sent_records[0]['subject']


def test_cluster_episodes_404_for_missing(client):
    resp = client.get('/api/media/clusters/missing-cluster-id/episodes')
    assert resp.status_code == 404
