"""Tests for Feature 10 — inline notes on media bullets.

Endpoints:
  - GET    /api/media/points/<point_id>/notes
  - POST   /api/media/points/<point_id>/notes
  - DELETE /api/media/point-notes/<note_id>

Also verifies /api/media/feed returns noteCount per point.
"""
import app_v3


def _seed_point():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "INSERT INTO media_feeds (id, source_type, name, feed_url) "
            "VALUES ('f1','podcast','Odd Lots','https://x')"
        )
        cur.execute(
            """
            INSERT INTO media_episodes (id, feed_id, guid, title, published_at, source_url, status)
            VALUES ('e1', 'f1', 'g1', 'GLP-1 supply crunch', NOW() - INTERVAL '1 day',
                    'https://example.com/ep1', 'done')
            """
        )
        cur.execute(
            """
            INSERT INTO media_digest_points (id, episode_id, point_order, text, tickers, theme_tags)
            VALUES ('p1','e1', 0, 'LLY pen conversion', ARRAY['LLY'], ARRAY['GLP-1'])
            """
        )


def test_notes_list_empty_initially(client):
    _seed_point()
    resp = client.get('/api/media/points/p1/notes')
    assert resp.status_code == 200
    assert resp.get_json()['notes'] == []


def test_notes_create_and_list(client):
    _seed_point()
    resp = client.post('/api/media/points/p1/notes', json={'noteText': 'Track Q2 pen mix'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'note' in body
    assert body['note']['noteText'] == 'Track Q2 pen mix'
    note_id = body['note']['id']

    # Second note
    resp = client.post('/api/media/points/p1/notes', json={'noteText': 'Check vs. NVO'})
    assert resp.status_code == 200

    resp = client.get('/api/media/points/p1/notes')
    notes = resp.get_json()['notes']
    assert len(notes) == 2
    assert notes[0]['noteText'] == 'Track Q2 pen mix'
    assert notes[1]['noteText'] == 'Check vs. NVO'

    # Delete first note
    resp = client.delete(f'/api/media/point-notes/{note_id}')
    assert resp.status_code == 200
    assert resp.get_json()['deleted'] is True

    resp = client.get('/api/media/points/p1/notes')
    assert len(resp.get_json()['notes']) == 1


def test_notes_create_rejects_empty(client):
    _seed_point()
    resp = client.post('/api/media/points/p1/notes', json={'noteText': '   '})
    assert resp.status_code == 400


def test_notes_create_404_for_missing_point(client):
    resp = client.post('/api/media/points/does-not-exist/notes', json={'noteText': 'hi'})
    assert resp.status_code == 404


def test_notes_delete_404_for_missing_note(client):
    resp = client.delete('/api/media/point-notes/nope')
    assert resp.status_code == 404


def test_feed_exposes_note_count(client):
    _seed_point()
    # Add two notes
    client.post('/api/media/points/p1/notes', json={'noteText': 'a'})
    client.post('/api/media/points/p1/notes', json={'noteText': 'b'})

    resp = client.get('/api/media/feed')
    assert resp.status_code == 200
    body = resp.get_json()
    p = body['episodes'][0]['points'][0]
    assert p['id'] == 'p1'
    assert p['noteCount'] == 2


def test_feed_note_count_zero_when_no_notes(client):
    _seed_point()
    resp = client.get('/api/media/feed')
    p = resp.get_json()['episodes'][0]['points'][0]
    assert p['noteCount'] == 0
