"""Tests for Feature 10 — inline notes on media bullets.

Endpoints:
  - GET    /api/media/points/<point_id>/notes
  - POST   /api/media/points/<point_id>/notes
  - DELETE /api/media/point-notes/<note_id>

Also verifies /api/media/feed returns noteCount per point.
"""
import json

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


# --- draft review flow -------------------------------------------------------
#
# Generation used to overwrite the live note directly, and on the agent it also
# moved the source documents into Processed/ before anyone had read the result,
# so rejecting a bad run meant restoring by hand from Prior Versions/.

def _insert_note(client, ticker, version, status, body):
    """Insert a note row directly; the generator is exercised elsewhere."""
    import uuid as _uuid
    from app_v3 import get_db
    note_id = str(_uuid.uuid4())
    with get_db(commit=True) as (_conn, cur):
        cur.execute(
            """INSERT INTO research_notes (id, ticker, version, note_markdown, status)
               VALUES (%s, %s, %s, %s, %s)""",
            (note_id, ticker, version, body, status))
    return note_id


def test_a_draft_never_reads_as_the_live_note(client):
    """The whole point of a draft: nothing downstream may pick it up."""
    _insert_note(client, 'AAA', '1.0', 'published', 'the live note')
    _insert_note(client, 'AAA', '1.1', 'draft', 'the draft')

    body = client.get('/api/notes/AAA').get_json()
    assert 'the live note' in json.dumps(body), body
    assert 'the draft' not in json.dumps(body)


def test_accepting_a_draft_publishes_it_and_supersedes_the_old_one(client):
    _insert_note(client, 'BBB', '1.0', 'published', 'old')
    draft = _insert_note(client, 'BBB', '1.1', 'draft', 'new')

    r = client.post(f'/api/notes/{draft}/accept')
    assert r.status_code == 200, r.get_json()

    body = json.dumps(client.get('/api/notes/BBB').get_json())
    assert 'new' in body and 'old' not in body

    from app_v3 import get_db
    with get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM research_notes "
                    "WHERE ticker = 'BBB' AND status = 'published'")
        assert cur.fetchone()['n'] == 1, 'exactly one live note per ticker'


def test_discarding_a_draft_leaves_the_published_note_alone(client):
    _insert_note(client, 'CCC', '1.0', 'published', 'keep me')
    draft = _insert_note(client, 'CCC', '1.1', 'draft', 'throw me away')

    assert client.post(f'/api/notes/{draft}/discard').status_code == 200
    body = json.dumps(client.get('/api/notes/CCC').get_json())
    assert 'keep me' in body and 'throw me away' not in body


def test_a_published_note_cannot_be_discarded(client):
    live = _insert_note(client, 'DDD', '1.0', 'published', 'live')
    assert client.post(f'/api/notes/{live}/discard').status_code == 404
    assert 'live' in json.dumps(client.get('/api/notes/DDD').get_json())


def test_the_draft_endpoint_returns_what_to_compare_against(client):
    _insert_note(client, 'EEE', '1.0', 'published', 'previous version')
    _insert_note(client, 'EEE', '1.1', 'draft', 'proposed version')

    body = client.get('/api/notes/EEE/draft').get_json()
    assert body['draft']['note_markdown'] == 'proposed version'
    assert body['published']['noteMarkdown'] == 'previous version'


def test_no_draft_is_not_an_error(client):
    _insert_note(client, 'FFF', '1.0', 'published', 'only a live note')
    assert client.get('/api/notes/FFF/draft').get_json()['draft'] is None


def test_publishing_queues_the_note_for_icloud(client):
    """Publishing is when the note becomes real, so it is when it should sync."""
    import app_v3
    app_v3._pending_local_syncs.clear()
    draft = _insert_note(client, 'GGG', '2.0', 'draft', 'body')

    body = client.post(f'/api/notes/{draft}/accept').get_json()
    assert body['queuedForICloud'] is True
    queued = [s for s in app_v3._pending_local_syncs if s['ticker'] == 'GGG']
    assert len(queued) == 1 and queued[0]['version'] == '2.0'


def test_discarding_queues_nothing(client):
    import app_v3
    app_v3._pending_local_syncs.clear()
    draft = _insert_note(client, 'HHH', '2.0', 'draft', 'body')
    client.post(f'/api/notes/{draft}/discard')
    assert not [s for s in app_v3._pending_local_syncs if s['ticker'] == 'HHH']


def test_a_checkpoint_survives_and_is_read_back(client):
    """A restart kills the thread but not the job row.

    Without checkpointing, a failure in the last batch discarded every batch
    before it -- the expensive ones.
    """
    import uuid as _uuid
    import app_v3
    from app_v3 import get_db
    job_id = str(_uuid.uuid4())
    with get_db(commit=True) as (_c, cur):
        cur.execute("""INSERT INTO research_pipeline_jobs
                       (id, batch_id, ticker, job_type, status, progress, current_step, total_steps, steps_detail)
                       VALUES (%s, %s, 'III', 'note', 'running', 20, 'x', 6, '{}')""",
                    (job_id, str(_uuid.uuid4())))

    assert app_v3._resume_note_checkpoint(job_id) is None
    app_v3._checkpoint_note_job(job_id, 2, 4, 'partial note text')
    cp = app_v3._resume_note_checkpoint(job_id)
    assert cp['batchesDone'] == 2 and cp['batchesTotal'] == 4
    assert cp['text'] == 'partial note text'


def test_a_checkpoint_failure_never_breaks_the_run(client):
    """Saving progress is a convenience; it must not be able to fail the job."""
    import app_v3
    app_v3._checkpoint_note_job('no-such-job-id', 1, 2, 'text')   # must not raise
