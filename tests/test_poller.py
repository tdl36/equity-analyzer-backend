"""Tests for the RSS poller."""
from pathlib import Path
import pytest
import responses
import app_v3
from media_trackers import poller

FIXTURE = Path(__file__).parent / 'fixtures' / 'odd_lots_sample.xml'


def _create_feed(url='https://fake.example/rss', name='Odd Lots'):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('feed-1', 'podcast', %s, %s) RETURNING id
        ''', (name, url))
        return cur.fetchone()['id']


@responses.activate
def test_poll_feed_inserts_episodes(clean_db):
    feed_id = _create_feed()
    responses.add(responses.GET, 'https://fake.example/rss',
                  body=FIXTURE.read_text(), status=200,
                  content_type='application/rss+xml')
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT guid, title FROM media_episodes WHERE feed_id=%s ORDER BY guid", (feed_id,))
        rows = cur.fetchall()
    guids = {r['guid'] for r in rows}
    assert guids == {'odd-lots-2026-04-20', 'odd-lots-2026-04-18'}


@responses.activate
def test_poll_feed_is_idempotent(clean_db):
    feed_id = _create_feed()
    responses.add(responses.GET, 'https://fake.example/rss',
                  body=FIXTURE.read_text(), status=200)
    poller.poll_feed(feed_id)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_episodes WHERE feed_id=%s", (feed_id,))
        assert cur.fetchone()['n'] == 2


@responses.activate
def test_poll_feed_records_error_on_404(clean_db):
    feed_id = _create_feed()
    responses.add(responses.GET, 'https://fake.example/rss', status=404)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT error_count, last_error FROM media_feeds WHERE id=%s", (feed_id,))
        row = cur.fetchone()
    assert row['error_count'] == 1
    assert row['last_error']


@responses.activate
def test_poll_feed_auto_mutes_after_5_errors(clean_db):
    feed_id = _create_feed()
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET error_count=4 WHERE id=%s", (feed_id,))
    responses.add(responses.GET, 'https://fake.example/rss', status=500)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT muted FROM media_feeds WHERE id=%s", (feed_id,))
        assert cur.fetchone()['muted'] is True


@responses.activate
def test_poll_feed_skips_episodes_older_than_last_episode_at(clean_db):
    feed_id = _create_feed()
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET last_episode_at='2026-04-19'::timestamp WHERE id=%s", (feed_id,))
    responses.add(responses.GET, 'https://fake.example/rss', body=FIXTURE.read_text(), status=200)
    poller.poll_feed(feed_id)
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT guid FROM media_episodes WHERE feed_id=%s", (feed_id,))
        guids = {r['guid'] for r in cur.fetchall()}
    assert guids == {'odd-lots-2026-04-20'}


def test_poll_all_feeds_iterates(clean_db, monkeypatch):
    _create_feed(url='https://a.example/rss', name='A')
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('feed-2', 'podcast', 'B', 'https://b.example/rss')
        ''')
    calls = []
    monkeypatch.setattr(poller, 'poll_feed', lambda fid: calls.append(fid))
    poller.poll_all_feeds()
    assert set(calls) == {'feed-1', 'feed-2'}


def test_poll_all_feeds_skips_muted(clean_db, monkeypatch):
    _create_feed(url='https://a.example/rss', name='A')
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET muted=TRUE WHERE id='feed-1'")
    calls = []
    monkeypatch.setattr(poller, 'poll_feed', lambda fid: calls.append(fid))
    poller.poll_all_feeds()
    assert calls == []
