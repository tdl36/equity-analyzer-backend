"""Tests for the transcription pipeline (scraper + Gemini fallback)."""
from unittest.mock import patch

import app_v3
from media_trackers import transcribe


def _seed_episode(episode_id='e1', status='new', audio_url=None, source_url=None,
                  duration_sec=None, show_notes='some notes'):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('f1', 'podcast', 'Test Pod', 'https://x/rss')
            ON CONFLICT (id) DO NOTHING
        ''')
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, audio_url, source_url,
                                        show_notes, duration_sec, status)
            VALUES (%s, 'f1', %s, 't', %s, %s, %s, %s, %s)
            ON CONFLICT (feed_id, guid) DO NOTHING
        ''', (episode_id, f'g-{episode_id}', audio_url, source_url,
              show_notes, duration_sec, status))
    return episode_id


def test_transcribe_episode_uses_scraper_when_available(clean_db):
    _seed_episode(source_url='https://joincolossus.com/episodes/abc')
    with patch.object(transcribe, 'try_publisher_scrape',
                      return_value=('SCRAPED TRANSCRIPT TEXT', 'publisher')):
        transcribe.transcribe_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT transcript, transcript_source, cost_usd, status FROM media_episodes WHERE id='e1'")
        row = cur.fetchone()
    assert row['transcript'] == 'SCRAPED TRANSCRIPT TEXT'
    assert row['transcript_source'] == 'publisher'
    assert float(row['cost_usd']) == 0.0
    assert row['status'] == 'extracting'


def test_transcribe_episode_falls_back_to_show_notes_when_no_audio(clean_db):
    _seed_episode(source_url=None, audio_url=None)
    transcribe.transcribe_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT transcript, transcript_source, status FROM media_episodes WHERE id='e1'")
        row = cur.fetchone()
    assert row['status'] == 'extracting'
    assert row['transcript_source'] == 'show_notes_only'
    # transcript column was not overwritten (COALESCE with NULL keeps existing NULL).
    assert row['transcript'] is None


def test_transcribe_skips_if_duration_over_2hr(clean_db):
    _seed_episode(audio_url='https://x/a.mp3', duration_sec=7201)
    transcribe.transcribe_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status, error_message FROM media_episodes WHERE id='e1'")
        row = cur.fetchone()
    assert row['status'] == 'skipped'
    assert '7201' in (row['error_message'] or '') or 'cap' in (row['error_message'] or '').lower()


def test_process_transcribe_batch_picks_up_to_3(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('f1','podcast','X','u') ON CONFLICT DO NOTHING
        ''')
        for i in range(5):
            cur.execute('''
                INSERT INTO media_episodes (id, feed_id, guid, title, show_notes, status)
                VALUES (%s, 'f1', %s, 't', 'notes', 'new')
            ''', (f'e{i}', f'g{i}'))

    calls = []

    def fake_transcribe(eid):
        calls.append(eid)

    with patch.object(transcribe, 'transcribe_episode', side_effect=fake_transcribe):
        transcribe.process_transcribe_batch()

    assert len(calls) == 3
