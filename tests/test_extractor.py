"""Tests for the show-notes extractor."""
from pathlib import Path
from unittest.mock import patch
import app_v3
from media_trackers import extractor

FIXTURE = (Path(__file__).parent / 'fixtures' / 'shownotes_sample.txt').read_text()


def _seed_episode(show_notes=FIXTURE, status='extracting', episode_id='e1'):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('f1', 'podcast', 'Odd Lots', 'https://x/rss')
            ON CONFLICT (id) DO NOTHING
        ''')
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, show_notes, status)
            VALUES (%s, 'f1', %s, 'Zepbound supply', %s, %s)
            ON CONFLICT (feed_id, guid) DO NOTHING
        ''', (episode_id, f'g-{episode_id}', show_notes, status))
    return episode_id


FAKE_LLM_RESPONSE = {
    'points': [
        {'text': 'LLY is shifting Zepbound from vials to pen injectors.',
         'tickers': ['LLY'], 'sector_tags': ['PHARMA'], 'theme_tags': ['GLP-1']},
        {'text': 'NVDA H200 allocations going to hyperscalers first.',
         'tickers': ['NVDA'], 'sector_tags': ['SEMIS'], 'theme_tags': ['AI']},
        {'text': 'TSM comments signal broader semis cycle tightening.',
         'tickers': ['TSM'], 'sector_tags': ['SEMIS'], 'theme_tags': []},
    ]
}


def test_extract_from_episode_inserts_points(clean_db):
    _seed_episode()
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.extract_from_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT text, tickers, sector_tags FROM media_digest_points WHERE episode_id='e1' ORDER BY point_order")
        rows = cur.fetchall()
    assert len(rows) == 3
    assert rows[0]['tickers'] == ['LLY']
    assert rows[1]['sector_tags'] == ['SEMIS']


def test_extract_marks_episode_done(clean_db):
    _seed_episode()
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.extract_from_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status FROM media_episodes WHERE id='e1'")
        assert cur.fetchone()['status'] == 'done'


def test_process_extract_batch_respects_max(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('f1','podcast','X','u') ON CONFLICT DO NOTHING
        ''')
        for i in range(15):
            cur.execute('''
                INSERT INTO media_episodes (id, feed_id, guid, title, show_notes, status)
                VALUES (%s, 'f1', %s, 't', %s, 'extracting')
            ''', (f'e{i}', f'g{i}', FIXTURE))
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.process_extract_batch()
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_episodes WHERE status='done'")
        assert cur.fetchone()['n'] == extractor.MAX_EPISODES_PER_BATCH


def test_extract_skips_episode_with_no_show_notes(clean_db):
    _seed_episode(show_notes=None)
    with patch.object(extractor, '_call_haiku', return_value=FAKE_LLM_RESPONSE):
        extractor.extract_from_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status, error_message FROM media_episodes WHERE id='e1'")
        row = cur.fetchone()
    assert row['status'] == 'skipped'
    assert 'no content' in row['error_message'].lower()


# _parse_json unit tests — real call_llm shape + code-fence handling
def test_parse_json_handles_call_llm_dict_with_code_fence():
    raw = {
        'text': '```json\n{"points": [{"text": "Test", "tickers": ["NVDA"]}]}\n```',
        'usage': {'input_tokens': 10, 'output_tokens': 20},
        'provider': 'anthropic',
        'model': 'claude-haiku-4-5-20251001',
    }
    result = extractor._parse_json(raw)
    assert result == {'points': [{'text': 'Test', 'tickers': ['NVDA']}]}


def test_parse_json_handles_plain_json_string():
    result = extractor._parse_json('{"points": []}')
    assert result == {'points': []}


def test_parse_json_handles_bare_code_fence_no_lang():
    result = extractor._parse_json('```\n{"points": [{"text": "X"}]}\n```')
    assert result == {'points': [{'text': 'X'}]}


def test_parse_json_handles_already_parsed_dict():
    result = extractor._parse_json({'points': [{'text': 'already'}]})
    assert result == {'points': [{'text': 'already'}]}


def test_parse_json_returns_empty_on_junk():
    result = extractor._parse_json('not json at all')
    assert result == {'points': []}
