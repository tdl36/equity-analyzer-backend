"""Tests for media-tracker material gating + alert dedup (M6)."""
import json
from unittest.mock import patch

import app_v3
from media_trackers import material


# -------- candidate pre-filter ------------------------------------------

def test_candidate_short_circuits_without_ticker_or_keyword():
    point = {
        'text': 'Some generic market musing.',
        'tickers': ['ZZZZ'],
        'sector_tags': ['RANDOM'],
        'theme_tags': ['nothing interesting'],
    }
    assert material._candidate_material(point, covered=set(), keywords=set(), muted_coverage=set()) is False


def test_candidate_hits_on_covered_ticker():
    point = {'text': 'NVDA data center revenue', 'tickers': ['NVDA'], 'sector_tags': [], 'theme_tags': []}
    assert material._candidate_material(point, covered={'NVDA'}, keywords=set(), muted_coverage=set()) is True


def test_candidate_respects_coverage_mute():
    point = {'text': 'NVDA data center revenue', 'tickers': ['NVDA'], 'sector_tags': [], 'theme_tags': []}
    assert material._candidate_material(
        point, covered={'NVDA'}, keywords=set(), muted_coverage={'NVDA'}
    ) is False


def test_candidate_matches_keyword_in_theme_tags():
    point = {
        'text': 'Supply is loosening across the category.',
        'tickers': [],
        'sector_tags': [],
        'theme_tags': ['GLP-1 supply'],
    }
    assert material._candidate_material(
        point, covered=set(), keywords={'glp-1 supply'}, muted_coverage=set()
    ) is True


# -------- judge parser ---------------------------------------------------

def test_material_judge_parses_json_and_code_fence():
    fake_llm = {'text': '```json\n{"material": true, "reason": "specific data point"}\n```'}
    with patch.object(material.app_v3, 'call_llm', return_value=fake_llm):
        assert material._judge_material('LLY Zepbound pens ramping') is True


def test_material_judge_returns_false_on_non_material():
    fake_llm = {'text': '{"material": false, "reason": "generic commentary"}'}
    with patch.object(material.app_v3, 'call_llm', return_value=fake_llm):
        assert material._judge_material('AI is changing everything') is False


def test_material_judge_returns_false_on_llm_error():
    with patch.object(material.app_v3, 'call_llm', side_effect=RuntimeError('boom')):
        assert material._judge_material('anything') is False


# -------- full gate/alert pipeline --------------------------------------

def _seed_episode_with_points(clean_points=None):
    """Insert feed + episode + one digest point. Returns (episode_id, point_id)."""
    points = clean_points or [{
        'text': 'LLY guided Zepbound pen supply up 40% q/q on conference call.',
        'tickers': ['LLY'],
        'sector_tags': ['PHARMA'],
        'theme_tags': ['GLP-1 supply'],
    }]
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('f1', 'podcast', 'Odd Lots', 'https://x/rss')
            ON CONFLICT (id) DO NOTHING
        ''')
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, source_url, status)
            VALUES ('e1', 'f1', 'g-e1', 'Zepbound supply with guest Bob Smith', 'https://ep1', 'done')
            ON CONFLICT (feed_id, guid) DO NOTHING
        ''')
        point_ids = []
        for i, p in enumerate(points):
            pid = f'p{i}'
            cur.execute('''
                INSERT INTO media_digest_points
                    (id, episode_id, point_order, text, tickers, sector_tags, theme_tags)
                VALUES (%s, 'e1', %s, %s, %s, %s, %s)
            ''', (pid, i, p['text'], p['tickers'], p['sector_tags'], p['theme_tags']))
            point_ids.append(pid)
    return 'e1', point_ids


def test_gate_creates_alert_on_first_material_point(clean_db):
    _seed_episode_with_points()
    with patch.object(material, '_load_coverage_universe', return_value={'LLY'}), \
         patch.object(material, '_load_watchlist_keywords', return_value=set()), \
         patch.object(material, '_load_muted_coverage', return_value=set()), \
         patch.object(material, '_judge_material', return_value=True):
        material.gate_and_alert_for_episode('e1')

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT id, alert_type, ticker, title, detail FROM agent_alerts WHERE alert_type='podcast_material'")
        alerts = cur.fetchall()
        cur.execute("SELECT material FROM media_digest_points WHERE episode_id='e1'")
        pts = cur.fetchall()

    assert len(alerts) == 1
    assert alerts[0]['ticker'] == 'LLY'
    assert 'Zepbound' in alerts[0]['title']
    detail = alerts[0]['detail'] if isinstance(alerts[0]['detail'], dict) else json.loads(alerts[0]['detail'])
    assert detail['episodeId'] == 'e1'
    assert detail['relatedEpisodes'] == []
    assert pts[0]['material'] is True


def test_gate_skips_when_judge_says_non_material(clean_db):
    _seed_episode_with_points()
    with patch.object(material, '_load_coverage_universe', return_value={'LLY'}), \
         patch.object(material, '_load_watchlist_keywords', return_value=set()), \
         patch.object(material, '_load_muted_coverage', return_value=set()), \
         patch.object(material, '_judge_material', return_value=False):
        material.gate_and_alert_for_episode('e1')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM agent_alerts")
        assert cur.fetchone()['n'] == 0


def test_gate_dedupes_on_same_ticker_and_theme_within_7d(clean_db):
    # Seed an existing alert for LLY with matching theme 'GLP-1 supply'
    existing_detail = {
        'pointId': 'old-point',
        'episodeId': 'old-ep',
        'episodeTitle': 'Old Zepbound episode',
        'feedName': 'Odd Lots',
        'feedId': 'f1',
        'sourceUrl': 'https://old',
        'tickers': ['LLY'],
        'sectorTags': ['PHARMA'],
        'themeTags': ['GLP-1 supply'],
        'relatedEpisodes': [],
    }
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO agent_alerts (id, alert_type, ticker, title, detail, status, created_at)
            VALUES ('a-existing', 'podcast_material', 'LLY', 'Old Zepbound alert', %s::jsonb, 'new', NOW() - INTERVAL '1 day')
        ''', (json.dumps(existing_detail),))

    _seed_episode_with_points()
    with patch.object(material, '_load_coverage_universe', return_value={'LLY'}), \
         patch.object(material, '_load_watchlist_keywords', return_value=set()), \
         patch.object(material, '_load_muted_coverage', return_value=set()), \
         patch.object(material, '_judge_material', return_value=True):
        material.gate_and_alert_for_episode('e1')

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT id, detail FROM agent_alerts WHERE alert_type='podcast_material' ORDER BY created_at")
        rows = cur.fetchall()
    # Still only one alert (dedup'd)
    assert len(rows) == 1
    assert rows[0]['id'] == 'a-existing'
    det = rows[0]['detail'] if isinstance(rows[0]['detail'], dict) else json.loads(rows[0]['detail'])
    assert len(det['relatedEpisodes']) == 1
    assert det['relatedEpisodes'][0]['episodeId'] == 'e1'


def test_gate_creates_separate_alert_for_different_ticker(clean_db):
    # Existing alert is for NVDA; new episode is LLY — no dedup.
    existing_detail = {
        'pointId': 'old', 'episodeId': 'old', 'episodeTitle': 'Old', 'feedName': 'X',
        'feedId': 'f1', 'sourceUrl': 'u', 'tickers': ['NVDA'], 'sectorTags': [],
        'themeTags': ['GLP-1 supply'], 'relatedEpisodes': [],
    }
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO agent_alerts (id, alert_type, ticker, title, detail, status, created_at)
            VALUES ('a-nvda', 'podcast_material', 'NVDA', 'NVDA thing', %s::jsonb, 'new', NOW())
        ''', (json.dumps(existing_detail),))

    _seed_episode_with_points()
    with patch.object(material, '_load_coverage_universe', return_value={'LLY'}), \
         patch.object(material, '_load_watchlist_keywords', return_value=set()), \
         patch.object(material, '_load_muted_coverage', return_value=set()), \
         patch.object(material, '_judge_material', return_value=True):
        material.gate_and_alert_for_episode('e1')

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM agent_alerts WHERE alert_type='podcast_material'")
        assert cur.fetchone()['n'] == 2
