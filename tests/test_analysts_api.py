"""Tests for Phase 3a analyst-team API + podcast-material routing."""
import json
from unittest.mock import patch

import app_v3
from media_trackers import material


# -------- basic CRUD -----------------------------------------------------

def test_create_and_list_analyst(client, clean_db):
    resp = client.post('/api/analysts', json={
        'name': 'Test Pharma Analyst',
        'sector': 'Healthcare',
        'subsector': 'Large Cap Pharma',
        'coverageTickers': ['lly', 'JNJ', 'BMY'],
    })
    assert resp.status_code == 201
    body = resp.get_json()
    analyst = body['analyst']
    assert analyst['name'] == 'Test Pharma Analyst'
    assert analyst['sector'] == 'Healthcare'
    assert set(analyst['coverageTickers']) == {'LLY', 'JNJ', 'BMY'}
    assert analyst['pendingCount'] == 0

    # List
    list_resp = client.get('/api/analysts')
    assert list_resp.status_code == 200
    listed = list_resp.get_json()
    assert listed['total'] == 1
    assert listed['analysts'][0]['id'] == analyst['id']
    assert listed['analysts'][0]['pendingCount'] == 0


def test_patch_analyst_coverage(client, clean_db):
    create = client.post('/api/analysts', json={
        'name': 'Patchable', 'coverageTickers': ['NVDA']
    }).get_json()
    aid = create['analyst']['id']
    resp = client.patch(f'/api/analysts/{aid}', json={
        'coverageTickers': ['NVDA', 'TSM', 'AMAT'],
        'subsector': 'Semis',
    })
    assert resp.status_code == 200
    body = resp.get_json()['analyst']
    assert set(body['coverageTickers']) == {'NVDA', 'TSM', 'AMAT'}
    assert body['subsector'] == 'Semis'


def test_delete_analyst_cascades_activities(client, clean_db):
    aid = client.post('/api/analysts', json={
        'name': 'Cascader', 'coverageTickers': ['NVDA']
    }).get_json()['analyst']['id']
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO analyst_activities (id, analyst_id, activity_type, ticker,
                                            status, trigger_source, input)
            VALUES ('act-1', %s, 'investigation', 'NVDA', 'pending_review',
                    'podcast_material', '{}'::jsonb)
        ''', (aid,))

    resp = client.delete(f'/api/analysts/{aid}')
    assert resp.status_code == 200
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM analyst_activities WHERE id='act-1'")
        assert cur.fetchone()['n'] == 0


# -------- podcast-material routing --------------------------------------

def _seed_episode_with_lly_point():
    """Insert feed + episode + one LLY digest point. Returns ids."""
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO media_feeds (id, source_type, name, feed_url)
            VALUES ('f1', 'podcast', 'Odd Lots', 'https://x/rss')
            ON CONFLICT (id) DO NOTHING
        ''')
        cur.execute('''
            INSERT INTO media_episodes (id, feed_id, guid, title, source_url, status)
            VALUES ('ep1', 'f1', 'g-ep1', 'Zepbound supply', 'https://ep1', 'done')
            ON CONFLICT (feed_id, guid) DO NOTHING
        ''')
        cur.execute('''
            INSERT INTO media_digest_points
                (id, episode_id, point_order, text, tickers, sector_tags, theme_tags)
            VALUES ('p0', 'ep1', 0, 'LLY Zepbound supply up 40%.',
                    ARRAY['LLY']::text[], ARRAY['PHARMA']::text[],
                    ARRAY['GLP-1 supply']::text[])
        ''')


def test_analyst_coverage_routing(client, clean_db):
    # Create an NVDA analyst and an LLY analyst.
    nvda_aid = client.post('/api/analysts', json={
        'name': 'NVDA Watcher', 'coverageTickers': ['NVDA'],
    }).get_json()['analyst']['id']
    lly_aid = client.post('/api/analysts', json={
        'name': 'Pharma Analyst', 'coverageTickers': ['LLY', 'JNJ'],
    }).get_json()['analyst']['id']

    _seed_episode_with_lly_point()

    with patch.object(material, '_load_coverage_universe', return_value={'LLY'}), \
         patch.object(material, '_load_watchlist_keywords', return_value=set()), \
         patch.object(material, '_load_muted_coverage', return_value=set()), \
         patch.object(material, '_judge_material', return_value=True):
        material.gate_and_alert_for_episode('ep1')

    # Only the LLY analyst should have a pending activity.
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT analyst_id, ticker, activity_type, status, trigger_source
              FROM analyst_activities
             ORDER BY analyst_id
        ''')
        rows = cur.fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row['analyst_id'] == lly_aid
    assert row['ticker'] == 'LLY'
    assert row['activity_type'] == 'investigation'
    assert row['status'] == 'pending_review'
    assert row['trigger_source'] == 'podcast_material'
    # Sanity: NVDA analyst untouched
    assert row['analyst_id'] != nvda_aid


# -------- seed + inbox + approve ----------------------------------------

def test_seed_default_roster_is_idempotent(client, clean_db):
    r1 = client.post('/api/analysts/seed-default-roster').get_json()
    assert r1['seeded'] > 0
    assert r1['skipped'] is False
    r2 = client.post('/api/analysts/seed-default-roster').get_json()
    assert r2['skipped'] is True
    assert r2['seeded'] == 0
    # Expect at least the 8 groups we hard-coded
    listed = client.get('/api/analysts').get_json()
    assert listed['total'] >= 8


def test_pending_inbox_endpoint(client, clean_db):
    aid = client.post('/api/analysts', json={
        'name': 'Inbox Analyst', 'coverageTickers': ['NVDA'],
    }).get_json()['analyst']['id']
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO analyst_activities (id, analyst_id, activity_type, ticker,
                                            status, trigger_source, input)
            VALUES ('act-pending', %s, 'investigation', 'NVDA',
                    'pending_review', 'podcast_material',
                    '{"pointText":"NVDA Blackwell shipments"}'::jsonb),
                   ('act-done', %s, 'investigation', 'NVDA',
                    'approved', 'podcast_material', '{}'::jsonb)
        ''', (aid, aid))

    resp = client.get('/api/analyst-activities/pending')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['total'] == 1
    item = body['activities'][0]
    assert item['id'] == 'act-pending'
    assert item['analystName'] == 'Inbox Analyst'
    assert item['input']['pointText'] == 'NVDA Blackwell shipments'


def test_approve_activity(client, clean_db):
    aid = client.post('/api/analysts', json={
        'name': 'Approver', 'coverageTickers': ['NVDA'],
    }).get_json()['analyst']['id']
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO analyst_activities (id, analyst_id, activity_type, ticker,
                                            status, trigger_source, input)
            VALUES ('act-x', %s, 'investigation', 'NVDA',
                    'pending_review', 'podcast_material', '{}'::jsonb)
        ''', (aid,))

    resp = client.patch('/api/analyst-activities/act-x', json={
        'status': 'approved',
        'output': {'note': 'reviewed'},
    })
    assert resp.status_code == 200
    body = resp.get_json()['activity']
    assert body['status'] == 'approved'
    assert body['output']['note'] == 'reviewed'
    assert body['reviewedAt'] is not None

    # No longer appears in inbox
    pending = client.get('/api/analyst-activities/pending').get_json()
    assert pending['total'] == 0
