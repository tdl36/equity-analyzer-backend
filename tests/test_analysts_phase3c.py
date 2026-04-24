"""Phase 3c: earnings recap workflow — /api/analyst-activities/<id>/run +
link-back from pipeline job result to activity."""
import json

import app_v3


def _create_analyst(client, tickers):
    return client.post('/api/analysts', json={
        'name': 'MDT Analyst',
        'coverageTickers': tickers,
    }).get_json()['analyst']['id']


def _queue_earnings_activity(client, ticker='MDT', topic='4Q26 Earnings'):
    client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': ticker,
        'topic': topic,
        'fingerprint': 'fp1',
        'fileCount': 3,
        'catalystJobId': 'initial-job-id',
    })
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM analyst_activities
             WHERE ticker=%s AND activity_type='earnings_recap'
             ORDER BY created_at DESC LIMIT 1
        ''', (ticker,))
        row = cur.fetchone()
    return row['id']


def test_run_endpoint_creates_synthesis_job_with_earnings_variant(client, clean_db):
    _create_analyst(client, ['MDT'])
    activity_id = _queue_earnings_activity(client)

    resp = client.post(f'/api/analyst-activities/{activity_id}/run', json={'length': 'standard'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ticker'] == 'MDT'
    assert body['promptVariant'] == 'earnings_recap'
    catalyst_job_id = body['catalystJobId']

    # Verify pipeline job was created with the right variant
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT job_type, status, steps_detail FROM research_pipeline_jobs WHERE id=%s", (catalyst_job_id,))
        row = cur.fetchone()
    assert row is not None
    assert row['job_type'] == 'synthesis'
    assert row['status'] == 'queued'
    sd = row['steps_detail']
    if isinstance(sd, str):
        sd = json.loads(sd)
    assert sd['prompt_variant'] == 'earnings_recap'
    assert sd['activityId'] == activity_id

    # Activity should be marked running with catalystJobId in output
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT status, output FROM analyst_activities WHERE id=%s', (activity_id,))
        act = cur.fetchone()
    out = act['output']
    if isinstance(out, str):
        out = json.loads(out)
    assert act['status'] == 'running'
    assert out['catalystJobId'] == catalyst_job_id
    assert out['promptVariant'] == 'earnings_recap'


def test_run_endpoint_rejects_if_activity_not_pending(client, clean_db):
    _create_analyst(client, ['MDT'])
    activity_id = _queue_earnings_activity(client)
    # Flip to approved
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE analyst_activities SET status='approved' WHERE id=%s", (activity_id,))
    r = client.post(f'/api/analyst-activities/{activity_id}/run', json={})
    assert r.status_code == 400


def test_run_endpoint_handles_unknown_activity(client, clean_db):
    r = client.post('/api/analyst-activities/nope/run', json={})
    assert r.status_code == 404


def test_takeaway_activity_uses_catalyst_prompt_variant(client, clean_db):
    _create_analyst(client, ['AAPL'])
    client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'AAPL', 'topic': 'WWDC 2026 takeaways', 'fingerprint': 'fp1', 'fileCount': 1,
    })
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT id FROM analyst_activities WHERE ticker='AAPL' ORDER BY created_at DESC LIMIT 1")
        act_id = cur.fetchone()['id']

    r = client.post(f'/api/analyst-activities/{act_id}/run', json={}).get_json()
    assert r['promptVariant'] == 'catalyst'


def test_agent_update_job_propagates_to_activity_on_complete(client, clean_db):
    _create_analyst(client, ['MDT'])
    activity_id = _queue_earnings_activity(client)
    run_body = client.post(f'/api/analyst-activities/{activity_id}/run', json={}).get_json()
    job_id = run_body['catalystJobId']

    # Simulate local agent reporting completion
    client.post('/api/agent/update-job', json={
        'jobId': job_id,
        'status': 'complete',
        'currentStep': 'Complete',
        'progress': 100,
        'result': {
            'markdown': '# MDT 4Q26\n\n### 1. Results vs. Expectations\nRev beat by 2%.',
            'sourceFiles': ['press_release.pdf', 'transcript.pdf'],
            'fileCount': 2,
        },
    })

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT status, output FROM analyst_activities WHERE id=%s', (activity_id,))
        row = cur.fetchone()
    out = row['output']
    if isinstance(out, str):
        out = json.loads(out)
    assert row['status'] == 'pending_review'
    assert out['synthesisMarkdown'].startswith('# MDT 4Q26')
    assert out['sourceFiles'] == ['press_release.pdf', 'transcript.pdf']


def test_agent_update_job_propagates_failure_to_activity(client, clean_db):
    _create_analyst(client, ['MDT'])
    activity_id = _queue_earnings_activity(client)
    run_body = client.post(f'/api/analyst-activities/{activity_id}/run', json={}).get_json()
    job_id = run_body['catalystJobId']

    client.post('/api/agent/update-job', json={
        'jobId': job_id,
        'status': 'failed',
        'error': 'Anthropic API transient error',
        'progress': 50,
    })

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT status, error FROM analyst_activities WHERE id=%s', (activity_id,))
        row = cur.fetchone()
    assert row['status'] == 'failed'
    assert 'transient' in (row['error'] or '').lower()


def test_approve_saves_to_research_and_queues_icloud(client, clean_db):
    _create_analyst(client, ['MDT'])
    activity_id = _queue_earnings_activity(client)
    run_body = client.post(f'/api/analyst-activities/{activity_id}/run', json={}).get_json()
    job_id = run_body['catalystJobId']
    client.post('/api/agent/update-job', json={
        'jobId': job_id, 'status': 'complete',
        'result': {'markdown': '# MDT 4Q26 Earnings Recap\n\n### 1. Results vs. Expectations\nRev beat.'},
    })

    # Approve
    r = client.post(f'/api/analyst-activities/{activity_id}/approve', json={})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['savedTo']['research'] is not None
    assert body['savedTo']['research']['docId'] == f'earnings-recap-{activity_id}'
    assert body['savedTo']['icloud'] is not None
    assert body['savedTo']['icloud']['path'].startswith('CATALYSTS/MDT/')

    # Status flipped
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT status, reviewed_at FROM analyst_activities WHERE id=%s', (activity_id,))
        row = cur.fetchone()
    assert row['status'] == 'approved'
    assert row['reviewed_at'] is not None

    # Research document created
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT name, content FROM research_documents WHERE id=%s', (f'earnings-recap-{activity_id}',))
        doc = cur.fetchone()
    assert doc is not None
    assert 'MDT' in doc['name']
    assert 'Earnings Recap' in doc['name']
    assert doc['content'].startswith('# MDT 4Q26 Earnings Recap')

    # iCloud sync queued
    syncs = client.get('/api/agent/sync-to-local').get_json().get('syncs', [])
    assert any(s.get('kind') == 'earnings_recap' and s.get('ticker') == 'MDT' for s in syncs)


def test_archived_endpoint_returns_approved_activities(client, clean_db):
    _create_analyst(client, ['MDT'])
    activity_id = _queue_earnings_activity(client)
    run_body = client.post(f'/api/analyst-activities/{activity_id}/run', json={}).get_json()
    client.post('/api/agent/update-job', json={
        'jobId': run_body['catalystJobId'], 'status': 'complete',
        'result': {'markdown': '# recap'},
    })
    client.post(f'/api/analyst-activities/{activity_id}/approve', json={})

    body = client.get('/api/analyst-activities/archived').get_json()
    ids = [a['id'] for a in body['activities']]
    assert activity_id in ids
    approved = [a for a in body['activities'] if a['id'] == activity_id][0]
    assert approved['status'] == 'approved'


def test_auto_mode_dispatches_earnings_recap_immediately(client, clean_db):
    aid = _create_analyst(client, ['MDT'])
    # Flip auto_mode on
    client.patch(f'/api/analysts/{aid}', json={'autoMode': {'enabled': True, 'expires_at': None}})

    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'MDT',
        'topic': '4Q26 Earnings',
        'fingerprint': 'fp-auto',
        'fileCount': 3,
        'catalystJobId': 'j1',
    })
    body = r.get_json()
    assert body['count'] == 1
    assert body['created'][0]['autoRan'] is True

    # Activity should already be running
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status, output FROM analyst_activities WHERE ticker='MDT'")
        row = cur.fetchone()
    assert row['status'] == 'running'
    out = row['output']
    if isinstance(out, str):
        out = json.loads(out)
    assert out.get('catalystJobId')
    assert out.get('promptVariant') == 'earnings_recap'


def test_auto_mode_off_leaves_activity_pending(client, clean_db):
    aid = _create_analyst(client, ['MDT'])
    # Default auto_mode is off
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'MDT', 'topic': '4Q26 Earnings', 'fingerprint': 'fp-off', 'fileCount': 1,
    })
    body = r.get_json()
    assert body['created'][0]['autoRan'] is False

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status FROM analyst_activities WHERE ticker='MDT'")
        assert cur.fetchone()['status'] == 'pending_review'


def test_legacy_enabled_toggle_only_auto_runs_earnings_not_takeaways(client, clean_db):
    aid = _create_analyst(client, ['AAPL'])
    # Legacy auto_mode with just {enabled: true} -> maps to earnings only
    client.patch(f'/api/analysts/{aid}', json={'autoMode': {'enabled': True, 'expires_at': None}})
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'AAPL', 'topic': 'WWDC takeaways', 'fingerprint': 'fp-x', 'fileCount': 1,
    })
    body = r.get_json()
    assert body['created'][0]['autoRan'] is False
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status, activity_type FROM analyst_activities WHERE ticker='AAPL'")
        row = cur.fetchone()
    assert row['activity_type'] == 'takeaway'
    assert row['status'] == 'pending_review'


def test_auto_mode_takeaways_flag_auto_runs_takeaway(client, clean_db):
    aid = _create_analyst(client, ['AAPL'])
    # Explicitly enable takeaways auto-run
    client.patch(f'/api/analysts/{aid}', json={'autoMode': {'earnings': False, 'takeaways': True, 'expires_at': None}})
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'AAPL', 'topic': 'WWDC takeaways', 'fingerprint': 'fp-tk', 'fileCount': 1,
    })
    body = r.get_json()
    assert body['created'][0]['autoRan'] is True
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status FROM analyst_activities WHERE ticker='AAPL'")
        assert cur.fetchone()['status'] == 'running'


def test_auto_mode_takeaways_does_not_auto_run_earnings(client, clean_db):
    aid = _create_analyst(client, ['MDT'])
    # Takeaways on, earnings off -> earnings stays pending
    client.patch(f'/api/analysts/{aid}', json={'autoMode': {'earnings': False, 'takeaways': True, 'expires_at': None}})
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'MDT', 'topic': '1Q26 Earnings', 'fingerprint': 'fp-e', 'fileCount': 1,
    })
    assert r.get_json()['created'][0]['autoRan'] is False
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status FROM analyst_activities WHERE ticker='MDT'")
        assert cur.fetchone()['status'] == 'pending_review'


def test_pending_inbox_includes_running_activities(client, clean_db):
    _create_analyst(client, ['MDT'])
    activity_id = _queue_earnings_activity(client)
    client.post(f'/api/analyst-activities/{activity_id}/run', json={})

    body = client.get('/api/analyst-activities/pending').get_json()
    ids = [a['id'] for a in body['activities']]
    assert activity_id in ids
    running = [a for a in body['activities'] if a['id'] == activity_id][0]
    assert running['status'] == 'running'
