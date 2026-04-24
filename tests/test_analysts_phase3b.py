"""Tests for Phase 3b: CATALYSTS folder drop -> analyst inbox routing."""
import app_v3


def _create_analyst(client, name, tickers):
    return client.post('/api/analysts', json={
        'name': name,
        'coverageTickers': tickers,
    }).get_json()['analyst']['id']


def _count_activities(ticker=None, trigger='catalyst_folder'):
    with app_v3.get_db() as (_c, cur):
        if ticker:
            cur.execute('''
                SELECT COUNT(*) AS n FROM analyst_activities
                 WHERE ticker=%s AND trigger_source=%s
            ''', (ticker, trigger))
        else:
            cur.execute('''
                SELECT COUNT(*) AS n FROM analyst_activities
                 WHERE trigger_source=%s
            ''', (trigger,))
        return int(cur.fetchone()['n'])


def test_queue_catalyst_activity_classifies_earnings(client, clean_db):
    aid = _create_analyst(client, 'DOV Watcher', ['DOV'])
    resp = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'DOV',
        'topic': 'DOV 1Q26 Earnings',
        'fingerprint': 'fp-1',
        'fileCount': 3,
        'catalystJobId': 'job-abc',
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['count'] == 1
    assert body['activityType'] == 'earnings_recap'

    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT activity_type, ticker, status, trigger_source, input
              FROM analyst_activities
             WHERE analyst_id=%s
        ''', (aid,))
        rows = cur.fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row['activity_type'] == 'earnings_recap'
    assert row['ticker'] == 'DOV'
    assert row['status'] == 'pending_review'
    assert row['trigger_source'] == 'catalyst_folder'
    inp = row['input']
    assert inp['topic'] == 'DOV 1Q26 Earnings'
    assert inp['fingerprint'] == 'fp-1'
    assert inp['fileCount'] == 3
    assert inp['catalystJobId'] == 'job-abc'


def test_queue_catalyst_activity_classifies_takeaway(client, clean_db):
    _create_analyst(client, 'Generic Watcher', ['AAPL'])
    resp = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'AAPL',
        'topic': 'Random Update',
        'fingerprint': 'fp-t',
        'fileCount': 1,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['count'] == 1
    assert body['activityType'] == 'takeaway'

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT activity_type FROM analyst_activities WHERE ticker='AAPL'")
        r = cur.fetchone()
    assert r['activity_type'] == 'takeaway'


def test_queue_catalyst_activity_classifies_earnings_keyword(client, clean_db):
    # Covers case-insensitive 'earnings' match without a quarter pattern
    _create_analyst(client, 'GE Watcher', ['GE'])
    body = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'GE',
        'topic': 'GE Earnings Preview',
        'fingerprint': 'fp-ge',
        'fileCount': 2,
    }).get_json()
    assert body['activityType'] == 'earnings_recap'


def test_queue_catalyst_dedups_within_24h(client, clean_db):
    _create_analyst(client, 'Dedup Watcher', ['NVDA'])
    payload = {
        'ticker': 'NVDA',
        'topic': 'NVDA 2Q26',
        'fingerprint': 'fp-dedup',
        'fileCount': 4,
    }
    r1 = client.post('/api/analysts/queue-catalyst-activity', json=payload).get_json()
    assert r1['count'] == 1
    # Same payload again -> deduped (zero created)
    r2 = client.post('/api/analysts/queue-catalyst-activity', json=payload).get_json()
    assert r2['count'] == 0
    assert _count_activities(ticker='NVDA') == 1


def test_queue_catalyst_new_fingerprint_creates_new_row(client, clean_db):
    _create_analyst(client, 'NewFP Watcher', ['MSFT'])
    client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'MSFT', 'topic': 'MSFT 3Q26',
        'fingerprint': 'fp-v1', 'fileCount': 1,
    })
    # Same topic, new fingerprint -> new row
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'MSFT', 'topic': 'MSFT 3Q26',
        'fingerprint': 'fp-v2', 'fileCount': 2,
    }).get_json()
    assert r['count'] == 1
    assert _count_activities(ticker='MSFT') == 2


def test_queue_catalyst_no_covering_analyst(client, clean_db):
    # No analyst covers ZZZZ -> count=0, no rows
    _create_analyst(client, 'NVDA only', ['NVDA'])
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'ZZZZ',
        'topic': 'ZZZZ 4Q26',
        'fingerprint': 'fp-none',
        'fileCount': 1,
    }).get_json()
    assert r['count'] == 0
    assert _count_activities(ticker='ZZZZ') == 0


def test_queue_catalyst_routes_to_multiple_analysts(client, clean_db):
    a1 = _create_analyst(client, 'Analyst A', ['ORCL'])
    a2 = _create_analyst(client, 'Analyst B', ['ORCL', 'SAP'])
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'ORCL',
        'topic': 'ORCL 1Q26 Earnings',
        'fingerprint': 'fp-multi',
        'fileCount': 5,
        'catalystJobId': 'job-multi',
    }).get_json()
    assert r['count'] == 2
    assert r['activityType'] == 'earnings_recap'
    analyst_ids = {c['analystId'] for c in r['created']}
    assert analyst_ids == {a1, a2}

    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT analyst_id, activity_type, trigger_source
              FROM analyst_activities
             WHERE ticker='ORCL'
             ORDER BY analyst_id
        ''')
        rows = cur.fetchall()
    assert len(rows) == 2
    assert all(row['activity_type'] == 'earnings_recap' for row in rows)
    assert all(row['trigger_source'] == 'catalyst_folder' for row in rows)


def test_queue_catalyst_missing_fields_400(client, clean_db):
    r1 = client.post('/api/analysts/queue-catalyst-activity', json={
        'topic': 'something', 'fingerprint': 'fp',
    })
    assert r1.status_code == 400
    r2 = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'NVDA', 'fingerprint': 'fp',
    })
    assert r2.status_code == 400


def test_queue_catalyst_uppercases_ticker(client, clean_db):
    _create_analyst(client, 'Case Test', ['TSM'])
    r = client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'tsm',
        'topic': 'TSM 2Q26 Earnings',
        'fingerprint': 'fp-case',
        'fileCount': 1,
    }).get_json()
    assert r['count'] == 1
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT ticker FROM analyst_activities WHERE trigger_source='catalyst_folder'")
        rows = cur.fetchall()
    assert all(row['ticker'] == 'TSM' for row in rows)
