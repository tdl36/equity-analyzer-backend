"""Phase 3d: earnings_calendar + ticker_earnings_config endpoints +
check_due_earnings dispatcher."""
import json
from datetime import date, timedelta
from unittest.mock import patch

import app_v3
import earnings as earnings_mod


def test_upsert_and_list_earnings_entry(client, clean_db):
    r = client.post('/api/earnings/calendar', json={
        'ticker': 'nvda',
        'quarterLabel': '1Q26',
        'confirmedDate': str(date.today() + timedelta(days=5)),
        'timing': 'AMC',
        'prUrl': 'https://investors.nvidia.com/Q1-26-pr.pdf',
    })
    assert r.status_code == 200
    entry = r.get_json()['entry']
    assert entry['ticker'] == 'NVDA'
    assert entry['quarterLabel'] == '1Q26'
    assert entry['prUrl'].endswith('pr.pdf')

    # Upsert: second call updates
    client.post('/api/earnings/calendar', json={
        'ticker': 'NVDA',
        'quarterLabel': '1Q26',
        'transcriptUrl': 'https://seekingalpha.com/nvda-q1-26.html',
    })
    listed = client.get('/api/earnings/calendar?ticker=NVDA').get_json()['items']
    assert len(listed) == 1
    assert listed[0]['transcriptUrl'].endswith('q1-26.html')

    # Upcoming window includes it
    up = client.get('/api/earnings/upcoming?days=14').get_json()['items']
    assert any(i['ticker'] == 'NVDA' for i in up)


def test_missing_required_fields_returns_400(client, clean_db):
    r = client.post('/api/earnings/calendar', json={'ticker': 'NVDA'})
    assert r.status_code == 400


def test_delete_entry(client, clean_db):
    entry = client.post('/api/earnings/calendar', json={
        'ticker': 'MDT', 'quarterLabel': '3Q26', 'confirmedDate': str(date.today()),
    }).get_json()['entry']
    r = client.delete(f'/api/earnings/calendar/{entry["id"]}')
    assert r.status_code == 200
    # gone
    listed = client.get('/api/earnings/calendar?ticker=MDT').get_json()['items']
    assert listed == []


def test_config_put_and_get(client, clean_db):
    r = client.put('/api/earnings/config/amzn', json={
        'irUrl': 'https://ir.aboutamazon.com/',
        'prUrlPattern': 'https://ir.aboutamazon.com/files/{year}Q{q}.pdf',
        'transcriptSource': 'manual',
    })
    assert r.status_code == 200
    got = client.get('/api/earnings/config/AMZN').get_json()
    assert got['irUrl'].endswith('aboutamazon.com/')
    assert got['transcriptSource'] == 'manual'


def test_fetch_now_creates_pipeline_job(client, clean_db):
    entry = client.post('/api/earnings/calendar', json={
        'ticker': 'BSX', 'quarterLabel': '1Q26',
        'confirmedDate': str(date.today()),
        'prUrl': 'https://investors.bostonscientific.com/pr.pdf',
    }).get_json()['entry']
    r = client.post(f'/api/earnings/calendar/{entry["id"]}/fetch-now')
    assert r.status_code == 200
    job_id = r.get_json()['jobId']

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT job_type, ticker, status, steps_detail FROM research_pipeline_jobs WHERE id=%s", (job_id,))
        row = cur.fetchone()
    assert row['job_type'] == 'earnings_fetch'
    assert row['ticker'] == 'BSX'
    sd = row['steps_detail']
    if isinstance(sd, str):
        sd = json.loads(sd)
    assert sd['pr_url'].endswith('pr.pdf')
    assert sd['calendar_id'] == entry['id']


def test_check_due_earnings_dispatches_today(client, clean_db):
    today = date.today()
    client.post('/api/earnings/calendar', json={
        'ticker': 'LLY', 'quarterLabel': '1Q26',
        'confirmedDate': str(today),
        'prUrl': 'https://investor.lilly.com/pr.pdf',
    })
    dispatched = earnings_mod.check_due_earnings()
    tickers = [d['ticker'] for d in dispatched]
    assert 'LLY' in tickers

    # Calendar entry status should flip to 'fetching'
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status FROM earnings_calendar WHERE ticker='LLY'")
        assert cur.fetchone()['status'] == 'fetching'


def test_check_due_earnings_skips_if_recent_job_exists(client, clean_db):
    today = date.today()
    entry = client.post('/api/earnings/calendar', json={
        'ticker': 'UNP', 'quarterLabel': '1Q26',
        'confirmedDate': str(today),
        'prUrl': 'https://up.com/pr.pdf',
    }).get_json()['entry']
    first = earnings_mod.check_due_earnings()
    assert len(first) == 1
    # Status back to upcoming so dispatcher considers it again; but dedup
    # on recent pipeline job should still suppress re-dispatch.
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE earnings_calendar SET status='upcoming' WHERE id=%s", (entry['id'],))
    second = earnings_mod.check_due_earnings()
    assert second == []


def test_fetch_result_updates_status(client, clean_db):
    entry = client.post('/api/earnings/calendar', json={
        'ticker': 'UNH', 'quarterLabel': '1Q26', 'confirmedDate': str(date.today()),
    }).get_json()['entry']
    r = client.post('/api/earnings/fetch-result', json={
        'calendarId': entry['id'],
        'status': 'fetched',
        'notes': 'PR + transcript saved',
    })
    assert r.status_code == 200
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status, fetch_notes FROM earnings_calendar WHERE id=%s", (entry['id'],))
        row = cur.fetchone()
    assert row['status'] == 'fetched'
    assert 'transcript' in row['fetch_notes']


def test_upcoming_endpoint_respects_window(client, clean_db):
    # Entry 60 days out should NOT be in a 14-day window
    client.post('/api/earnings/calendar', json={
        'ticker': 'XYZ', 'quarterLabel': '2Q26',
        'confirmedDate': str(date.today() + timedelta(days=60)),
    })
    window14 = client.get('/api/earnings/upcoming?days=14').get_json()['items']
    assert not any(i['ticker'] == 'XYZ' for i in window14)
    window90 = client.get('/api/earnings/upcoming?days=90').get_json()['items']
    assert any(i['ticker'] == 'XYZ' for i in window90)
