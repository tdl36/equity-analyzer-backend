"""Thesis-biased recap + Finnhub sync tests."""
import json
from unittest.mock import patch

import app_v3
import finnhub_sync


# ---- _render_thesis_block ----

def test_render_thesis_block_empty_when_no_thesis():
    assert app_v3._render_thesis_block(None, 'MDT') == ''
    assert app_v3._render_thesis_block({}, 'MDT') == ''
    assert app_v3._render_thesis_block({'thesesByTicker': {}}, 'MDT') == ''


def test_render_thesis_block_full():
    pb = {
        'thesesByTicker': {
            'MDT': {
                'summary': 'Renal denervation + diabetes pump optionality drive mid-single-digit rev growth.',
                'bulls': ['Symplicity approval tailwind', 'MiniMed 780G ramp'],
                'bears': ['DRG headwinds', 'CRM share loss'],
                'signposts': ['RDN reimbursement Q1', 'Med-Surg margin back to 30%'],
                'compSet': ['BSX', 'ABT', 'BDX'],
                'keyRatios': ['Organic rev growth', 'Op margin'],
            }
        }
    }
    out = app_v3._render_thesis_block(pb, 'MDT')
    assert 'COVERAGE THESIS' in out
    assert 'Renal denervation' in out
    assert 'Symplicity approval' in out
    assert 'DRG headwinds' in out
    assert 'RDN reimbursement' in out
    assert 'BSX, ABT, BDX' in out
    assert 'Organic rev growth, Op margin' in out


def test_render_thesis_block_handles_ticker_case_insensitive():
    pb = {'thesesByTicker': {'MDT': {'summary': 's'}}}
    assert 'MY COVERAGE THESIS' in app_v3._render_thesis_block(pb, 'mdt')


# ---- dispatch_activity_run wires thesis into steps_detail ----

def _create_analyst_with_thesis(client, ticker, thesis):
    aid = client.post('/api/analysts', json={
        'name': 'Test', 'coverageTickers': [ticker],
    }).get_json()['analyst']['id']
    client.patch(f'/api/analysts/{aid}', json={'playbook': {'thesesByTicker': {ticker: thesis}}})
    return aid


def test_dispatch_earnings_recap_includes_thesis_block(client, clean_db):
    aid = _create_analyst_with_thesis(client, 'MDT', {
        'summary': 'Quality medtech compounder',
        'bulls': ['RDN launch'],
        'bears': ['DRG'],
    })
    client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'MDT', 'topic': '1Q26 Earnings', 'fingerprint': 'fp', 'fileCount': 1,
    })
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT id FROM analyst_activities WHERE ticker='MDT' ORDER BY created_at DESC LIMIT 1")
        act_id = cur.fetchone()['id']
    r = client.post(f'/api/analyst-activities/{act_id}/run', json={}).get_json()
    job_id = r['catalystJobId']
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT steps_detail FROM research_pipeline_jobs WHERE id=%s", (job_id,))
        sd = cur.fetchone()['steps_detail']
    if isinstance(sd, str):
        sd = json.loads(sd)
    assert 'thesis_block' in sd
    tb = sd['thesis_block']
    assert 'COVERAGE THESIS' in tb
    assert 'Quality medtech compounder' in tb


def test_dispatch_takeaway_does_not_include_thesis(client, clean_db):
    aid = _create_analyst_with_thesis(client, 'AAPL', {'summary': 'x'})
    client.post('/api/analysts/queue-catalyst-activity', json={
        'ticker': 'AAPL', 'topic': 'WWDC takeaways', 'fingerprint': 'fp', 'fileCount': 1,
    })
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT id FROM analyst_activities WHERE ticker='AAPL'")
        act_id = cur.fetchone()['id']
    r = client.post(f'/api/analyst-activities/{act_id}/run', json={}).get_json()
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT steps_detail FROM research_pipeline_jobs WHERE id=%s", (r['catalystJobId'],))
        sd = cur.fetchone()['steps_detail']
    if isinstance(sd, str):
        sd = json.loads(sd)
    # Takeaway uses generic prompt — thesis_block is unused but field is present/empty
    assert sd.get('thesis_block', '') == ''


# ---- Finnhub sync ----

def _symbol_responses():
    return {
        'MDT': [{'symbol': 'MDT', 'date': '2026-05-21', 'hour': 'bmo', 'epsEstimate': 1.47}],
        'BSX': [{'symbol': 'BSX', 'date': '2026-04-30', 'hour': 'amc', 'epsEstimate': 0.75}],
    }


def test_finnhub_sync_upserts_only_covered(client, clean_db):
    client.post('/api/analysts', json={'name': 't', 'coverageTickers': ['MDT', 'BSX']})
    resp_by_symbol = _symbol_responses()

    def _fake_fetch(symbol, days_ahead=120, days_back=7, api_key=None):
        return resp_by_symbol.get(symbol.upper(), [])

    with patch.object(finnhub_sync, '_api_key', return_value='k'):
        with patch.object(finnhub_sync, 'fetch_for_symbol', side_effect=_fake_fetch):
            # sleep gets called between calls; patch it to avoid 1.1s * N wait
            with patch('time.sleep', return_value=None):
                stats = finnhub_sync.sync(days_ahead=60, days_back=7)
    assert stats['upserted'] == 2
    assert sorted(stats['covered_matched']) == ['BSX', 'MDT']
    assert stats['covered_unmatched'] == []
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT ticker, quarter_label, timing FROM earnings_calendar ORDER BY ticker')
        rows = cur.fetchall()
    tickers = [r['ticker'] for r in rows]
    assert tickers == ['BSX', 'MDT']
    bsx = next(r for r in rows if r['ticker'] == 'BSX')
    assert bsx['timing'] == 'AMC'
    mdt = next(r for r in rows if r['ticker'] == 'MDT')
    assert mdt['timing'] == 'BMO'


def test_finnhub_sync_reports_unmatched_tickers(client, clean_db):
    client.post('/api/analysts', json={'name': 't', 'coverageTickers': ['MDT', 'XYZ']})

    def _fake_fetch(symbol, days_ahead=120, days_back=7, api_key=None):
        if symbol == 'MDT':
            return [{'symbol': 'MDT', 'date': '2026-05-21', 'hour': 'bmo'}]
        return []  # XYZ returns nothing

    with patch.object(finnhub_sync, '_api_key', return_value='k'):
        with patch.object(finnhub_sync, 'fetch_for_symbol', side_effect=_fake_fetch):
            with patch('time.sleep', return_value=None):
                stats = finnhub_sync.sync()
    assert stats['covered_matched'] == ['MDT']
    assert stats['covered_unmatched'] == ['XYZ']


def test_finnhub_sync_without_api_key_returns_error(client, clean_db):
    client.post('/api/analysts', json={'name': 't', 'coverageTickers': ['MDT']})
    with patch.object(finnhub_sync, '_api_key', return_value=None):
        import pytest
        with pytest.raises(RuntimeError):
            finnhub_sync.sync(days_ahead=60)


def test_finnhub_quarter_label_helper():
    from datetime import date
    assert finnhub_sync._quarter_label(date(2026, 2, 10)) == '1Q26'
    assert finnhub_sync._quarter_label(date(2026, 7, 25)) == '3Q26'
    assert finnhub_sync._quarter_label(date(2025, 12, 31)) == '4Q25'


# ---- Endpoints ----

def test_sync_finnhub_endpoint_returns_error_without_key(client, clean_db):
    client.post('/api/analysts', json={'name': 't', 'coverageTickers': ['MDT']})
    r = client.post('/api/earnings/sync-finnhub', json={})
    assert r.status_code == 400
    body = r.get_json()
    assert 'Finnhub API key' in (body.get('error') or '')


def test_sync_finnhub_endpoint_starts_async_with_key(client, clean_db):
    client.post('/api/analysts', json={'name': 't', 'coverageTickers': ['MDT']})
    client.post('/api/settings', json={'finnhub_api_key': 'testkey'})
    # Patch the background thread's sync call so it doesn't actually fetch
    with patch.object(finnhub_sync, 'sync', return_value={'ok': True}):
        r = client.post('/api/earnings/sync-finnhub', json={})
    assert r.status_code == 200
    body = r.get_json()
    assert body['started'] is True
    assert body['daysAhead'] == 70


def test_finnhub_status_endpoint(client, clean_db):
    r = client.get('/api/earnings/finnhub-status')
    assert r.status_code == 200
    body = r.get_json()
    assert body['hasKey'] is False
    assert body['lastSync'] is None
    # Save key
    client.post('/api/settings', json={'finnhub_api_key': 'testkey'})
    body = client.get('/api/earnings/finnhub-status').get_json()
    assert body['hasKey'] is True
