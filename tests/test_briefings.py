"""Tests for per-analyst briefing emails."""
import json
from unittest.mock import patch

import app_v3
import briefings


def _create_analyst_with_briefings(client, name, tickers, briefings_cfg):
    r = client.post('/api/analysts', json={'name': name, 'coverageTickers': tickers}).get_json()
    aid = r['analyst']['id']
    client.patch(f'/api/analysts/{aid}', json={'playbook': {'briefings': briefings_cfg}})
    return aid


def test_briefing_enabled_helper():
    assert briefings._briefing_enabled(None, 'bmo') is False
    assert briefings._briefing_enabled({}, 'bmo') is False
    assert briefings._briefing_enabled({'briefings': {}}, 'bmo') is False
    assert briefings._briefing_enabled({'briefings': {'bmo': True}}, 'bmo') is True
    assert briefings._briefing_enabled({'briefings': {'bmo': True}}, 'amc') is False


def test_send_briefings_skips_disabled_analysts(client, clean_db):
    _create_analyst_with_briefings(client, 'a1', ['MDT'], {'bmo': False})
    with patch.object(briefings, '_email_send') as mock_send:
        stats = briefings.send_briefings_for_context('bmo')
    assert stats['sent'] == 0
    assert stats['skipped'] >= 1
    mock_send.assert_not_called()


def test_send_briefings_emails_enabled_analyst_with_content(client, clean_db):
    aid = _create_analyst_with_briefings(client, 'a1', ['MDT'], {'bmo': True})
    # Seed an earnings entry for today so the briefing has content
    from datetime import date
    client.post('/api/earnings/calendar', json={
        'ticker': 'MDT', 'quarterLabel': '1Q26',
        'confirmedDate': str(date.today()),
        'timing': 'BMO',
    })
    with patch.object(briefings, '_email_send') as mock_send:
        stats = briefings.send_briefings_for_context('bmo')
    assert stats['sent'] == 1
    mock_send.assert_called_once()
    call = mock_send.call_args
    subject = call.args[0] if call.args else call.kwargs.get('subject')
    html = call.args[1] if len(call.args) > 1 else call.kwargs.get('html')
    assert 'BMO briefing' in subject
    assert 'MDT' in html
    assert 'Earnings Today' in html


def test_send_briefings_skips_enabled_but_empty_on_scheduled_contexts(client, clean_db):
    _create_analyst_with_briefings(client, 'a1', ['MDT'], {'amc': True})
    # No earnings, no bullets, no pending inbox -> should skip
    with patch.object(briefings, '_email_send') as mock_send:
        stats = briefings.send_briefings_for_context('amc')
    assert stats['sent'] == 0
    assert stats['skipped'] >= 1
    mock_send.assert_not_called()


def test_event_recap_briefing_respects_toggle(client, clean_db):
    aid = _create_analyst_with_briefings(client, 'a1', ['MDT'], {'on_recap_ready': True})
    # Seed activity with synthesisMarkdown
    import uuid
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO analyst_activities
                (id, analyst_id, activity_type, ticker, status, trigger_source, input, output)
            VALUES (%s, %s, 'earnings_recap', 'MDT', 'pending_review', 'catalyst_folder',
                    %s::jsonb, %s::jsonb)
        ''', (str(uuid.uuid4()), aid,
              json.dumps({'topic': '1Q26 Earnings'}),
              json.dumps({'synthesisMarkdown': '# Recap\n\nMDT beat.'})))
        cur.execute("SELECT id FROM analyst_activities WHERE ticker='MDT' ORDER BY created_at DESC LIMIT 1")
        act_id = cur.fetchone()['id']

    with patch.object(briefings, '_email_send') as mock_send:
        out = briefings.send_event_recap_briefing(act_id)
    assert out['sent'] is True
    mock_send.assert_called_once()
    subject = mock_send.call_args.args[0] if mock_send.call_args.args else mock_send.call_args.kwargs.get('subject')
    assert 'recap ready' in subject.lower()


def test_event_recap_briefing_skips_when_toggle_off(client, clean_db):
    aid = _create_analyst_with_briefings(client, 'a1', ['MDT'], {'on_recap_ready': False})
    import uuid
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO analyst_activities (id, analyst_id, activity_type, ticker, status, trigger_source, input, output)
            VALUES (%s, %s, 'earnings_recap', 'MDT', 'pending_review', 'catalyst_folder', %s::jsonb, %s::jsonb)
        ''', (str(uuid.uuid4()), aid, json.dumps({'topic': '1Q26 Earnings'}), json.dumps({'synthesisMarkdown': '# R'})))
        cur.execute("SELECT id FROM analyst_activities WHERE ticker='MDT' ORDER BY created_at DESC LIMIT 1")
        act_id = cur.fetchone()['id']
    with patch.object(briefings, '_email_send') as mock_send:
        out = briefings.send_event_recap_briefing(act_id)
    assert out['sent'] is False
    mock_send.assert_not_called()


def test_briefings_send_endpoint_validates_context(client, clean_db):
    r = client.post('/api/briefings/send', json={'context': 'invalid'})
    assert r.status_code == 400


def test_briefings_send_endpoint_routes(client, clean_db):
    _create_analyst_with_briefings(client, 'a1', ['MDT'], {'bmo': True})
    from datetime import date
    client.post('/api/earnings/calendar', json={'ticker': 'MDT', 'quarterLabel': '1Q26', 'confirmedDate': str(date.today())})
    with patch.object(briefings, '_email_send') as mock_send:
        r = client.post('/api/briefings/send', json={'context': 'bmo'})
    assert r.status_code == 200
    body = r.get_json()
    assert body['sent'] == 1
    mock_send.assert_called_once()
