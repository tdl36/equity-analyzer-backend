"""Tests for media-tracker notifications (M8)."""
import json
from unittest.mock import patch

import app_v3
from media_trackers import notifications


# -------- notify_new_material_alert channel gating ----------------------


def test_notify_skips_when_all_channels_off(clean_db):
    # Explicitly turn everything off
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'tab': False, 'push': False, 'telegram': False, 'email': False}),),
        )
    with patch.object(notifications, '_telegram_send') as mock_tg, \
         patch.object(notifications, '_push_send') as mock_push:
        notifications.notify_new_material_alert(
            ticker='LLY', point_text='x', episode_title='ep', feed_name='feed'
        )
    mock_tg.assert_not_called()
    mock_push.assert_not_called()


def test_notify_calls_telegram_when_enabled(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'tab': True, 'push': False, 'telegram': True, 'email': False}),),
        )
    with patch.object(notifications, '_telegram_send') as mock_tg, \
         patch.object(notifications, '_push_send') as mock_push:
        notifications.notify_new_material_alert(
            ticker='LLY',
            point_text='Zepbound supply up 40%',
            episode_title='Odd Lots: GLP-1',
            feed_name='Odd Lots',
            source_url='https://example.com/ep1',
        )
    mock_tg.assert_called_once()
    mock_push.assert_not_called()
    # Message should contain ticker, point text, episode title, source link
    msg = mock_tg.call_args[0][0]
    assert 'LLY' in msg
    assert 'Zepbound' in msg
    assert 'Odd Lots' in msg
    assert 'https://example.com/ep1' in msg


def test_notify_calls_push_when_enabled(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'tab': True, 'push': True, 'telegram': False, 'email': False}),),
        )
    with patch.object(notifications, '_telegram_send') as mock_tg, \
         patch.object(notifications, '_push_send') as mock_push:
        notifications.notify_new_material_alert(
            ticker='NVDA', point_text='data center rev +50%', episode_title='ep', feed_name='f'
        )
    mock_push.assert_called_once()
    mock_tg.assert_not_called()


# -------- send_daily_email_digest --------------------------------------


def test_daily_digest_noop_when_email_off(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'tab': True, 'push': False, 'telegram': False, 'email': False}),),
        )
    with patch.object(notifications, '_email_send') as mock_email:
        notifications.send_daily_email_digest()
    mock_email.assert_not_called()


def test_daily_digest_noop_when_no_material(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'tab': True, 'push': False, 'telegram': False, 'email': True}),),
        )
    with patch.object(notifications, '_email_send') as mock_email:
        notifications.send_daily_email_digest()
    mock_email.assert_not_called()


def test_daily_digest_groups_by_ticker(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'tab': True, 'push': False, 'telegram': False, 'email': True}),),
        )
        # Seed 4 material alerts across 2 tickers
        for i, (tic, title) in enumerate([
            ('LLY', 'Zepbound supply 40% up'),
            ('LLY', 'Mounjaro demand strong in EU'),
            ('NVDA', 'Data center revenue record'),
            ('NVDA', 'Hopper sellout continues'),
        ]):
            detail = {
                'episodeTitle': f'Episode {i}',
                'feedName': 'Odd Lots',
                'sourceUrl': f'https://example.com/ep{i}',
            }
            cur.execute(
                '''
                INSERT INTO agent_alerts (id, alert_type, ticker, title, detail, status, created_at)
                VALUES (%s, 'podcast_material', %s, %s, %s::jsonb, 'new', NOW() - INTERVAL '1 hour')
                ''',
                (f'a{i}', tic, title, json.dumps(detail)),
            )
    with patch.object(notifications, '_email_send') as mock_email:
        notifications.send_daily_email_digest()
    mock_email.assert_called_once()
    subject, html = mock_email.call_args[0][0], mock_email.call_args[0][1]
    assert '4 new alerts' in subject
    assert 'LLY' in html
    assert 'NVDA' in html
    assert 'Zepbound' in html
    assert 'Hopper' in html


def test_daily_digest_ignores_old_alerts(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'tab': True, 'push': False, 'telegram': False, 'email': True}),),
        )
        cur.execute(
            '''
            INSERT INTO agent_alerts (id, alert_type, ticker, title, detail, status, created_at)
            VALUES ('old1', 'podcast_material', 'LLY', 'Old alert', '{}'::jsonb, 'new', NOW() - INTERVAL '2 days')
            '''
        )
    with patch.object(notifications, '_email_send') as mock_email:
        notifications.send_daily_email_digest()
    mock_email.assert_not_called()


# -------- _load_channels ------------------------------------------------


def test_load_channels_default_when_unset(clean_db):
    ch = notifications._load_channels()
    assert ch == {'tab': True, 'push': False, 'telegram': False, 'email': False}


def test_load_channels_merges_partial_settings(clean_db):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            '''
            INSERT INTO app_settings (key, value) VALUES ('media_notification_channels', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            ''',
            (json.dumps({'telegram': True}),),
        )
    ch = notifications._load_channels()
    assert ch['telegram'] is True
    # defaults preserved
    assert ch['tab'] is True
    assert ch['push'] is False
