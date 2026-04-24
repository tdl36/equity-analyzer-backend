"""Notification senders for media-tracker alerts (M8).

Three channels: push (web push via VAPID), telegram, email (7am digest).
Each sender is a no-op if the relevant channel is off in
app_settings.media_notification_channels or the config is missing.
None raise — they print warnings and return.
"""
import json
import os

import requests

import app_v3


def _load_channels() -> dict:
    """Read app_settings.media_notification_channels; return {tab,push,telegram,email}."""
    default = {'tab': True, 'push': False, 'telegram': False, 'email': False}
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key='media_notification_channels'")
            row = cur.fetchone()
        if row and row['value']:
            val = row['value'] if isinstance(row['value'], dict) else json.loads(row['value'])
            return {**default, **(val or {})}
    except Exception:
        pass
    return default


def _telegram_send(message: str) -> None:
    """Uses TELEGRAM_BOT_TOKEN env + app_settings.telegram_chat_id (fallback TELEGRAM_CHAT_ID env)."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = ''
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key='telegram_chat_id'")
            row = cur.fetchone()
        if row and row['value']:
            chat_id = row['value'] if isinstance(row['value'], str) else str(row['value'])
    except Exception:
        pass
    chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID', '')
    if not token or not chat_id:
        print('telegram: no token/chat_id, skipping')
        return
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={
                'chat_id': chat_id,
                'text': message[:4000],
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True,
            },
            timeout=10,
        )
        if not r.ok:
            print(f'telegram send failed: {r.status_code} {r.text[:200]}')
    except Exception as e:
        print(f'telegram send error: {e}')


def _push_send(title: str, body: str, url: str = '/') -> dict:
    """Web push to any saved push subscriptions in notification_prefs.push_subscriptions.

    Uses pywebpush + VAPID if env vars set (VAPID_PRIVATE_KEY, VAPID_CLAIMS_EMAIL).
    Returns {total, sent, failed, reason, errors} so callers can surface diagnostics.
    """
    stats = {'total': 0, 'sent': 0, 'failed': 0, 'reason': None, 'errors': []}
    try:
        from pywebpush import webpush
    except ImportError:
        stats['reason'] = 'pywebpush not installed'
        print(stats['reason'])
        return stats
    vapid_priv = os.environ.get('VAPID_PRIVATE_KEY', '')
    vapid_claims_email = os.environ.get('VAPID_CLAIMS_EMAIL', '')
    if not vapid_priv:
        stats['reason'] = 'VAPID_PRIVATE_KEY not set'
        print(stats['reason'])
        return stats
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM notification_prefs WHERE key='push_subscriptions'")
            row = cur.fetchone()
        subs = []
        if row and row['value']:
            val = row['value'] if isinstance(row['value'], list) else json.loads(row['value'])
            subs = val or []
    except Exception as e:
        stats['reason'] = f'db read failed: {e}'
        print(stats['reason'])
        return stats
    stats['total'] = len(subs)
    if not subs:
        stats['reason'] = 'no push subscriptions saved'
        print(stats['reason'])
        return stats
    payload = json.dumps({'title': title, 'body': body, 'url': url})
    for sub in subs:
        try:
            webpush(
                sub,
                payload,
                vapid_private_key=vapid_priv,
                vapid_claims={'sub': f'mailto:{vapid_claims_email or "noreply@tonydlee.com"}'},
            )
            stats['sent'] += 1
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(str(e))
            print(f'push send error for one sub: {e}')
    return stats


def _email_send(subject: str, html: str, to: str = '') -> None:
    """SMTP send via env vars (SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_PORT, SMTP_TO).

    Falls back to printing when SMTP env is missing.
    """
    import smtplib
    import ssl
    from email.message import EmailMessage

    host = os.environ.get('SMTP_HOST', '')
    user = os.environ.get('SMTP_USER', '')
    password = os.environ.get('SMTP_PASSWORD', '')
    port = int(os.environ.get('SMTP_PORT', '465'))
    to_addr = to or os.environ.get('SMTP_TO', 'tonydlee@gmail.com')
    if not host or not user or not password:
        print('smtp config missing; skipping email')
        return
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = to_addr
    msg.set_content('Charlie digest (HTML version attached)')
    msg.add_alternative(html, subtype='html')
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx) as server:
            server.login(user, password)
            server.send_message(msg)
    except Exception as e:
        print(f'email send error: {e}')


# -------- Public API ---------------------------------------------------------


def notify_new_material_alert(
    ticker: str,
    point_text: str,
    episode_title: str,
    feed_name: str,
    source_url: str = '',
) -> None:
    """Called from material.py right after a new podcast_material alert is created.
    Fires push + telegram if those channels are enabled (email batches at 7am, not per-item).
    """
    ch = _load_channels()
    title_prefix = f'Charlie {ticker}' if ticker else 'Charlie'
    body = (point_text or '')[:200]
    if ch.get('push'):
        _push_send(title_prefix, body, url='/#alerts')
    if ch.get('telegram'):
        md_parts = [f'*{title_prefix} — {feed_name}*', point_text or '', '', f'_{episode_title}_']
        if source_url:
            md_parts.append(f'[Source]({source_url})')
        _telegram_send('\n'.join(md_parts))


def send_daily_email_digest() -> None:
    """Run daily at 7am. Aggregate last-24h material alerts + send HTML email if enabled."""
    ch = _load_channels()
    if not ch.get('email'):
        return
    with app_v3.get_db() as (_c, cur):
        cur.execute(
            """
            SELECT id, ticker, title, detail, created_at FROM agent_alerts
             WHERE alert_type='podcast_material'
               AND created_at > NOW() - INTERVAL '24 hours'
             ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
    if not rows:
        return
    items = []
    for r in rows:
        det = r['detail'] if isinstance(r['detail'], dict) else (json.loads(r['detail']) if r['detail'] else {})
        items.append({
            'ticker': r['ticker'],
            'title': r['title'],
            'feedName': det.get('feedName'),
            'episodeTitle': det.get('episodeTitle'),
            'sourceUrl': det.get('sourceUrl'),
        })
    by_tic = {}
    for it in items:
        by_tic.setdefault(it['ticker'] or '?', []).append(it)
    html_parts = [
        '<div style="font-family:sans-serif;max-width:700px;">',
        f'<h2>Charlie — Material Alerts, last 24h ({len(items)})</h2>',
    ]
    for tic, arr in sorted(by_tic.items()):
        html_parts.append(f'<h3 style="margin:16px 0 8px;color:#b45309;">{tic}</h3>')
        for it in arr:
            src = f' · <a href="{it["sourceUrl"]}">source</a>' if it['sourceUrl'] else ''
            html_parts.append(
                f'<p style="margin:4px 0 12px;"><strong>{it["title"]}</strong><br/>'
                f'<span style="color:#666;font-size:12px;">{it["feedName"]} — '
                f'{it["episodeTitle"]}{src}</span></p>'
            )
    html_parts.append('</div>')
    html = '\n'.join(html_parts)
    to = ''
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key='media_email_digest_to'")
            row = cur.fetchone()
        if row and row['value']:
            to = row['value'] if isinstance(row['value'], str) else str(row['value'])
    except Exception:
        pass
    _email_send(f'Charlie Digest — {len(items)} new alerts', html, to=to)
