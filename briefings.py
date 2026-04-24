"""Per-analyst briefing emails.

Three scheduled briefings per day (BMO / midday / AMC) + one event trigger
when an auto-mode earnings recap completes. Each briefing bundles:

  - Earnings hitting today (from earnings_calendar)
  - Recently-completed earnings recaps by this analyst (since prior briefing)
  - Pending inbox items (earnings_recap / takeaway activities)
  - Fresh material podcast bullets mentioning covered tickers (since prior)

Per-analyst opt-in via analyst.playbook.briefings = {
    bmo: bool,           # default false
    midday: bool,        # default false
    amc: bool,           # default false
    on_recap_ready: bool # default false — event trigger
}.

Recipient: analyst.playbook.briefings.email  OR  app_settings.briefing_email_to  OR  env SMTP_TO.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import app_v3
from media_trackers.notifications import _email_send


def _get_setting(key: str) -> str | None:
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key=%s", (key,))
            row = cur.fetchone()
        if row and row.get('value'):
            return str(row['value'])
    except Exception:
        pass
    return None


def _recipient_for(playbook: dict | None) -> str:
    if isinstance(playbook, dict):
        email = (playbook.get('briefings') or {}).get('email')
        if email:
            return str(email).strip()
    return _get_setting('briefing_email_to') or _get_setting('media_email_digest_to') or 'tonydlee@gmail.com'


def _briefing_enabled(playbook: dict | None, context: str) -> bool:
    """context ∈ {'bmo','midday','amc','on_recap_ready'}"""
    if not isinstance(playbook, dict):
        return False
    b = playbook.get('briefings') or {}
    return bool(b.get(context))


def _today() -> date:
    return datetime.utcnow().date()


def _fetch_earnings_today(tickers: list[str]) -> list[dict]:
    if not tickers:
        return []
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT ticker, quarter_label, timing, confirmed_date, expected_date
              FROM earnings_calendar
             WHERE ticker = ANY(%s)
               AND COALESCE(confirmed_date, expected_date) = %s
             ORDER BY timing NULLS LAST, ticker
        ''', (tickers, _today()))
        rows = cur.fetchall() or []
    return [dict(r) for r in rows]


def _fetch_recent_recaps(analyst_id: str, since: datetime) -> list[dict]:
    """Earnings recaps completed since the given timestamp (status flips back
    to pending_review with output.synthesisMarkdown set when the LLM finishes
    in Auto Mode, or straight to 'approved' if user has already approved)."""
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id, ticker, activity_type, status, input, output, updated_at
              FROM analyst_activities
             WHERE analyst_id = %s
               AND activity_type = 'earnings_recap'
               AND updated_at >= %s
               AND output ? 'synthesisMarkdown'
             ORDER BY updated_at DESC
             LIMIT 20
        ''', (analyst_id, since))
        rows = cur.fetchall() or []
    return [dict(r) for r in rows]


def _fetch_pending_inbox(analyst_id: str) -> list[dict]:
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id, ticker, activity_type, input, status, created_at
              FROM analyst_activities
             WHERE analyst_id = %s
               AND status IN ('pending_review', 'running')
             ORDER BY created_at DESC
             LIMIT 10
        ''', (analyst_id,))
        rows = cur.fetchall() or []
    return [dict(r) for r in rows]


def _fetch_new_bullets(tickers: list[str], since: datetime, limit: int = 15) -> list[dict]:
    if not tickers:
        return []
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT p.text, p.tickers, p.material, e.title AS episode_title,
                   f.name AS feed_name, e.source_url, e.published_at
              FROM media_digest_points p
              JOIN media_episodes e ON e.id = p.episode_id
              JOIN media_feeds f ON f.id = e.feed_id
             WHERE p.tickers && %s
               AND e.published_at >= %s
             ORDER BY e.published_at DESC
             LIMIT %s
        ''', (tickers, since, limit))
        rows = cur.fetchall() or []
    return [dict(r) for r in rows]


def _prior_briefing_time(analyst_id: str, context: str) -> datetime:
    """Last time we sent a briefing of this context to this analyst.
    Fallback: 18h ago (catches everything since yesterday's cycle)."""
    val = _get_setting(f'briefing_last_{analyst_id}_{context}')
    if val:
        try:
            return datetime.fromisoformat(val)
        except Exception:
            pass
    return datetime.utcnow() - timedelta(hours=18)


def _record_briefing_sent(analyst_id: str, context: str) -> None:
    now_iso = datetime.utcnow().isoformat()
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        ''', (f'briefing_last_{analyst_id}_{context}', now_iso))


def _html_for(analyst: dict, context: str, sections: dict) -> str:
    tickers = analyst.get('coverage_tickers') or []
    earnings = sections.get('earnings') or []
    recaps = sections.get('recaps') or []
    inbox = sections.get('inbox') or []
    bullets = sections.get('bullets') or []

    ctx_label = {
        'bmo': 'BMO (pre-market open)',
        'midday': 'Midday',
        'amc': 'AMC (after-market close)',
        'on_recap_ready': 'Recap Ready',
    }.get(context, context)

    parts: list[str] = []
    parts.append(f'<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:720px;color:#1f2937;">')
    parts.append(f'<h2 style="margin:0 0 4px;color:#111827;">{analyst.get("name") or "Analyst"} · {ctx_label}</h2>')
    parts.append(f'<div style="color:#6b7280;font-size:12px;margin-bottom:20px;">Coverage: {", ".join(tickers) or "(none)"}</div>')

    if earnings:
        parts.append('<h3 style="margin:16px 0 6px;color:#b45309;">📊 Earnings Today</h3><ul style="margin:0 0 16px;padding-left:20px;">')
        for e in earnings:
            timing = f' · <strong>{e["timing"]}</strong>' if e.get('timing') else ''
            parts.append(f'<li><strong style="font-family:monospace;">{e["ticker"]}</strong> — {e.get("quarter_label") or ""}{timing}</li>')
        parts.append('</ul>')

    if recaps:
        parts.append('<h3 style="margin:16px 0 6px;color:#059669;">✅ Recently-Completed Recaps</h3>')
        for r in recaps:
            out = r.get('output') or {}
            if isinstance(out, str):
                import json as _j
                try: out = _j.loads(out)
                except Exception: out = {}
            inp = r.get('input') or {}
            if isinstance(inp, str):
                import json as _j
                try: inp = _j.loads(inp)
                except Exception: inp = {}
            md = (out.get('synthesisMarkdown') or '')[:4000]
            parts.append(
                f'<div style="margin:12px 0;padding:12px;border:1px solid #e5e7eb;border-radius:8px;">'
                f'<div style="font-size:12px;color:#6b7280;margin-bottom:6px;"><strong>{r.get("ticker")}</strong> · {inp.get("topic") or ""}</div>'
                f'<pre style="white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.6;margin:0;">{md}</pre>'
                f'</div>'
            )

    if inbox:
        parts.append('<h3 style="margin:16px 0 6px;color:#d97706;">📥 Pending Inbox</h3><ul style="margin:0 0 16px;padding-left:20px;">')
        for it in inbox:
            inp = it.get('input') or {}
            if isinstance(inp, str):
                import json as _j
                try: inp = _j.loads(inp)
                except Exception: inp = {}
            parts.append(f'<li>{it.get("ticker")} · {it.get("activity_type")} · {inp.get("topic") or inp.get("pointText","")[:80]} <em style="color:#9ca3af;">[{it.get("status")}]</em></li>')
        parts.append('</ul>')

    if bullets:
        parts.append('<h3 style="margin:16px 0 6px;color:#7c3aed;">🎧 New Podcast Bullets on Coverage</h3><ul style="margin:0 0 16px;padding-left:20px;">')
        for b in bullets:
            tix = ', '.join(b.get('tickers') or [])
            mat = ' · <span style="color:#b91c1c;font-weight:600;">MATERIAL</span>' if b.get('material') else ''
            parts.append(
                f'<li style="margin-bottom:8px;"><strong>{tix}</strong>{mat} — {b.get("text") or ""}'
                f'<div style="font-size:11px;color:#9ca3af;">{b.get("feed_name") or ""} · {b.get("episode_title") or ""}</div></li>'
            )
        parts.append('</ul>')

    if not (earnings or recaps or inbox or bullets):
        parts.append('<p style="color:#6b7280;font-style:italic;">No updates since the last briefing.</p>')

    parts.append('<p style="color:#9ca3af;font-size:11px;margin-top:24px;">Sent by Charlie · briefing context: ' + context + '</p></div>')
    return ''.join(parts)


def send_briefings_for_context(context: str) -> dict:
    """Iterate all analysts with this briefing context enabled and send email.
    Returns {sent: N, skipped: N, errors: N}."""
    stats = {'sent': 0, 'skipped': 0, 'errors': 0}
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT * FROM analysts')
        analysts = [dict(r) for r in cur.fetchall() or []]
    for a in analysts:
        try:
            pb = a.get('playbook') or {}
            if isinstance(pb, str):
                import json as _j
                try: pb = _j.loads(pb)
                except Exception: pb = {}
            if not _briefing_enabled(pb, context):
                stats['skipped'] += 1
                continue
            tickers = list(a.get('coverage_tickers') or [])
            since = _prior_briefing_time(a['id'], context)
            sections = {
                'earnings': _fetch_earnings_today(tickers),
                'recaps': _fetch_recent_recaps(a['id'], since),
                'inbox': _fetch_pending_inbox(a['id']),
                'bullets': _fetch_new_bullets(tickers, since),
            }
            # Skip sending if truly nothing worth reporting on BMO/midday/AMC
            # cycles (event-triggered always sends).
            if context != 'on_recap_ready' and not any(sections.values()):
                stats['skipped'] += 1
                continue
            html = _html_for(a, context, sections)
            ctx_label = {'bmo': 'BMO', 'midday': 'Midday', 'amc': 'AMC', 'on_recap_ready': 'Recap Ready'}[context]
            subject = f'[Charlie] {a.get("name") or "Analyst"} — {ctx_label} briefing'
            _email_send(subject, html, to=_recipient_for(pb))
            _record_briefing_sent(a['id'], context)
            stats['sent'] += 1
        except Exception as e:
            print(f'briefing send error for {a.get("id")}: {e}')
            stats['errors'] += 1
    return stats


def send_event_recap_briefing(activity_id: str) -> dict:
    """Called right after an earnings recap completes. If the owning
    analyst has briefings.on_recap_ready enabled, send an immediate email
    with that recap."""
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute('''
                SELECT act.*, a.* FROM analyst_activities act
                  LEFT JOIN analysts a ON a.id = act.analyst_id
                 WHERE act.id = %s
            ''', (activity_id,))
            row = cur.fetchone()
        if not row:
            return {'sent': False, 'reason': 'activity not found'}
        pb = row.get('playbook') or {}
        if isinstance(pb, str):
            import json as _j
            try: pb = _j.loads(pb)
            except Exception: pb = {}
        if not _briefing_enabled(pb, 'on_recap_ready'):
            return {'sent': False, 'reason': 'on_recap_ready disabled'}

        analyst = {
            'id': row.get('analyst_id'),
            'name': row.get('name'),
            'coverage_tickers': row.get('coverage_tickers') or [],
        }
        sections = {
            'earnings': [],
            'recaps': [dict(row)],
            'inbox': [],
            'bullets': [],
        }
        html = _html_for(analyst, 'on_recap_ready', sections)
        inp = row.get('input') or {}
        if isinstance(inp, str):
            import json as _j
            try: inp = _j.loads(inp)
            except Exception: inp = {}
        topic = inp.get('topic') or row.get('activity_type') or 'Earnings Recap'
        subject = f'[Charlie] {row.get("ticker")} {topic} — recap ready'
        _email_send(subject, html, to=_recipient_for(pb))
        # Also fire an instant alert on any enabled channel
        try:
            from media_trackers.notifications import _push_send, _telegram_send, _load_channels
            title = f"Recap ready · {row.get('ticker')}"
            body = f"{topic} — tap to review"
            ch = _load_channels()
            if ch.get('push'):
                _push_send(title=title, body=body, url='/#analysts')
            if ch.get('telegram'):
                _telegram_send(f"*{title}*\n{body}")
        except Exception as _e:
            print(f'recap alert send failed: {_e}')
        return {'sent': True}
    except Exception as e:
        print(f'send_event_recap_briefing error: {e}')
        return {'sent': False, 'error': str(e)}
