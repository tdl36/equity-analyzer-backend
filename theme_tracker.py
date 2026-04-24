"""Cross-stock theme tracker (Phase 2+).

Scheduled hourly scan. For each analyst, look at podcast bullets in the last
48h that mention at least one of the analyst's covered tickers AND carry a
theme_tag. If the same theme_tag appears across >= MIN_TICKERS distinct
covered tickers AND has >= MIN_BULLETS bullets, route a 'theme_alert'
activity to that analyst's inbox (with dedup).

Different from media_theme_clusters (global weekly) in that this is
per-analyst, rolling 48h, and auto-routed.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import app_v3

MIN_BULLETS = 3
MIN_TICKERS = 2
WINDOW_HOURS = 48
DEDUP_HOURS = 24  # don't re-fire the same (analyst, theme) within this window


def _normalize_theme(tag: str) -> str:
    return (tag or '').strip().lower()


def scan_for_analyst(analyst_id: str, coverage: list[str]) -> list[dict]:
    """Return list of {theme, tickers, bulletIds, bulletCount, sample} that
    qualify as theme alerts for this analyst."""
    if not coverage:
        return []
    since = datetime.utcnow() - timedelta(hours=WINDOW_HOURS)
    # Fetch all bullets in window whose tickers intersect the analyst's coverage
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT p.id, p.text, p.tickers, p.theme_tags, p.material,
                   e.title AS episode_title, e.source_url, e.published_at,
                   f.name AS feed_name
              FROM media_digest_points p
              JOIN media_episodes e ON e.id = p.episode_id
              JOIN media_feeds f ON f.id = e.feed_id
             WHERE p.tickers && %s
               AND e.published_at >= %s
               AND COALESCE(array_length(p.theme_tags, 1), 0) > 0
        ''', (list(coverage), since))
        rows = cur.fetchall() or []
    # Group by normalized theme_tag
    by_theme: dict[str, dict] = {}
    cov_set = {c.upper() for c in coverage}
    for r in rows:
        for tag in (r.get('theme_tags') or []):
            nt = _normalize_theme(tag)
            if not nt:
                continue
            bucket = by_theme.setdefault(nt, {
                'theme': tag, 'bulletIds': set(), 'tickers': set(),
                'sample': [], 'material': 0,
            })
            bucket['bulletIds'].add(r['id'])
            for t in (r.get('tickers') or []):
                if t and t.upper() in cov_set:
                    bucket['tickers'].add(t.upper())
            if r.get('material'):
                bucket['material'] += 1
            if len(bucket['sample']) < 5:
                bucket['sample'].append({
                    'text': (r.get('text') or '')[:240],
                    'feed': r.get('feed_name') or '',
                    'episode': r.get('episode_title') or '',
                    'sourceUrl': r.get('source_url') or '',
                })
    qualified: list[dict] = []
    for nt, b in by_theme.items():
        if len(b['bulletIds']) < MIN_BULLETS:
            continue
        if len(b['tickers']) < MIN_TICKERS:
            continue
        qualified.append({
            'theme': b['theme'],
            'normalizedTheme': nt,
            'tickers': sorted(b['tickers']),
            'bulletIds': list(b['bulletIds']),
            'bulletCount': len(b['bulletIds']),
            'materialCount': b['material'],
            'sample': b['sample'],
        })
    # Rank by bullet count desc then ticker count desc
    qualified.sort(key=lambda q: (-q['bulletCount'], -len(q['tickers'])))
    return qualified


def _dedup_recent(analyst_id: str, normalized_theme: str) -> bool:
    """Return True if we already fired this theme for this analyst within
    DEDUP_HOURS — caller should skip."""
    cutoff = datetime.utcnow() - timedelta(hours=DEDUP_HOURS)
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM analyst_activities
             WHERE analyst_id = %s
               AND activity_type = 'theme_alert'
               AND created_at >= %s
               AND (input->>'normalizedTheme') = %s
             LIMIT 1
        ''', (analyst_id, cutoff, normalized_theme))
        return cur.fetchone() is not None


def run_theme_scan() -> dict:
    """Main entry point for the scheduler. Iterates all analysts, scans
    themes, creates theme_alert activities. Returns stats."""
    stats = {'analysts': 0, 'alerts_created': 0, 'deduped': 0}
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT id, name, coverage_tickers FROM analysts')
        analysts = [dict(r) for r in cur.fetchall() or []]
    for a in analysts:
        stats['analysts'] += 1
        qualified = scan_for_analyst(a['id'], list(a.get('coverage_tickers') or []))
        for q in qualified:
            nt = q['normalizedTheme']
            if _dedup_recent(a['id'], nt):
                stats['deduped'] += 1
                continue
            try:
                with app_v3.get_db(commit=True) as (_c, cur):
                    cur.execute('''
                        INSERT INTO analyst_activities
                            (id, analyst_id, activity_type, ticker, status, trigger_source, input, created_at)
                        VALUES (%s, %s, 'theme_alert', %s, 'pending_review', 'podcast_theme', %s::jsonb, NOW())
                    ''', (str(uuid.uuid4()), a['id'],
                          q['tickers'][0] if q['tickers'] else '',
                          json.dumps({
                              'theme': q['theme'],
                              'normalizedTheme': nt,
                              'tickers': q['tickers'],
                              'bulletIds': q['bulletIds'],
                              'bulletCount': q['bulletCount'],
                              'materialCount': q['materialCount'],
                              'sample': q['sample'],
                              'windowHours': WINDOW_HOURS,
                          })))
                stats['alerts_created'] += 1
                # Fire an instant alert on any enabled channel
                try:
                    from media_trackers.notifications import _push_send, _telegram_send, _load_channels
                    title = f"Theme alert · {a.get('name') or 'Analyst'}"
                    body = f"{q['theme']} · {q['bulletCount']} bullets across {', '.join(q['tickers'])}"
                    ch = _load_channels()
                    if ch.get('push'):
                        _push_send(title=title, body=body, url='/#analysts')
                    if ch.get('telegram'):
                        _telegram_send(f"*{title}*\n{body}")
                except Exception as _e:
                    print(f'theme alert send failed: {_e}')
            except Exception as e:
                print(f'theme_tracker: insert failed for analyst {a["id"]}: {e}')
    return stats
