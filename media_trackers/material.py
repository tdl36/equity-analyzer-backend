"""Material-point gating: which bullet points become alerts.

After the extractor inserts `media_digest_points` rows for an episode, we:
  1. Pre-filter candidate points (ticker in coverage OR keyword/theme hit)
  2. Ask Sonnet ("material or boilerplate?")
  3. Mark material=TRUE on surviving points
  4. Dedup across the last 7 days of agent_alerts with same ticker +
     shared theme OR >=2 shared title tokens; if dup -> append to
     detail.relatedEpisodes. Otherwise -> INSERT a new agent_alerts row
     with alert_type='podcast_material'.

All DB / LLM calls are best-effort: failure is logged and treated as
non-material so the extract pipeline continues to mark the episode done.
"""
import json
import re
import uuid

import app_v3
from media_trackers.prompts import MATERIAL_JUDGE_PROMPT


DEDUP_WINDOW_DAYS = 7


def _load_coverage_universe():
    """Return set of covered tickers: saved_analyses (if present) + ticker_settings
    + non-muted signals_watchlist tickers."""
    covered = set()
    with app_v3.get_db() as (_c, cur):
        # saved_analyses: listed in spec, may not exist in schema yet
        try:
            cur.execute("SELECT DISTINCT ticker FROM saved_analyses WHERE ticker IS NOT NULL")
            for r in cur.fetchall():
                if r['ticker']:
                    covered.add(r['ticker'].upper())
        except Exception:
            pass
        # ticker_settings: the live "coverage" table
        try:
            cur.execute("SELECT ticker FROM ticker_settings WHERE ticker IS NOT NULL")
            for r in cur.fetchall():
                if r['ticker']:
                    covered.add(r['ticker'].upper())
        except Exception:
            pass
        # signals_watchlist tickers
        try:
            cur.execute("SELECT value FROM signals_watchlist WHERE kind='ticker' AND muted=FALSE")
            for r in cur.fetchall():
                if r['value']:
                    covered.add(r['value'].upper())
        except Exception:
            pass
    return covered


def _load_watchlist_keywords():
    """Return set of lowercased watchlist keyword values (kind='keyword' or
    'exec', not muted)."""
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute(
                "SELECT value FROM signals_watchlist WHERE kind IN ('keyword','exec') AND muted=FALSE"
            )
            rows = cur.fetchall()
        return {r['value'].lower() for r in rows if r['value']}
    except Exception:
        return set()


def _load_muted_coverage():
    """Read app_settings.media_muted_coverage_tickers (list of ticker strings)."""
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute(
                "SELECT value FROM app_settings WHERE key='media_muted_coverage_tickers'"
            )
            row = cur.fetchone()
        if row and row['value']:
            val = row['value'] if isinstance(row['value'], list) else json.loads(row['value'])
            return {t.upper() for t in val if t}
    except Exception:
        pass
    return set()


def _primary_ticker(point):
    tickers = point.get('tickers') or []
    return tickers[0].upper() if tickers else None


def _candidate_material(point, covered, keywords, muted_coverage):
    """Fast pre-filter: does this point touch our coverage/watchlist at all?

    Candidate if ANY:
      - one of point.tickers is in `covered` AND not in muted_coverage
      - any watchlist keyword substring-hit in point.text
      - any of point.theme_tags matches a watchlist keyword (case-insensitive)
    """
    for t in (point.get('tickers') or []):
        t_up = t.upper()
        if t_up in muted_coverage:
            continue
        if t_up in covered:
            return True
    text_l = (point.get('text') or '').lower()
    for kw in keywords:
        if kw and kw in text_l:
            return True
    for theme in (point.get('theme_tags') or []):
        if theme and theme.lower() in keywords:
            return True
    return False


def _judge_material(point_text, anthropic_api_key=None):
    """Sonnet judge call. Returns True/False.

    Transparently tolerant of errors — unknown -> False (treated as
    non-material). The goal is to let gating fail closed so we don't
    spam alerts on LLM outages.
    """
    try:
        result = app_v3.call_llm(
            messages=[{'role': 'user', 'content': point_text}],
            system=MATERIAL_JUDGE_PROMPT,
            tier='standard',
            max_tokens=200,
            anthropic_api_key=anthropic_api_key or '',
        )
        text = result.get('text', '') if isinstance(result, dict) else str(result)
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        parsed = json.loads(text)
        return bool(parsed.get('material'))
    except Exception as e:
        print(f'material judge error (treating as non-material): {e}')
        return False


def _tokens_from_title(title):
    """Lowercased 3+-char tokens from episode title — used for cheap
    guest_match fuzzy dedup."""
    return set(re.findall(r'\b[a-z]{3,}\b', (title or '').lower()))


def _find_duplicate_alert(point, episode, primary_ticker):
    """Look for an existing material alert within DEDUP_WINDOW_DAYS on the
    same ticker. Duplicate if (ticker_overlap >= 1) AND (theme_overlap >= 1
    OR title_token_overlap >= 2). Returns (alert_id, detail_dict) or
    (None, None)."""
    with app_v3.get_db() as (_c, cur):
        cur.execute(
            """
            SELECT id, title, detail, created_at FROM agent_alerts
             WHERE alert_type='podcast_material'
               AND created_at > NOW() - (%s * INTERVAL '1 day')
               AND status IN ('new','actioned')
               AND ticker = %s
            """,
            (DEDUP_WINDOW_DAYS, primary_ticker),
        )
        candidates = cur.fetchall()

    p_themes = {t.lower() for t in (point.get('theme_tags') or []) if t}
    p_tickers = {t.upper() for t in (point.get('tickers') or [])}
    p_title_tokens = _tokens_from_title(episode.get('title'))

    for c in candidates:
        raw_detail = c['detail']
        if isinstance(raw_detail, dict):
            det = raw_detail
        elif raw_detail:
            try:
                det = json.loads(raw_detail)
            except Exception:
                det = {}
        else:
            det = {}
        c_themes = {t.lower() for t in (det.get('themeTags') or []) if t}
        c_tickers = {t.upper() for t in (det.get('tickers') or [])}
        c_title_tokens = _tokens_from_title(det.get('episodeTitle'))

        if not (p_tickers & c_tickers):
            continue
        if (p_themes & c_themes) or (len(p_title_tokens & c_title_tokens) >= 2):
            return c['id'], det
    return None, None


def gate_and_alert_for_episode(episode_id):
    """For all digest_points of this episode that pass candidate + judge,
    set material=true and create/append to agent_alerts rows.

    Called by extractor.extract_from_episode AFTER points are inserted.
    Safe to run on empty/missing episode (nothing happens).
    """
    covered = _load_coverage_universe()
    keywords = _load_watchlist_keywords()
    muted_coverage = _load_muted_coverage()
    import os as _os
    anthropic_key = _os.environ.get('ANTHROPIC_API_KEY', '')

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT * FROM media_episodes WHERE id=%s', (episode_id,))
        episode = cur.fetchone()
        if not episode:
            return
        cur.execute(
            """
            SELECT id, text, tickers, sector_tags, theme_tags, point_order
              FROM media_digest_points
             WHERE episode_id=%s
             ORDER BY point_order
            """,
            (episode_id,),
        )
        points = cur.fetchall()
    if not points:
        return

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT id, name FROM media_feeds WHERE id=%s', (episode['feed_id'],))
        feed = cur.fetchone() or {}

    for p in points:
        point = {
            'text': p['text'],
            'tickers': p['tickers'] or [],
            'sector_tags': p['sector_tags'] or [],
            'theme_tags': p['theme_tags'] or [],
        }
        if not _candidate_material(point, covered, keywords, muted_coverage):
            continue
        if not _judge_material(point['text'], anthropic_key):
            continue

        primary = _primary_ticker(point)
        if not primary:
            continue

        # Mark point material
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute(
                "UPDATE media_digest_points SET material=TRUE WHERE id=%s",
                (p['id'],),
            )

        # Dedup vs existing alerts
        dup_id, dup_detail = _find_duplicate_alert(point, episode, primary)
        if dup_id:
            related = dup_detail.get('relatedEpisodes') or []
            related.append({
                'episodeId': episode['id'],
                'episodeTitle': episode.get('title'),
                'feedName': feed.get('name'),
                'pointText': point['text'],
                'sourceUrl': episode.get('source_url'),
            })
            dup_detail['relatedEpisodes'] = related[:10]
            with app_v3.get_db(commit=True) as (_c, cur):
                cur.execute(
                    "UPDATE agent_alerts SET detail=%s::jsonb WHERE id=%s",
                    (json.dumps(dup_detail), dup_id),
                )
        else:
            alert_id = str(uuid.uuid4())
            title = (point['text'] or '').strip()[:140]
            detail = {
                'pointId': p['id'],
                'episodeId': episode['id'],
                'episodeTitle': episode.get('title'),
                'feedName': feed.get('name'),
                'feedId': feed.get('id'),
                'sourceUrl': episode.get('source_url'),
                'tickers': point['tickers'],
                'sectorTags': point['sector_tags'],
                'themeTags': point['theme_tags'],
                'relatedEpisodes': [],
            }
            with app_v3.get_db(commit=True) as (_c, cur):
                cur.execute(
                    """
                    INSERT INTO agent_alerts (id, alert_type, ticker, title, detail, status, created_at)
                    VALUES (%s, 'podcast_material', %s, %s, %s::jsonb, 'new', NOW())
                    """,
                    (alert_id, primary, title, json.dumps(detail)),
                )
            # Fire push/telegram if enabled. Email digest is batched at 7am.
            try:
                from media_trackers.notifications import notify_new_material_alert
                notify_new_material_alert(
                    ticker=primary,
                    point_text=point['text'],
                    episode_title=episode.get('title') or '',
                    feed_name=feed.get('name') or '',
                    source_url=episode.get('source_url') or '',
                )
            except Exception as e:
                print(f'notify_new_material_alert failed: {e}')

            # Phase 3a: route to analyst team. Create a pending_review
            # investigation row for each analyst covering this ticker.
            try:
                with app_v3.get_db() as (_c, cur):
                    cur.execute(
                        'SELECT id, name FROM analysts WHERE %s = ANY(coverage_tickers)',
                        (primary,),
                    )
                    matching = cur.fetchall() or []
                for a in matching:
                    with app_v3.get_db(commit=True) as (_c, cur):
                        cur.execute('''
                            INSERT INTO analyst_activities
                              (id, analyst_id, activity_type, ticker, status,
                               trigger_source, input, created_at)
                            VALUES (%s, %s, 'investigation', %s, 'pending_review',
                                    'podcast_material', %s::jsonb, NOW())
                        ''', (
                            str(uuid.uuid4()),
                            a['id'],
                            primary,
                            json.dumps({
                                'alertId': alert_id,
                                'pointText': point['text'],
                                'episodeTitle': episode.get('title'),
                                'feedName': feed.get('name'),
                                'sourceUrl': episode.get('source_url'),
                                'tickers': point['tickers'],
                                'themeTags': point['theme_tags'],
                            }),
                        ))
            except Exception as e:
                print(f'analyst routing failed: {e}')
