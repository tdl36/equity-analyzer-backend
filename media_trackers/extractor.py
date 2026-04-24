"""Extract bullet-tagged digest points from episode content via Haiku."""
import json
import uuid

import app_v3
from media_trackers.prompts import EXTRACTION_PROMPT

MAX_EPISODES_PER_BATCH = 10


def _call_haiku(content: str) -> dict:
    """Call Claude Haiku with the extraction prompt. Returns parsed JSON."""
    raw = app_v3.call_llm(
        messages=[{'role': 'user', 'content': content[:30000]}],
        system=EXTRACTION_PROMPT,
        tier='fast',
        max_tokens=2000,
    )
    return _parse_json(raw)


def _parse_json(raw) -> dict:
    # call_llm returns {text, usage, provider, model} — extract the text.
    if isinstance(raw, dict):
        if 'points' in raw:
            return raw  # already-parsed result
        raw = raw.get('text') or raw.get('content') or ''
    if not isinstance(raw, str):
        raw = str(raw)
    # Strip markdown code fence if Haiku wrapped the JSON.
    stripped = raw.strip()
    if stripped.startswith('```'):
        # Remove opening fence (```json or ```)
        first_nl = stripped.find('\n')
        if first_nl > 0:
            stripped = stripped[first_nl + 1:]
        # Remove closing fence
        if stripped.endswith('```'):
            stripped = stripped[:-3].rstrip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            repaired = app_v3._repair_truncated_json(stripped)
            return json.loads(repaired)
        except Exception:
            return {'points': []}


def extract_from_episode(episode_id: str) -> None:
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "UPDATE media_episodes SET status='extracting' WHERE id=%s AND status='extracting' RETURNING *",
            (episode_id,))
        ep = cur.fetchone()
    if not ep:
        return

    content = ep['transcript'] or ep['show_notes']
    if not content:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute(
                "UPDATE media_episodes SET status='skipped', error_message='no content' WHERE id=%s",
                (episode_id,))
        return

    try:
        result = _call_haiku(content)
    except Exception as e:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute(
                "UPDATE media_episodes SET status='failed', error_message=%s WHERE id=%s",
                (str(e)[:500], episode_id))
        return

    points = result.get('points') or []
    with app_v3.get_db(commit=True) as (_c, cur):
        for i, p in enumerate(points):
            cur.execute('''
                INSERT INTO media_digest_points
                    (id, episode_id, point_order, text, tickers, sector_tags, theme_tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                str(uuid.uuid4()), episode_id, i,
                (p.get('text') or '')[:2000],
                p.get('tickers') or [],
                p.get('sector_tags') or [],
                p.get('theme_tags') or [],
            ))
        cur.execute("UPDATE media_episodes SET status='done' WHERE id=%s", (episode_id,))

    # Material gating + alerts (non-fatal: episode is already marked done)
    try:
        from media_trackers.material import gate_and_alert_for_episode
        gate_and_alert_for_episode(episode_id)
    except Exception as e:
        print(f'material gating failed for {episode_id}: {e}')


def process_extract_batch() -> None:
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM media_episodes
             WHERE status = 'extracting'
             ORDER BY published_at DESC NULLS LAST
             LIMIT %s
        ''', (MAX_EPISODES_PER_BATCH,))
        ids = [r['id'] for r in cur.fetchall()]
    for eid in ids:
        extract_from_episode(eid)
