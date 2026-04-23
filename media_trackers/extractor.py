"""Extract bullet-tagged digest points from episode content via Haiku."""
import json
import uuid

import app_v3
from media_trackers.prompts import EXTRACTION_PROMPT

MAX_EPISODES_PER_BATCH = 3


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
    # call_llm may return str or dict depending on provider; normalize.
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            repaired = app_v3._repair_truncated_json(raw)
            return json.loads(repaired)
        except Exception:
            return {'points': []}


def extract_from_episode(episode_id: str) -> None:
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "UPDATE media_episodes SET status='extracting' WHERE id=%s AND status='new' RETURNING *",
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


def process_extract_batch() -> None:
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM media_episodes
             WHERE status = 'new'
             ORDER BY published_at DESC NULLS LAST
             LIMIT %s
        ''', (MAX_EPISODES_PER_BATCH,))
        ids = [r['id'] for r in cur.fetchall()]
    for eid in ids:
        extract_from_episode(eid)
