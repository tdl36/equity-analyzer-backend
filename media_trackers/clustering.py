"""Weekly theme clustering for media digest points (M8).

Runs Sunday 5am: ask Sonnet to group last 7d material points into 3-8 themes.
Upserts media_theme_clusters rows (wipe+reinsert per week) and tags
media_digest_points.cluster_id.
"""
import json
import uuid
from datetime import date, timedelta

import app_v3

CLUSTER_PROMPT = """You group buy-side podcast bullet points into themes. Given the list below, return 3-8 coherent theme clusters.

Rules:
- Each cluster has 2+ related bullets. Skip singleton clusters.
- theme = short label (2-6 words) like "GLP-1 supply constraints", "ARM server adoption", "rate cut timing"
- summary = 1-2 sentence synthesis of what the bullets collectively say
- primary_tickers = top 1-3 tickers across the bullets in this cluster

Return JSON only:
{"clusters": [
  {"theme": "...", "summary": "...", "primary_tickers": ["..."], "point_ids": ["...", "..."]},
  ...
]}
"""


def run_weekly_clustering() -> None:
    from media_trackers.extractor import _parse_json

    with app_v3.get_db() as (_c, cur):
        cur.execute(
            """
            SELECT id, text, tickers, sector_tags, theme_tags
              FROM media_digest_points
             WHERE material = TRUE
               AND created_at > NOW() - INTERVAL '7 days'
             ORDER BY created_at DESC
             LIMIT 300
            """
        )
        points = cur.fetchall()
    if len(points) < 4:
        print(f'clustering: only {len(points)} material points in last 7d, skipping')
        return

    point_list = []
    for p in points:
        point_list.append({
            'id': p['id'],
            'text': (p['text'] or '')[:400],
            'tickers': p['tickers'] or [],
            'themes': p['theme_tags'] or [],
        })

    user_content = json.dumps(point_list)
    try:
        result = app_v3.call_llm(
            messages=[{'role': 'user', 'content': user_content}],
            system=CLUSTER_PROMPT,
            tier='standard',
            max_tokens=4000,
        )
        parsed = _parse_json(result if isinstance(result, dict) else {'text': result})
    except Exception as e:
        print(f'clustering llm call failed: {e}')
        return

    clusters = parsed.get('clusters') if isinstance(parsed, dict) else []
    if not clusters:
        return

    # week_start = most recent Monday
    today = date.today()
    week_start = today - timedelta(days=today.weekday())

    with app_v3.get_db(commit=True) as (_c, cur):
        # Wipe current week's clusters then re-insert (idempotent weekly run)
        cur.execute("DELETE FROM media_theme_clusters WHERE week_start=%s", (week_start,))
        for c in clusters:
            theme = (c.get('theme') or '').strip()
            if not theme:
                continue
            summary = (c.get('summary') or '').strip()
            tickers = [t.upper() for t in (c.get('primary_tickers') or []) if t][:5]
            point_ids = [pid for pid in (c.get('point_ids') or []) if pid]
            cluster_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO media_theme_clusters (id, theme, summary, point_ids, primary_tickers, week_start)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (cluster_id, theme, summary, point_ids, tickers, week_start),
            )
            if point_ids:
                cur.execute(
                    "UPDATE media_digest_points SET cluster_id=%s WHERE id = ANY(%s)",
                    (cluster_id, point_ids),
                )
