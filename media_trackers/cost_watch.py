"""Weekly media-tracker cost watch (M8).

Runs daily 11pm. Sums media_episodes.cost_usd over last 7 days:
  - >= soft cap (default $15/wk, read from app_settings.media_cost_weekly_warn_usd)
    -> create a 'system' alert row in agent_alerts
  - >= hard cap ($50/wk) -> ALSO flip app_settings.media_scheduler_enabled=false
    to auto-pause the scheduler
"""
import json
import uuid

import app_v3

SOFT_WARN_DEFAULT = 15.0
HARD_CAP = 50.0  # USD per week


def check_cost_warning() -> None:
    with app_v3.get_db() as (_c, cur):
        cur.execute(
            """
            SELECT COALESCE(SUM(cost_usd), 0) AS total
              FROM media_episodes
             WHERE created_at > NOW() - INTERVAL '7 days'
            """
        )
        row = cur.fetchone()
    total = float(row['total'] or 0) if row else 0.0

    soft = SOFT_WARN_DEFAULT
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key='media_cost_weekly_warn_usd'")
            r = cur.fetchone()
        if r and r['value'] not in (None, ''):
            raw = r['value']
            if isinstance(raw, (int, float)):
                soft = float(raw)
            else:
                try:
                    soft = float(str(raw).strip('"'))
                except (ValueError, TypeError):
                    soft = SOFT_WARN_DEFAULT
    except Exception:
        pass

    if total < soft:
        return  # all good

    detail = {'weeklyCostUsd': round(total, 2), 'softWarnUsd': soft, 'hardCapUsd': HARD_CAP}
    alert_title = f'Media tracker spent ${total:.2f} this week (soft cap ${soft:.0f})'

    if total >= HARD_CAP:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute(
                """
                INSERT INTO app_settings (key, value) VALUES ('media_scheduler_enabled', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                ('false',),
            )
        alert_title = f'Media tracker auto-paused at ${total:.2f} (hard cap ${HARD_CAP:.0f})'
        detail['hardCapped'] = True

    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            """
            INSERT INTO agent_alerts (id, alert_type, ticker, title, detail, status, created_at)
            VALUES (%s, 'system', '', %s, %s::jsonb, 'new', NOW())
            ON CONFLICT DO NOTHING
            """,
            (str(uuid.uuid4()), alert_title, json.dumps(detail)),
        )
