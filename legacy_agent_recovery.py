"""Record unresolved legacy execution without replay, cancellation or lost evidence."""
import json
from datetime import datetime, timezone, timedelta

EVENT = 'legacy_outcome_unverified_v1'


def entries(row):
    value = row.get('logs') or []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, list) and all(isinstance(x, dict) for x in value) else None


def eligible(row, now=None):
    logs = entries(row)
    if logs is None or row.get('status') not in ('running', 'queued'):
        return False
    if row.get('completed_at') or row.get('report') or row.get('decision') or row.get('has_decision'):
        return False
    if any(x.get('event') in ('execution_queued', 'execution_started', 'batch_enrolled_v1', 'model_call_v1') for x in logs):
        return False
    now = now or datetime.now(timezone.utc)
    dates = [row.get('created_at')] + [x['ts'] for x in logs if x.get('ts')]
    try:
        parsed = [datetime.fromisoformat(x.replace('Z', '+00:00')) if isinstance(x, str) else x for x in dates]
        parsed = [x.replace(tzinfo=timezone.utc) if x.tzinfo is None else x for x in parsed]
    except (ValueError, TypeError, AttributeError):
        return False
    return bool(parsed) and max(parsed) < now - timedelta(days=1)


def record(get_db, run_id):
    with get_db(commit=True) as (_, cur):
        cur.execute('SELECT status,created_at,completed_at,logs,report,decision FROM agent_runs WHERE id=%s FOR UPDATE', (run_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError('Saved run not found.')
        logs = entries(row)
        if row['status'] == 'attention' and logs and any(x.get('event') == EVENT for x in logs):
            return False
        if not eligible(row):
            raise ValueError('Only old legacy runs without saved outputs or execution enrollment can use this disposition. Inspect the current evidence.')
        event = dict(event=EVENT, agent='system', ts=datetime.now(timezone.utc).isoformat(),
                     message='Legacy provider outcome unresolved. Saved evidence preserved; no cancellation, completion or retry authorized.')
        cur.execute("UPDATE agent_runs SET status='attention', logs=COALESCE(logs,'[]'::jsonb) || %s::jsonb WHERE id=%s", (json.dumps([event]), run_id))
        return True
