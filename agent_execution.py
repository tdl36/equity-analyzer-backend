"""One-shot durable execution reservation; never retries ambiguous provider calls."""
import json
from uuid import uuid4


def queued_log():
    return json.dumps([{'event': 'execution_queued', 'agent': 'system',
                        'message': 'Queued for one execution attempt.'}])


def claim(get_db, run_id):
    """Atomically reserve only newly enrolled queued runs across processes.

    No lease expiry or automatic takeover: the graph may have sent paid requests.
    Legacy rows without the enrollment marker require separate reconciliation.
    """
    attempt = str(uuid4())
    with get_db(commit=True) as (_, cur):
        cur.execute("""
            UPDATE agent_runs SET status='running', logs=COALESCE(logs,'[]'::jsonb) ||
                jsonb_build_array(jsonb_build_object('event','execution_started',
                    'executionId',%s::text,'agent','system',
                    'message','Execution reserved. Interrupted attempts require reconciliation before retry.',
                    'ts',NOW() AT TIME ZONE 'UTC'))
            WHERE id=%s AND status='queued' AND completed_at IS NULL
              AND logs @> '[{"event":"execution_queued"}]'::jsonb
              AND NOT logs @> '[{"event":"execution_started"}]'::jsonb
            RETURNING id
        """, (attempt, run_id))
        return attempt if cur.fetchone() else None
