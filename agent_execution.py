"""One-shot durable execution reservation; never retries ambiguous provider calls."""
import json
from uuid import uuid4


def queued_log(recover_single=False):
    entries = [{'event': 'execution_queued', 'agent': 'system',
                'message': 'Queued for one execution attempt.'}]
    if recover_single:
        entries.append({'event': 'recover_single_v1', 'agent': 'system',
                        'message': 'May resume only before the first execution reservation.'})
    return json.dumps(entries)


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


def recover_pending(get_db, run):
    """Discover only new opted-in singles; run must take the atomic claim first.

    Never take over a started attempt. Two-minute grace avoids competing with the
    normal immediate dispatch. Batch jobs are not opted in: their concurrency
    settings are not yet durable. Read parameters from the saved row, not defaults.
    """
    with get_db() as (_, cur):
        cur.execute("""
            SELECT id,ticker,analysis_date,llm_provider,llm_model FROM agent_runs
            WHERE status='queued' AND completed_at IS NULL
              AND created_at < NOW()-INTERVAL '2 minutes'
              AND logs @> '[{"event":"execution_queued"},{"event":"recover_single_v1"}]'::jsonb
              AND NOT logs @> '[{"event":"execution_started"}]'::jsonb
            ORDER BY created_at LIMIT 10
        """)
        rows = list(cur.fetchall())
    for row in rows:
        run(str(row['id']), row['ticker'], str(row['analysis_date']),
            row['llm_provider'], row['llm_model'])
    return len(rows)


def start_recovery(get_db, run):
    import threading
    def loop():
        # Sequential within this recovery loop; graph entry fences all processes.
        pause = threading.Event()
        while not pause.wait(60):
            try:
                recover_pending(get_db, run)
            except Exception as exc:
                print('[Analyst queue recovery]', type(exc).__name__)
    threading.Thread(target=loop, daemon=True, name='analyst-queue-recovery').start()


def enqueue_single(get_db, ticker, date_str, provider, model, request_id=None):
    """One durable job per client request identity; reject changed frozen inputs."""
    from uuid import UUID
    run_id = str(UUID(request_id)) if request_id is not None else str(uuid4())
    with get_db(commit=True) as (_, cur):
        cur.execute("""
            INSERT INTO agent_runs(id,ticker,analysis_date,llm_provider,llm_model,status,logs)
            VALUES(%s,%s,%s,%s,%s,'queued',%s::jsonb)
            ON CONFLICT(id) DO NOTHING RETURNING id
        """, (run_id,ticker,date_str,provider,model,queued_log(recover_single=True)))
        if cur.fetchone():
            return run_id, True
        cur.execute('SELECT ticker,analysis_date,llm_provider,llm_model FROM agent_runs WHERE id=%s',(run_id,))
        saved = cur.fetchone()
        if not saved or (saved['ticker'],str(saved['analysis_date']),saved['llm_provider'],saved['llm_model']) != (ticker,date_str,provider,model):
            raise ValueError('This submission identity belongs to different saved inputs. Inspect the original run.')
        return run_id, False
