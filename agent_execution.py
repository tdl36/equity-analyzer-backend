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
        cur.execute("SELECT logs FROM agent_runs WHERE id=%s",(run_id,))
        saved=cur.fetchone()
        entries=(saved['logs'] or []) if saved else []
        if isinstance(entries,str):entries=json.loads(entries)
        batches=[entry for entry in entries if isinstance(entry,dict) and entry.get('event')=='batch_enrolled_v1']
        if batches:
            if len(batches)!=1:return None
            batch=batches[0];batch_id=batch.get('batchId');limit=batch.get('concurrency')
            if not isinstance(batch_id,str) or type(limit) is not int or not 1<=limit<=3:return None
            # All claimants serialize this batch before counting outstanding attempts.
            # An uncertain running attempt continues to occupy its slot after restart.
            cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(batch_id,))
            cur.execute("SELECT COUNT(*) AS n FROM agent_runs WHERE status='running' AND completed_at IS NULL AND logs @> %s::jsonb",(json.dumps([{'event':'batch_enrolled_v1','batchId':batch_id}]),))
            if cur.fetchone()['n']>=limit:return None
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
    """Discover only newly enrolled recoverable jobs; run must take the atomic claim first.

    Never take over a started attempt. Two-minute grace avoids competing with the
    normal immediate dispatch. New batch jobs retain their concurrency limit; claim enforces it across processes. Read parameters from the saved row, not defaults.
    """
    with get_db() as (_, cur):
        cur.execute("""
            SELECT id,ticker,analysis_date,llm_provider,llm_model FROM agent_runs candidate
            WHERE status='queued' AND completed_at IS NULL
              AND created_at < NOW()-INTERVAL '2 minutes'
              AND logs @> '[{"event":"execution_queued"}]'::jsonb
              AND (logs @> '[{"event":"recover_single_v1"}]'::jsonb OR logs @> '[{"event":"batch_enrolled_v1"}]'::jsonb)
              AND NOT logs @> '[{"event":"execution_started"}]'::jsonb
              AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements(candidate.logs) b
                WHERE b->>'event'='batch_enrolled_v1'
                  AND (SELECT COUNT(*) FROM agent_runs active
                       WHERE active.status='running' AND active.completed_at IS NULL
                         AND active.logs @> jsonb_build_array(jsonb_build_object('event','batch_enrolled_v1','batchId',b->>'batchId')))
                      >= CASE WHEN b->>'concurrency' ~ '^[1-3]$' THEN (b->>'concurrency')::int ELSE 0 END)
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


def enqueue_batch(get_db, tickers, date_str, provider, model, concurrency, request_id=None):
    """Atomically retain the batch's frozen inputs, children and retry receipt."""
    from uuid import UUID
    batch_id = str(UUID(request_id)) if request_id is not None else str(uuid4())
    key = 'analyst_batch_v1:' + batch_id
    inputs = dict(tickers=tickers,date=date_str,provider=provider,model=model,concurrency=concurrency)
    with get_db(commit=True) as (_, cur):
        cur.execute("INSERT INTO app_settings(key,value,updated_at) VALUES(%s,'{}'::jsonb,NOW()) ON CONFLICT(key) DO NOTHING RETURNING key",(key,))
        created = bool(cur.fetchone())
        cur.execute('SELECT value FROM app_settings WHERE key=%s FOR UPDATE',(key,))
        row = cur.fetchone()
        if not created:
            receipt = row['value'] if isinstance(row['value'],dict) else json.loads(row['value'])
            if receipt.get('inputs') != inputs or not receipt.get('runs'):
                raise ValueError('Batch identity belongs to different or unavailable saved inputs. Inspect the original batch.')
            cur.execute('SELECT id::text AS id FROM agent_runs WHERE id::text=ANY(%s)',([r['id'] for r in receipt['runs']],))
            if {r['id'] for r in cur.fetchall()} != {r['id'] for r in receipt['runs']}:
                raise ValueError('A saved batch child is unavailable. Inspect the original batch; no replacement was created.')
            return batch_id,receipt['runs'],False
        runs=[]
        for ticker in tickers:
            run_id=str(uuid4())
            entries=json.loads(queued_log())+[dict(event='batch_enrolled_v1',batchId=batch_id,concurrency=concurrency,agent='system',message='Batch settings saved; only unstarted jobs may recover.')]
            cur.execute("INSERT INTO agent_runs(id,ticker,analysis_date,llm_provider,llm_model,status,report,logs) VALUES(%s,%s,%s,%s,%s,'queued',%s,%s::jsonb)",(run_id,ticker,date_str,provider,model,json.dumps({'batchId':batch_id}),json.dumps(entries)))
            runs.append(dict(id=run_id,ticker=ticker))
        cur.execute('UPDATE app_settings SET value=%s::jsonb,updated_at=NOW() WHERE key=%s',(json.dumps(dict(inputs=inputs,runs=runs)),key))
        return batch_id,runs,True


def execution_evidence(logs):
    """Expose bounded reservation metadata, never infer provider outcome from logs."""
    from uuid import UUID
    if isinstance(logs, str):
        try:
            logs = json.loads(logs)
        except ValueError:
            logs = []
    entries = [x for x in logs if isinstance(x, dict)] if isinstance(logs, list) else []
    reservations = [x for x in entries if x.get('event') == 'execution_started']
    ids = []
    for entry in reservations:
        try:
            identifier = str(UUID(str(entry.get('executionId'))))
        except (ValueError, TypeError, AttributeError):
            continue
        if identifier not in ids:
            ids.append(identifier)
    return {
        'enrolled': any(x.get('event') == 'execution_queued' for x in entries),
        'reservationCount': len(reservations),
        'reservationIds': ids[:5],
        'reservationIdsTruncated': len(ids) > 5,
        'providerOutcome': 'unverified',
    }
