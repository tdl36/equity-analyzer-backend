"""Recover manual meeting jobs under connection-scoped, cross-process ownership."""
import json
import threading
import uuid


def obj(value):
    return value if isinstance(value, dict) else json.loads(value or '{}')


def execute(get_db, job_id, run, recovery=False):
    from amendment_ownership import worker_session
    with worker_session(get_db, 'manual-meeting:'+job_id) as owned:
        if not owned:
            return False
        with get_db(commit=True) as (_, cur):
            cur.execute("SELECT input,result,status FROM mp_jobs WHERE id=%s AND stage='pipeline' FOR UPDATE", (job_id,))
            row=cur.fetchone()
            if not row or row['status']!='running':
                return False
            inp=obj(row['input']); checkpoint=obj(row['result'])
            attempts=int(inp.get('recoveryAttempts',0))
            if recovery:
                if attempts>=2:
                    cur.execute("UPDATE mp_jobs SET status='failed',error='Automatic recovery limit reached. Retry this meeting pack from its checkpoint.',updated_at=NOW() WHERE id=%s", (job_id,))
                    return False
                inp['recoveryAttempts']=attempts+1
            inp['workerToken']=str(uuid.uuid4())
            inp['recoveryEnabled']=True
            cur.execute("UPDATE mp_jobs SET input=%s::jsonb,error=NULL,updated_at=NOW() WHERE id=%s", (json.dumps(inp),job_id))
        run(inp,checkpoint)
        return True


def start(get_db, resume, has_key):
    def loop():
        while True:
            try:
                if has_key():
                    with get_db() as (_,cur):
                        cur.execute("SELECT id FROM mp_jobs WHERE stage='pipeline' AND status='running' AND input->>'recoveryEnabled'='true' AND updated_at<NOW()-INTERVAL '60 seconds' ORDER BY created_at LIMIT 20")
                        jobs=[r['id'] for r in cur.fetchall()]
                    for jid in jobs:
                        execute(get_db,jid,lambda inp,cp:resume(jid,inp,cp),recovery=True)
            except Exception as exc:
                print('[Manual meeting recovery]',type(exc).__name__)
            threading.Event().wait(30)
    threading.Thread(target=loop,daemon=True,name='manual-meeting-recovery').start()
