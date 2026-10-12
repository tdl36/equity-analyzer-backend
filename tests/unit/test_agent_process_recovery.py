"""Hard process-death checks against a disposable Unix-socket PostgreSQL only.

No application startup, credentials, model adapter or production connection.
The child invokes the actual reservation/recovery/reporter functions, then is
killed without cleanup to model lost ownership rather than a graceful exit.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from agent_execution import enqueue_single, enqueue_batch
from agent_presence import inspect
from tests.unit import test_stock_analysis_postgres as pg


CHILD = r'''
import json, os, sys
from pathlib import Path
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from agent_execution import claim, recover_pending
from agent_presence import Presence
config=json.loads(sys.stdin.readline())
root=Path(config['socket']).resolve()
assert str(root).startswith('/private/tmp/charlie-stock-pg-') or str(root).startswith('/tmp/charlie-stock-pg-')
assert (root/'db'/'postmaster.pid').is_file()
@contextmanager
def db(commit=False):
    connection=psycopg2.connect(host=str(root),port=55493,dbname='postgres',user=config['user'])
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:yield connection,cursor
        if commit:connection.commit()
        else:connection.rollback()
    except Exception:connection.rollback();raise
    finally:connection.close()
print('ready',flush=True)
if sys.stdin.readline().strip()!='go':sys.exit(0)
if config['mode']=='recover':
    attempts=[]
    def dispatch(*args):
        execution=claim(db,args[0])
        if execution:attempts.append({'inputs':args,'execution':execution})
    recover_pending(db,dispatch)
    print(json.dumps(attempts),flush=True)
else:
    execution=claim(db,config['run'])
    reporter=Presence(db,config['run'],execution) if execution and config['mode']=='hold' else None
    print(json.dumps({'execution':execution}),flush=True)
    if reporter:sys.stdin.readline()  # parent kills us; deliberately no finally/close
'''


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class ProcessRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_, c):
            c.execute('CREATE TABLE app_settings(key TEXT PRIMARY KEY,value JSONB,updated_at TIMESTAMP)')
            c.execute('CREATE TABLE agent_runs(id TEXT PRIMARY KEY,status TEXT,logs JSONB,completed_at TIMESTAMP,created_at TIMESTAMP DEFAULT NOW(),ticker TEXT,analysis_date DATE,llm_provider TEXT,llm_model TEXT,report TEXT)')

    @classmethod
    def tearDownClass(cls):
        pg.StockPostgresTests.tearDownClass.__func__(cls)

    def setUp(self):
        with self.db(True) as (_, c):
            c.execute('DELETE FROM agent_runs')
            c.execute('DELETE FROM app_settings')

    def child(self, mode, run=None):
        process=subprocess.Popen([sys.executable, '-u', '-c', CHILD],
            cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def cleanup():
            if process.poll() is None:process.kill()
            process.communicate(timeout=10)
        self.addCleanup(cleanup)
        process.stdin.write(json.dumps(dict(socket=str(self.root),user=os.environ.get('USER','tonydlee'),mode=mode,run=run))+'\n')
        process.stdin.flush()
        # Wait for real process startup, with a timeout rather than timing sleeps.
        import select
        self.assertTrue(select.select([process.stdout],[],[],15)[0], 'Child startup timeout')
        self.assertEqual(process.stdout.readline().strip(),'ready')
        return process

    def go(self, process):
        process.stdin.write('go\n');process.stdin.flush()
        import select
        self.assertTrue(select.select([process.stdout],[],[],15)[0], 'Child result timeout')
        return json.loads(process.stdout.readline())

    def age_jobs(self):
        with self.db(True) as (_, c):
            c.execute("UPDATE agent_runs SET created_at=NOW()-INTERVAL '10 minutes'")

    def test_death_before_reservation_recovers_once_with_frozen_inputs(self):
        run,_=enqueue_single(self.db,'TEST','2026-10-11','fixture','pinned-model')
        abandoned=self.child('hold',run)
        abandoned.kill();abandoned.wait(timeout=10)  # no go: reservation never occurred
        self.age_jobs()
        recovered=self.go(self.child('recover'))
        self.assertEqual(len(recovered),1)
        self.assertEqual(recovered[0]['inputs'],[run,'TEST','2026-10-11','fixture','pinned-model'])
        self.assertEqual(self.go(self.child('recover')),[])
        self.assertIsNone(self.go(self.child('claim',run))['execution'])

    def test_death_after_reservation_preserves_uncertain_batch_slot(self):
        _,runs,_=enqueue_batch(self.db,['ONE','TWO'],'2026-10-11','fixture','pinned-model',1)
        owner=self.child('hold',runs[0]['id'])
        execution=self.go(owner)['execution'];self.assertIsNotNone(execution)
        self.assertEqual(inspect(self.db,runs[0]['id'],[execution])['state'],'reporting')
        owner.kill();owner.wait(timeout=10)
        # Expiry describes absence; it never releases the potentially paid attempt.
        with self.db() as (_, c):
            c.execute('SELECT value,updated_at FROM app_settings WHERE key=%s',('analyst_presence_v1:'+runs[0]['id'],))
            receipt=c.fetchone()
        from agent_presence import describe
        self.assertEqual(describe(receipt,[execution],datetime.now(timezone.utc)+timedelta(seconds=100))['state'],'stale')
        free,_=enqueue_single(self.db,'FREE','2026-10-11','fixture','pinned-model')
        self.age_jobs()
        recovered=self.go(self.child('recover'))
        self.assertEqual([x['inputs'][0] for x in recovered],[free])
        for run in runs:self.assertIsNone(self.go(self.child('claim',run['id']))['execution'])
        with self.db() as (_, c):
            c.execute('SELECT id,status,logs,completed_at FROM agent_runs WHERE id=ANY(%s)',([r['id'] for r in runs],))
            rows={r['id']:r for r in c.fetchall()}
        first=rows[runs[0]['id']]
        self.assertEqual(first['status'],'running');self.assertIsNone(first['completed_at'])
        self.assertEqual(sum(x.get('event')=='execution_started' for x in first['logs']),1)
        self.assertEqual(rows[runs[1]['id']]['status'],'queued')

    def test_independent_processes_cannot_duplicate_reservation(self):
        run,_=enqueue_single(self.db,'TEST','2026-10-11','fixture','pinned-model',str(uuid4()))
        children=[self.child('claim',run) for _ in range(4)]
        for child in children:child.stdin.write('go\n');child.stdin.flush()
        executions=[]
        for child in children:
            output,error=child.communicate(timeout=15)
            self.assertEqual(child.returncode,0,error)
            executions.append(json.loads(output)['execution'])
        self.assertEqual(sum(bool(x) for x in executions),1)

    def test_independent_processes_enforce_shared_batch_limit(self):
        _,runs,_=enqueue_batch(self.db,['ONE','TWO','THREE','FOUR'],'2026-10-11','fixture','pinned-model',2)
        children=[self.child('claim',run['id']) for run in runs]
        for child in children:child.stdin.write('go\n');child.stdin.flush()
        executions=[]
        for child in children:
            output,error=child.communicate(timeout=15)
            self.assertEqual(child.returncode,0,error)
            executions.append(json.loads(output)['execution'])
        self.assertEqual(sum(bool(x) for x in executions),2)
        self.age_jobs()
        self.assertEqual(self.go(self.child('recover')),[])
