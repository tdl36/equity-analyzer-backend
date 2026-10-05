"""Integration tests in a newly initialized temporary PostgreSQL cluster ONLY.
Never reads DATABASE_URL and cannot connect to the user's or production database.
"""
import copy
import json
import os
import subprocess
import tempfile
import threading
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from flask import Flask
import psycopg2
from psycopg2.extras import RealDictCursor
from thesis_imports import create_blueprint
from tests.unit.test_thesis_imports import sample

PG = Path('/opt/homebrew/opt/postgresql@16/bin')

@unittest.skipUnless((PG/'initdb').exists(), 'Disposable PostgreSQL binaries unavailable')
class PostgresDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='charlie-thesis-pg-',dir='/tmp')
        cls.root=Path(cls.temp.name);cls.data=cls.root/'db'
        subprocess.run([str(PG/'initdb'),'-D',str(cls.data),'-A','trust','--no-locale','-E','UTF8'],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([str(PG/'pg_ctl'),'-D',str(cls.data),'-l',str(cls.root/'server.log'),'-o',f"-k {cls.root} -h '' -p 55491",'-w','start'],check=True,stdout=subprocess.DEVNULL)
        cls.dsn=f'host={cls.root} port=55491 dbname=postgres user={os.environ.get("USER", "tonydlee")}'
        @contextmanager
        def db(commit=False):
            conn=psycopg2.connect(cls.dsn)
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    yield conn,cur
                if commit:conn.commit()
                else:conn.rollback()
            except Exception:
                conn.rollback();raise
            finally:conn.close()
        cls.db=staticmethod(db)
        with db(True) as (_,cur):
            cur.execute('''CREATE TABLE portfolio_analyses(ticker VARCHAR(20) UNIQUE,company TEXT,analysis JSONB,updated_at TIMESTAMP DEFAULT NOW());
                CREATE TABLE thesis_snapshots(id SERIAL PRIMARY KEY,ticker TEXT,snapshot_type TEXT,thesis_summary TEXT,pillar_count INT,conviction TEXT,raw_snapshot JSONB);''')
        cls.app=Flask(__name__);cls.app.register_blueprint(create_blueprint(db));cls.app.config['TESTING']=True
        cls.app.test_client().get('/api/thesis-imports')

    @classmethod
    def tearDownClass(cls):
        subprocess.run([str(PG/'pg_ctl'),'-D',str(cls.data),'-m','immediate','-w','stop'],check=True,stdout=subprocess.DEVNULL)
        cls.temp.cleanup()

    def setUp(self):
        self.client=self.app.test_client();self.ticker='T'+uuid.uuid4().hex[:12].upper()
        self.p=sample(self.ticker)

    def stage(self,p=None):
        r=self.client.post('/api/thesis-imports',json=p or self.p)
        self.assertIn(r.status_code,(200,201),r.json)
        return self.client.get('/api/thesis-imports/'+r.json['id']).json

    def approve(self,d):
        return self.client.post('/api/thesis-imports/'+d['id']+'/approve',json={'confirm':True,'ticker':self.ticker,'fingerprint':d['fingerprint']})

    def scalar(self,sql,args=()):
        with self.db() as (_,cur):
            cur.execute(sql,args);return list(cur.fetchone().values())[0]

    def test_initial_duplicate_restart_and_future_upgrade(self):
        d=self.stage();self.assertIsNone(d['baseline']);self.assertEqual(self.stage()['id'],d['id'])
        self.assertEqual(self.scalar('SELECT COUNT(*) FROM portfolio_analyses WHERE ticker=%s',(self.ticker,)),0)
        # Recreate blueprint to demonstrate durable restart, not an in-memory draft.
        other=Flask('restart');other.register_blueprint(create_blueprint(self.db))
        self.assertEqual(other.test_client().get('/api/thesis-imports/'+d['id']).json['status'],'pending')
        self.assertEqual(self.approve(d).status_code,200);self.assertTrue(self.approve(d).json['replayed'])
        self.assertEqual(self.scalar('SELECT COUNT(*) FROM thesis_revisions WHERE ticker=%s',(self.ticker,)),1)
        exported=self.client.get('/api/thesis-imports/prepare/'+self.ticker).json['package']
        self.assertEqual(exported['baseline']['mode'],'upgrade')
        exported['analysis']['thesis']['summary']='Changed conviction after new evidence'
        next_d=self.stage(exported);self.assertTrue(next_d['sections'][0]['changed'])
        self.assertEqual(self.approve(next_d).status_code,200)
        self.assertEqual(self.scalar('SELECT COUNT(*) FROM thesis_revisions WHERE ticker=%s',(self.ticker,)),3)
        old=self.scalar("SELECT snapshot FROM thesis_revisions WHERE ticker=%s AND source='import_baseline'",(self.ticker,))
        self.assertEqual(old['thesis']['summary'],'A conditional investment hypothesis.')
        live=self.scalar('SELECT analysis FROM portfolio_analyses WHERE ticker=%s',(self.ticker,))
        self.assertEqual(live['thesis']['summary'],'Changed conviction after new evidence')
        self.assertEqual(live['signposts'][0]['metric'],'Retention')
        self.assertEqual(len(live['history']),1)

    def test_guards_confirmation_ticker_baseline_and_dismissal(self):
        d=self.stage()
        url='/api/thesis-imports/'+d['id']+'/approve'
        self.assertEqual(self.client.post(url,json={}).status_code,400)
        self.assertEqual(self.client.post(url,json={'confirm':True,'ticker':'OTHER','fingerprint':d['fingerprint']}).status_code,409)
        with self.db(True) as (_,cur):
            cur.execute('INSERT INTO portfolio_analyses VALUES(%s,%s,%s::jsonb,NOW())',(self.ticker,'Other',json.dumps(self.p['analysis'])))
        self.assertTrue(self.client.get('/api/thesis-imports/'+d['id']).json['stale'])
        self.assertEqual(self.approve(d).status_code,409)
        self.assertEqual(self.scalar('SELECT COUNT(*) FROM thesis_revisions WHERE ticker=%s',(self.ticker,)),0)
        fresh=copy.deepcopy(self.p);fresh['baseline']={'expectedNoExistingThesis':True};self.assertEqual(self.client.post('/api/thesis-imports',json=fresh).status_code,409)
        self.assertEqual(self.client.post('/api/thesis-imports/'+d['id']+'/dismiss',json={}).status_code,200)
        self.assertEqual(self.approve(d).status_code,409)

    def test_prepared_baseline_rejects_changes_before_upload(self):
        p=self.client.get('/api/thesis-imports/prepare/'+self.ticker).json['package']
        self.p['baseline']=p['baseline']
        with self.db(True) as (_,cur):
            cur.execute('INSERT INTO portfolio_analyses VALUES(%s,%s,%s::jsonb,NOW())',(self.ticker,'Other','{}'))
        self.assertEqual(self.client.post('/api/thesis-imports',json=self.p).status_code,409)

    def test_journal_failure_rolls_back_thesis_and_approval(self):
        d=self.stage()
        with self.db(True) as (_,cur):
            cur.execute("ALTER TABLE thesis_revisions ADD CONSTRAINT fail_test CHECK(ticker <> %s)",(self.ticker,))
        try:
            self.assertEqual(self.approve(d).status_code,503)
            self.assertEqual(self.scalar('SELECT COUNT(*) FROM portfolio_analyses WHERE ticker=%s',(self.ticker,)),0)
            self.assertEqual(self.client.get('/api/thesis-imports/'+d['id']).json['status'],'pending')
        finally:
            with self.db(True) as (_,cur):cur.execute('ALTER TABLE thesis_revisions DROP CONSTRAINT fail_test')
        self.assertEqual(self.approve(d).status_code,200)

    def test_concurrent_initial_approvals_allow_only_one_candidate(self):
        a=self.stage();self.p['analysis']['thesis']['summary']='Second candidate';b=self.stage()
        results=[]
        def accept(d):
            with self.app.test_client() as c:
                results.append(c.post('/api/thesis-imports/'+d['id']+'/approve',json={'confirm':True,'ticker':self.ticker,'fingerprint':d['fingerprint']}).status_code)
        workers=[threading.Thread(target=accept,args=(d,)) for d in (a,b)]
        for t in workers:t.start()
        for t in workers:t.join(15)
        self.assertEqual(sorted(results),[200,409])
        self.assertEqual(self.scalar('SELECT COUNT(*) FROM thesis_revisions WHERE ticker=%s',(self.ticker,)),1)

    def test_concurrent_duplicate_uploads_share_a_receipt(self):
        ids=[]
        def upload():
            with self.app.test_client() as c:ids.append(c.post('/api/thesis-imports',json=self.p).json['id'])
        workers=[threading.Thread(target=upload) for _ in range(2)]
        for t in workers:t.start()
        for t in workers:t.join(15)
        self.assertEqual(len(ids),2);self.assertEqual(len(set(ids)),1)
