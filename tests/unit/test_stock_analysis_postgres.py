"""Native worker and Studio persistence against an isolated disposable cluster only."""
import base64
import json
import os
import subprocess
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from flask import Flask
import psycopg2
from psycopg2.extras import RealDictCursor
from company_research import create_blueprint
from tests.unit.test_stock_analysis import SOURCE,result,GROUPS
PG=Path('/opt/homebrew/opt/postgresql@16/bin')

@unittest.skipUnless((PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class StockPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='charlie-stock-pg-',dir='/tmp');cls.root=Path(cls.temp.name)
        subprocess.run([str(PG/'initdb'),'-D',str(cls.root/'db'),'-A','trust','--no-locale','-E','UTF8'],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([str(PG/'pg_ctl'),'-D',str(cls.root/'db'),'-l',str(cls.root/'server.log'),'-o',f"-k {cls.root} -h '' -p 55493",'-w','start'],check=True,stdout=subprocess.DEVNULL)
        @contextmanager
        def db(commit=False):
            conn=psycopg2.connect(host=str(cls.root),port=55493,dbname='postgres',user=os.environ.get('USER','tonydlee'))
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:yield conn,cur
                if commit:conn.commit()
                else:conn.rollback()
            except Exception:conn.rollback();raise
            finally:conn.close()
        cls.db=staticmethod(db)
        with db(True) as (_,cur):
            cur.execute('''CREATE TABLE mp_companies(id SERIAL PRIMARY KEY,ticker TEXT UNIQUE,name TEXT);
            CREATE TABLE document_files(ticker TEXT,filename TEXT,file_data TEXT,file_type TEXT,metadata JSONB);
            CREATE TABLE investment_case_versions(ticker TEXT,revision INTEGER,body JSONB);
            CREATE TABLE studio_outputs(id SERIAL PRIMARY KEY,title TEXT,type TEXT,status TEXT,source_config JSONB,settings JSONB,content JSONB);''')

    @classmethod
    def tearDownClass(cls):
        subprocess.run([str(PG/'pg_ctl'),'-D',str(cls.root/'db'),'-m','immediate','-w','stop'],check=True,stdout=subprocess.DEVNULL);cls.temp.cleanup()

    def setUp(self):
        self.ticker='T'+uuid.uuid4().hex[:10].upper();self.calls=[]
        with self.db(True) as (_,cur):
            cur.execute('INSERT INTO document_files VALUES(%s,%s,%s,%s,%s::jsonb)',(self.ticker,SOURCE['filename'],base64.b64encode(SOURCE['text'].encode()).decode(),'txt','{}'))
        def ask(prompt,key,tokens,ident,stage):
            self.calls.append(stage)
            if getattr(self,'simulate_known',False):
                from company_research import KnownResponseError
                raise KnownResponseError('Research response exceeded its output bound. Inspect the saved stages before retrying.')
            if getattr(self,'simulate_failure',False):raise TimeoutError('Synthetic timeout')
            if stage.startswith('report'):
                raw=result(GROUPS[int(stage[-1])])
                # Use the worker's verified frozen source identity, never an invented identity.
                with self.db() as (_,cur):
                    cur.execute('SELECT sources FROM stock_analysis_runs WHERE id=%s',(ident,));sid=cur.fetchone()['sources'][0]['id']
                for c in raw['citations']:
                    for e in c['evidence']:e['sourceId']=sid
                return raw
            return {'findings':[{'claimId':'/summary/one_liner','status':'supported','reason':'Synthetic matching-source review.'}]}
        self.ask=ask
        app=Flask('stock-test');app.config['TESTING']=True
        app.register_blueprint(create_blueprint(self.db,ask,lambda k:'synthetic',lambda:'synthetic-model',studio=True));self.client=app.test_client()
        self.thread=patch('company_research.threading.Thread').start();self.addCleanup(patch.stopall)
        patch.dict('sys.modules',{'notegen':SimpleNamespace(extract_file_text=lambda d,max_chars:SOURCE['text'])}).start()
        self.payload={'requestId':str(uuid.uuid4()),'filenames':[SOURCE['filename']],'revision':0,'confirmed':True,'mode':'snapshot'}
        self.url='/api/research/stock-analysis/'+self.ticker
    def start(self,payload=None):
        p=payload or self.payload;r=self.client.post(self.url,json=p);self.assertEqual(r.status_code,202,r.json)
        self.thread.call_args.kwargs['target'](*self.thread.call_args.kwargs['args'])
        return self.client.get('/api/research/stock-analysis-run/'+p['requestId']).json
    def test_end_to_end_persistence_idempotence_canonical_company_and_visual(self):
        r=self.start();self.assertEqual(r['status'],'complete',r.get('error'));self.assertEqual(len(r['state']['completed']),12)
        self.assertIsInstance(r['input']['companyId'],int);self.assertNotIn('text',r['sources'][0])
        self.assertTrue(self.client.post(self.url,json=self.payload).json['replayed']);self.assertEqual(len(self.calls),7)
        self.assertEqual(self.client.post(self.url,json={**self.payload,'mode':'deep'}).status_code,409)
        path='/api/research/stock-analysis-run/'+r['id']+'/infographic'
        self.assertFalse(self.client.get(path).json['available'])
        self.assertEqual(self.client.post(path,json={}).status_code,400)
        visual=self.client.post(path,json={'verifiedFigures':True});self.assertEqual(visual.status_code,201,visual.json)
        again=self.client.post(path,json={'verifiedFigures':True});self.assertTrue(again.json['replayed']);self.assertEqual(visual.json['id'],again.json['id'])
        self.assertIn('From business model',self.client.get(path).json['html'])
        self.assertEqual(len(self.calls),7)
        # A recreated server reads the same stored report and infographic.
        restart=Flask('restart');restart.register_blueprint(create_blueprint(self.db,self.ask,lambda k:'synthetic',lambda:'synthetic-model',studio=True))
        self.assertEqual(restart.test_client().get(path).json['id'],visual.json['id'])
        with self.db() as (_,cur):
            cur.execute('SELECT COUNT(*) AS n FROM investment_case_versions WHERE ticker=%s',(self.ticker,));self.assertEqual(cur.fetchone()['n'],0)
        next_payload={**self.payload,'requestId':str(uuid.uuid4()),'mode':'update','priorId':r['id']}
        next_r=self.start(next_payload);self.assertEqual(next_r['status'],'complete',next_r.get('error'))
        self.assertEqual(next_r['state']['comparison']['priorId'],r['id']);self.assertEqual(next_r['state']['comparison']['sections'],[])
    def test_unknown_outcome_requires_explicit_retry_and_rechecks_sources(self):
        self.simulate_failure=True;r=self.start();self.assertEqual(r['status'],'attention');self.assertEqual(r['state']['inFlight'],'report-0')
        path='/api/research/stock-analysis-run/'+r['id']
        self.assertEqual(self.client.post(path+'/resume',json={}).status_code,409)
        self.assertEqual(self.client.post(path+'/resume',json={'acknowledgeRetry':True}).status_code,200)
        with self.db(True) as (_,cur):cur.execute("UPDATE document_files SET metadata='{"+'"usage":"reference_only"'+"}'::jsonb WHERE ticker=%s",(self.ticker,))
        self.thread.call_args.kwargs['target'](*self.thread.call_args.kwargs['args'])
        changed=self.client.get(path).json;self.assertEqual(changed['status'],'attention');self.assertIn('permission',changed['error']);self.assertEqual(len(self.calls),1)
    def test_prior_issuer_and_invalid_input_cannot_dispatch(self):
        for p in ({**self.payload,'mode':'invalid'},{**self.payload,'confirmed':False},{**self.payload,'mode':'update'}):
            self.assertEqual(self.client.post(self.url,json=p).status_code,400)
        self.assertEqual(self.client.post(self.url,json={**self.payload,'priorId':str(uuid.uuid4())}).status_code,409)
        self.assertEqual(self.thread.call_count,0)
    def test_known_provider_response_can_resume_but_is_counted(self):
        self.simulate_known=True;r=self.start();self.assertEqual(r['status'],'attention')
        self.assertNotIn('inFlight',r['state']);self.assertEqual(r['state']['knownFailure']['stage'],'report-0')
        path='/api/research/stock-analysis-run/'+r['id']
        self.assertEqual(self.client.post(path+'/resume',json={}).status_code,200)
        self.simulate_known=False
        self.thread.call_args.kwargs['target'](*self.thread.call_args.kwargs['args'])
        r=self.client.get(path).json;self.assertEqual(r['status'],'complete');self.assertEqual(r['state']['retries'],['report-0'])
    def test_legacy_output_limit_reservation_is_not_network_uncertainty(self):
        self.simulate_failure=True;r=self.start();path='/api/research/stock-analysis-run/'+r['id']
        with self.db(True) as (_,cur):
            cur.execute('UPDATE stock_analysis_runs SET error=%s WHERE id=%s',('Research response exceeded its output bound. Inspect the saved stages before retrying.',r['id']))
        self.assertEqual(self.client.post(path+'/resume',json={}).status_code,200)
        self.assertEqual(self.client.get(path).json['state']['retries'],['report-0'])
