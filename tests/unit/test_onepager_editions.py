"""Edition persistence on an isolated disposable PostgreSQL cluster, never live data."""
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from flask import Flask
from onepager_editions import create_blueprint
from onepager_edition_contract import baseline
from tests.unit.test_onepager_edition_contract import edition_fixture, AID, EID
from tests.unit import test_stock_analysis_postgres as pg_tests


@unittest.skipUnless((pg_tests.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class EditionPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg_tests.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_, c):
            c.execute('''CREATE TABLE stock_analysis_runs(id TEXT PRIMARY KEY,ticker TEXT,status TEXT,input JSONB,baseline JSONB,state JSONB,sources JSONB);
                CREATE TABLE mp_jobs(id TEXT PRIMARY KEY,ticker TEXT,status TEXT,stage TEXT,input JSONB,result JSONB);
                CREATE TABLE onepager_corrections(id UUID PRIMARY KEY,run_id TEXT,fingerprint TEXT,payload_hash TEXT,proposal JSONB);''')

    @classmethod
    def tearDownClass(cls):
        pg_tests.StockPostgresTests.tearDownClass.__func__(cls)

    def setUp(self):
        run, assignment, body, corrections, docs, financials = edition_fixture()
        assignment['result']['visualFundamentals'] = financials
        expected = baseline(run, assignment, financials)
        body.update({k: v for k, v in expected.items() if k != 'ticker'})
        self.run, self.assignment, self.body = run, assignment, body
        with self.db(True) as (_, c):
            c.execute('DROP TABLE IF EXISTS onepager_editions')
            c.execute('DELETE FROM stock_analysis_runs; DELETE FROM mp_jobs; DELETE FROM onepager_corrections; DELETE FROM document_files')
            c.execute('INSERT INTO stock_analysis_runs VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb)',
                      tuple([run[k] for k in ['id','ticker','status']] + [json.dumps(run[k]) for k in ['input','baseline','state','sources']]))
            c.execute('INSERT INTO mp_jobs VALUES(%s,%s,%s,%s,%s::jsonb,%s::jsonb)',
                      (AID, 'TEST', 'complete', 'research_assignment', json.dumps(assignment['input']), json.dumps(assignment['result'])))
            for row in corrections:
                c.execute('INSERT INTO onepager_corrections VALUES(%s,%s,%s,%s,%s::jsonb)',
                          (row['id'], row['run_id'], row['fingerprint'], row['payload_hash'], json.dumps(row['proposal'])))
            for doc in docs.values():
                c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s::jsonb)',
                          ('TEST', doc['filename'], doc['file_data'], json.dumps(doc['metadata'])))
        self.url = f'/api/research/assignments/{AID}/onepager-editions'

    def client(self):
        app = Flask(__name__); app.config['TESTING'] = True
        app.register_blueprint(create_blueprint(self.db)); return app.test_client()

    def test_read_is_private_and_never_creates_schema_or_work(self):
        r = self.client().get(self.url)
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(r.json['editions'], [])
        self.assertEqual(r.headers['Cache-Control'], 'private, no-store')
        self.assertNotIn('Synthetic source', str(r.json))
        with self.db() as (_, c):
            c.execute("SELECT to_regclass('onepager_editions') AS name")
            self.assertIsNone(c.fetchone()['name'])

    def test_concurrent_replay_restart_and_original_retention(self):
        def submit(_):
            r = self.client().post(self.url, json=self.body)
            self.assertIn(r.status_code, (200, 201), r.json); return r.json
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(submit, range(2)))
        self.assertEqual(sum(x['reused'] for x in responses), 1)
        self.assertTrue(submit(0)['reused'])
        records = self.client().get(self.url).json['editions']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['record']['status'], 'pending')
        self.assertFalse(records[0]['stale'])
        conflict = self.client().post(self.url, json={**self.body, 'parentId': EID})
        self.assertEqual(conflict.status_code, 409)
        with self.db() as (_, c):
            c.execute('SELECT state FROM stock_analysis_runs'); self.assertEqual(c.fetchone()['state'], self.run['state'])
            c.execute('SELECT result FROM mp_jobs'); self.assertEqual(c.fetchone()['result'], self.assignment['result'])
            c.execute('SELECT proposal FROM onepager_corrections'); self.assertEqual(c.fetchone()['proposal']['review'], 'pending')

    def test_stale_financials_and_restricted_sources_roll_back(self):
        with self.db(True) as (_, c):
            c.execute("UPDATE document_files SET metadata=%s::jsonb", (json.dumps({'usage': 'reference_only'}),))
        r = self.client().post(self.url, json=self.body)
        self.assertEqual(r.status_code, 400)
        with self.db() as (_, c):
            c.execute("SELECT to_regclass('onepager_editions') AS name"); self.assertIsNone(c.fetchone()['name'])
        with self.db(True) as (_, c):
            c.execute("UPDATE document_files SET metadata='{}'::jsonb")
            c.execute("UPDATE mp_jobs SET result=jsonb_set(result,'{visualFundamentals,price}','101'::jsonb)")
        self.assertEqual(self.client().post(self.url, json=self.body).status_code, 400)

    def test_saved_pending_is_stale_after_snapshot_changes_but_replay_is_safe(self):
        self.assertEqual(self.client().post(self.url, json=self.body).status_code, 201)
        with self.db(True) as (_, c):
            c.execute("UPDATE mp_jobs SET result=jsonb_set(result,'{visualFundamentals,price}','101'::jsonb)")
        r = self.client().get(self.url)
        self.assertTrue(r.json['editions'][0]['stale'])
        self.assertTrue(self.client().post(self.url, json=self.body).json['reused'])
        self.assertEqual(self.client().post(self.url+'/accept', json={}).status_code, 404)

    def test_forged_status_unknown_selection_and_stopped_assignment(self):
        for patch in [{'status': 'accepted'}, {'correctionIds': [EID]}, {'correctionIds': []}]:
            r = self.client().post(self.url, json={**self.body, **patch})
            self.assertEqual(r.status_code, 400, r.json)
        with self.db(True) as (_, c): c.execute("UPDATE mp_jobs SET status='cancelled'")
        self.assertEqual(self.client().post(self.url, json=self.body).status_code, 409)

    def test_preview_both_layouts_private_readonly_and_scoped(self):
        path=self.url+'/'+EID+'/preview'
        self.assertEqual(self.client().get(path).status_code,404)
        with self.db() as (_,c):
            c.execute("SELECT to_regclass('onepager_editions') AS name");self.assertIsNone(c.fetchone()['name'])
        self.assertEqual(self.client().post(self.url,json=self.body).status_code,201)
        for layout in ('brief','illustrated'):
            r=self.client().get(path+'?layout='+layout)
            self.assertEqual(r.status_code,200,r.json)
            self.assertEqual(r.headers['Cache-Control'],'private, no-store')
            self.assertEqual(r.json['id'],EID);self.assertEqual(r.json['status'],'pending')
            self.assertIn('Proposed claim',r.json['html'])
            self.assertIn('Pending editorial proposal — not accepted',r.json['html'])
            self.assertNotIn('Synthetic source',r.json['html'])
        self.assertEqual(self.client().get(path+'?accepted=true').status_code,400)
        self.assertEqual(self.client().get(path+'?layout=accepted').status_code,400)
        self.assertEqual(self.client().get(path.replace(AID,EID)).status_code,404)
        with self.db() as (_,c):
            c.execute('SELECT state FROM stock_analysis_runs');self.assertEqual(c.fetchone()['state'],self.run['state'])
            c.execute('SELECT result FROM mp_jobs');self.assertEqual(c.fetchone()['result'],self.assignment['result'])
            c.execute('SELECT count(*) AS n FROM onepager_editions');self.assertEqual(c.fetchone()['n'],1)

    def test_preview_rechecks_snapshot_policy_bytes_and_record(self):
        for mutation in ("UPDATE mp_jobs SET result=jsonb_set(result,'{visualFundamentals,price}','101'::jsonb)",
                         "UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb",
                         "UPDATE document_files SET file_data='Y2hhbmdlZA=='",
                         "UPDATE onepager_editions SET record=jsonb_set(record,'{status}','\"accepted\"'::jsonb)",
                         "UPDATE onepager_corrections SET payload_hash='changed'"):
            self.setUp()
            self.assertEqual(self.client().post(self.url,json=self.body).status_code,201)
            with self.db(True) as (_,c):c.execute(mutation)
            r=self.client().get(self.url+'/'+EID+'/preview')
            self.assertEqual(r.status_code,409,r.json)
            self.assertNotIn('html',r.json)
            self.assertEqual(r.headers['Cache-Control'],'private, no-store')

    def test_preview_locks_evidence_through_render(self):
        from threading import Event
        from unittest.mock import patch
        import psycopg2
        from research_infographic import render
        entered,release=Event(),Event()
        self.assertEqual(self.client().post(self.url,json=self.body).status_code,201)
        def held(*args,**kwargs):
            entered.set()
            if not release.wait(5):raise AssertionError('Synthetic lock test timed out')
            return render(*args,**kwargs)
        with patch('research_infographic.render',side_effect=held),ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(lambda:self.client().get(self.url+'/'+EID+'/preview'))
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(psycopg2.errors.LockNotAvailable):
                    with self.db(True) as (_,c):
                        c.execute("SET LOCAL lock_timeout='100ms'")
                        c.execute("UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb")
            finally:release.set()
            self.assertEqual(future.result().status_code,200)

    def test_preview_rechecks_unchanged_source_as_well_as_corrected_source(self):
        import copy
        from onepager_corrections import preparation,prepare
        run=copy.deepcopy(self.run)
        other=copy.deepcopy(run['sources'][0]);other.update(id='s2',filename='unchanged.txt')
        run['sources'].append(other)
        run['input'].setdefault('hashes',{})['unchanged.txt']=other['originalHash']
        fingerprint=preparation(run)['fingerprint']
        with self.db(True) as (_,c):
            c.execute('SELECT * FROM onepager_corrections');row=c.fetchone();p=row['proposal']
            c.execute('SELECT filename,file_data,metadata FROM document_files');doc=dict(c.fetchone())
            body={k:p[k] for k in ('path','statement','basis','reason','sourceId','excerpt','locator')}
            body.update(requestId=str(row['id']),fingerprint=fingerprint)
            _,signature,proposal=prepare(run,body,doc)
            c.execute('UPDATE stock_analysis_runs SET input=%s::jsonb,sources=%s::jsonb',(json.dumps(run['input']),json.dumps(run['sources'])))
            c.execute('UPDATE onepager_corrections SET fingerprint=%s,payload_hash=%s,proposal=%s::jsonb',(fingerprint,signature,json.dumps(proposal)))
            c.execute("INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES('TEST','unchanged.txt',%s,'{}'::jsonb)",(doc['file_data'],))
        self.body.update({k:v for k,v in baseline(run,self.assignment,self.assignment['result']['visualFundamentals']).items() if k!='ticker'})
        self.assertEqual(self.client().post(self.url,json=self.body).status_code,201)
        path=self.url+'/'+EID+'/preview'
        self.assertEqual(self.client().get(path).status_code,200)
        with self.db(True) as (_,c):c.execute("UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb WHERE filename='unchanged.txt'")
        r=self.client().get(path)
        self.assertEqual(r.status_code,409,r.json)
        self.assertIn('restricted',r.json['error'])
