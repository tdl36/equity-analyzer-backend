"""Synthetic collection and isolated PostgreSQL only; no paid APIs or user database."""
import base64
import copy
import hashlib
import json
import unittest
import uuid
from unittest.mock import patch
from flask import Flask
from research_assignments import Coordinator,plan,child
from research_assignment_outputs import note,thesis
from tests.unit import test_stock_analysis_postgres as stock_tests
PG=stock_tests.PG
from tests.unit import test_collection_refresh as refresh_tests

P={'ticker':'SYK','since':'2026-08-01','until':'2026-10-09','outputs':['summary_note','stock_summary','thesis','visual']}

class PlanningTests(unittest.TestCase):
    def test_frozen_window_and_invalid_input(self):
        self.assertEqual(plan(P)['since'],'2026-08-01')
        for change in [{'ticker':None},{'ticker':'../bad'},{'outputs':[{}]},{'outputs':[]},{'outputs':['thesis','thesis']},{'until':'2099-01-01'},{'since':'2024-01-01'},{'instruction':'x'*1801}]:
            with self.subTest(change=change),self.assertRaises(ValueError):plan({**P,**change})

class CollectionAssignmentTests(unittest.TestCase):
    setUp=refresh_tests.RefreshTests.setUp
    def command(self):
        return {'action':'research_assignment','payload':{**P,'ticker':'MDT','assignmentId':str(uuid.uuid4()),'sourcePolicy':{'mode':'auto','rules':[]}}}
    def test_targeted_claim_preserves_other_requests_and_global_lease(self):
        a=self.m.apply_cloud_command(str(uuid.uuid4()),self.command())['refreshRequestId']
        b=self.m.apply_cloud_command(str(uuid.uuid4()),self.command())['refreshRequestId']
        claimed=self.m.claim(b)
        self.assertEqual(claimed['id'],b)
        self.assertIsNone(self.m.claim(a))
        self.m.mark(b,claimed['owner'],'queued','')
        self.assertEqual(self.m.claim(a)['id'],a)
    def test_exact_window_replay_cancellation_and_policy_preservation(self):
        self.m.save({**self.cfg,'enabled':False});cmd=self.command();ident=str(uuid.uuid4())
        first=self.m.apply_cloud_command(ident,cmd)
        self.assertEqual(self.m.apply_cloud_command(ident,cmd),first)
        claimed=self.m.claim();self.assertEqual(claimed['id'],first['refreshRequestId'])
        run=self.c.status(claimed['run']);self.assertEqual((run['since'],run['until_date']),('2026-08-01','2026-10-09'))
        self.assertEqual(len(run['tasks']),4);self.assertEqual(claimed['config']['sourceReviewId'],ident)
        self.assertFalse(self.m.status()['policies'][0]['enabled'])
        cancel={'action':'cancel_assignment','payload':{'assignmentId':cmd['payload']['assignmentId'],'ticker':'MDT'}}
        self.m.apply_cloud_command(str(uuid.uuid4()),cancel)
        self.assertEqual(self.m.status()['requests'][0]['status'],'cancelled')
        # Out-of-order delayed create cannot resurrect a stopped assignment.
        self.assertTrue(self.m.apply_cloud_command(str(uuid.uuid4()),cmd)['cancelled'])
    def test_completion_requires_import_and_retains_exact_receipts(self):
        from tests.unit.test_charlie_collector import pdf
        from pathlib import Path
        cmd=self.command();self.m.apply_cloud_command(str(uuid.uuid4()),cmd);c=self.m.claim()
        f=Path(self.tmp.name)/'one.pdf';f.write_bytes(pdf())
        d=self.c.stage(c['run'],'MDT','transcript',f,'https://research.alpha-sense.com/doc/one')['documents'][0];self.c.handoff(d['id'])
        for kind in ('transcript','broker-report','press-release','presentation'):
            self.c.observe(c['run'],'MDT',kind,'https://research.alpha-sense.com/search',1 if kind=='transcript' else 0,'Synthetic reviewed filters')
            self.c.finish(c['run'],'MDT',kind,1 if kind=='transcript' else 0)
        with patch.object(self.c,'verify',return_value={'verifications':[]}),patch('collection_original_sync.sync'):
            with self.assertRaisesRegex(ValueError,'cloud import'):self.m.complete(c['id'],c['owner'])
            receipt={'imported':True,'filename':'one.pdf','sha256':d['sha256']}
            self.c.db.execute("UPDATE original_imports SET status='imported',receipt=?",(json.dumps(receipt),));self.c.db.commit()
            done=self.m.complete(c['id'],c['owner']);self.assertEqual(done['assignmentSources'][0]['sha256'],d['sha256'])

@unittest.skipUnless((PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class AssignmentPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stock_tests.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,cur):
            cur.execute('''CREATE TABLE mp_jobs(id TEXT PRIMARY KEY,stage TEXT,ticker TEXT,status TEXT,input JSONB,result JSONB,error TEXT,created_at TIMESTAMP DEFAULT NOW(),updated_at TIMESTAMP DEFAULT NOW());
            CREATE TABLE app_settings(key TEXT PRIMARY KEY,value TEXT,updated_at TIMESTAMP DEFAULT NOW());
            CREATE TABLE portfolio_analyses(ticker TEXT UNIQUE,company TEXT,analysis JSONB,updated_at TIMESTAMP DEFAULT NOW());''')
    @classmethod
    def tearDownClass(cls):stock_tests.StockPostgresTests.tearDownClass.__func__(cls)
    def setUp(self):
        self.ticker='T'+uuid.uuid4().hex[:10].upper();self.ident=str(uuid.uuid4());self.submits={};self.report_status='complete';self.stale=False;self.resume_error=None
        self.app=Flask(__name__);self.app.config['TESTING']=True
        import thesis_imports
        self.app.register_blueprint(thesis_imports.create_blueprint(self.db))
        self.co=Coordinator(self.app,self.db,self.invoke,lambda:True);self.co.wake=lambda:None
        self.app.register_blueprint(self.co.bp);self.client=self.app.test_client()
        self.raw=b'Synthetic company evidence.';self.sha=hashlib.sha256(self.raw).hexdigest()
        with self.db(True) as (_,cur):cur.execute('INSERT INTO document_files VALUES(%s,%s,%s,%s,%s::jsonb)',(self.ticker,'one.txt',base64.b64encode(self.raw).decode(),'txt','{}'))
    def invoke(self,endpoint,method,body,params):
        if endpoint.startswith('thesis_imports.'):
            with self.app.test_request_context('/internal',method=method,json=body):
                try:r=self.app.make_response(self.app.view_functions[endpoint](**params));return r.get_json(),r.status_code
                except ValueError as e:return {'error':str(e)},409
        if endpoint=='stock_analysis.listing':return {'baseline':{'revision':0}},200
        if endpoint=='stock_analysis.submit':self.submits.setdefault(body['requestId'],copy.deepcopy(body));return {'id':body['requestId']},200
        if endpoint=='stock_analysis.control':return ({'error':self.resume_error},409) if self.resume_error else ({'ok':True},200)
        if endpoint=='stock_analysis.detail':return self.report(),200
        raise AssertionError(endpoint)
    def report(self):
        c={'statement':'Synthetic company evidence.','basis':'reported_fact','passageMatched':True,'review':'supported','evidence':[{'sourceId':'S1','matched':True,'excerpt':self.raw.decode()}]}
        return dict(id=child(self.ident,'stock-analysis'),ticker=self.ticker,status=self.report_status,baselineStale=self.stale,input={},sources=[{'id':'S1','filename':'one.txt','originalHash':self.sha,'sourceUrl':'https://research.alpha-sense.com/doc/one'}],state={'completed':['a']*12,'report':{},'citations':{p:copy.deepcopy(c) for p in ['/summary/investment_thesis/0','/summary/one_liner','/risks/0/risk','/monitor/0/metric']}})
    def start(self,outputs=None):
        p={**P,'ticker':self.ticker,'requestId':self.ident}
        if outputs:p['outputs']=outputs
        r=self.client.post('/api/research/assignments',json=p);self.assertEqual(r.status_code,202,r.json);return p
    def collected(self,sha=None):
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET status='applied',result=%s::jsonb WHERE id=%s",(json.dumps({'refreshRequestId':self.ident}),child(self.ident,'collection')))
            v={'requests':[{'id':self.ident,'status':'complete','result':{'assignmentSources':[{'filename':'one.txt','sha256':sha or self.sha}]}}]}
            cur.execute("INSERT INTO app_settings(key,value) VALUES('collection_control_snapshot',%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",(json.dumps(v),))
    def advance(self):
        with self.app.app_context():self.co.advance(self.ident)
        return self.co.read(self.ident)
    def test_end_to_end_replay_restart_drafts_no_approval(self):
        p=self.start();self.assertEqual(self.client.post('/api/research/assignments',json=p).status_code,200)
        self.assertEqual(self.advance()['result']['step'],'Waiting for Mac to receive collection')
        self.collected();r=self.advance();self.assertEqual(r['status'],'complete',r['error'])
        self.assertEqual(len(r['result']['artifacts']),3);self.assertTrue(r['result']['thesisDraftId']);self.assertEqual(len(self.submits),1)
        self.advance();self.assertEqual(len(self.submits),1)
        draft=self.client.get('/api/thesis-imports/'+r['result']['thesisDraftId']).json;self.assertEqual(draft['status'],'pending')
        with self.db() as (_,cur):cur.execute('SELECT * FROM portfolio_analyses WHERE ticker=%s',(self.ticker,));self.assertIsNone(cur.fetchone())
        out=self.client.get('/api/research/assignments/'+self.ident+'/artifact/visual');self.assertEqual(out.status_code,200);self.assertIn('sandbox',out.headers['Content-Security-Policy'])
        public=self.client.get('/api/research/assignments?ticker='+self.ticker).json['assignments'][0]
        self.assertNotIn('thesisPackage',public['result']);self.assertNotIn('html',public['result']['artifacts'][0])
    def test_hash_mismatch_no_dispatch(self):
        self.start();self.collected('bad');r=self.advance();self.assertEqual(r['status'],'attention');self.assertEqual(self.submits,{})
    def test_delivery_overdue_visible_and_recovers_without_duplicate(self):
        self.start()
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET created_at=NOW()-INTERVAL '20 hours' WHERE id=%s",(child(self.ident,'collection'),))
        r=self.advance();self.assertEqual(r['status'],'running');self.assertIn('overdue',r['result']['delayWarning'])
        self.assertEqual(self.submits,{})
        self.collected();r=self.advance();self.assertEqual(r['status'],'complete');self.assertNotIn('delayWarning',r['result'])
        self.assertEqual(len(self.submits),1)
    def test_stale_collection_snapshot_visible_without_paid_dispatch(self):
        self.start();self.collected()
        with self.db(True) as (_,cur):
            v={'requests':[{'id':self.ident,'status':'collecting'}]}
            cur.execute("UPDATE app_settings SET value=%s,updated_at=NOW()-INTERVAL '20 minutes' WHERE key='collection_control_snapshot'",(json.dumps(v),))
        r=self.advance();self.assertEqual(r['status'],'running');self.assertIn('stale',r['result']['delayWarning']);self.assertEqual(self.submits,{})
    def test_stale_thesis_retains_other_outputs(self):
        self.start();self.collected();self.stale=True;r=self.advance();self.assertEqual(r['status'],'attention');self.assertEqual(len(r['result']['artifacts']),3);self.assertNotIn('thesisDraftId',r['result'])
    def test_unknown_call_requires_explicit_resume(self):
        self.start(['stock_summary']);self.collected();self.report_status='running';self.resume_error='Previous call outcome is unknown.'
        r=self.advance();self.assertEqual(r['status'],'attention');self.assertIn('unknown',r['error'])
        r=self.client.post('/api/research/assignments/'+self.ident+'/resume',json={});self.assertEqual(r.status_code,409)
        self.resume_error=None;self.report_status='complete';self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/resume',json={'acknowledgeRetry':True}).status_code,200)
        self.assertEqual(self.advance()['status'],'complete');self.assertEqual(len(self.submits),1)
    def test_stop_before_mac_receipt_creates_durable_cancel(self):
        self.start();self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/stop',json={}).status_code,200)
        self.assertEqual(self.advance()['status'],'cancelled');self.assertEqual(self.submits,{})
        with self.db() as (_,cur):cur.execute('SELECT input FROM mp_jobs WHERE id=%s',(child(self.ident,'cancel-collection'),));self.assertEqual(cur.fetchone()['input']['payload']['assignmentId'],self.ident)
    def test_changed_detailed_thesis_blocks_draft_without_losing_outputs(self):
        self.start();self.collected()
        with self.db(True) as (_,cur):cur.execute("INSERT INTO portfolio_analyses(ticker,company,analysis) VALUES(%s,'Synthetic','{}')",(self.ticker,))
        r=self.advance();self.assertEqual(r['status'],'attention');self.assertEqual(len(r['result']['artifacts']),3)

    def test_upgrade_preserves_old_ids_and_does_not_apply(self):
        from tests.unit.test_thesis_imports import sample
        old=sample(self.ticker)
        with self.db(True) as (_,cur):cur.execute('INSERT INTO portfolio_analyses(ticker,company,analysis) VALUES(%s,%s,%s::jsonb)',(self.ticker,old['companyName'],json.dumps(old['analysis'])))
        self.start();self.collected();r=self.advance();self.assertEqual(r['status'],'complete',r['error'])
        draft=self.client.get('/api/thesis-imports/'+r['result']['thesisDraftId']).json['package']
        self.assertEqual(draft['baseline']['mode'],'upgrade')
        self.assertIn('Frozen existing detailed-thesis context',next(iter(self.submits.values()))['question'])
        self.assertEqual(draft['analysis']['thesis']['pillars'][0],old['analysis']['thesis']['pillars'][0])
        self.assertGreater(len(draft['analysis']['thesis']['pillars']),len(old['analysis']['thesis']['pillars']))
    def test_restart_uses_frozen_collection_after_snapshot_rotation(self):
        self.start(['visual']);self.collected();self.report_status='running'
        self.resume_error='A worker is still active.';self.assertEqual(self.advance()['status'],'running')
        with self.db(True) as (_,cur):cur.execute("UPDATE app_settings SET value='{}' WHERE key='collection_control_snapshot'")
        self.report_status='complete';self.assertEqual(self.advance()['status'],'complete')
    def test_real_stock_worker_to_real_thesis_inbox_with_synthetic_model(self):
        import company_research
        from types import SimpleNamespace
        from tests.unit.test_stock_analysis import SOURCE,result,GROUPS
        calls=[]
        def ask(prompt,key,tokens,ident,stage):
            calls.append(stage)
            if stage.startswith('report'):
                raw=result(GROUPS[int(stage[-1])])
                if 'summary' in raw['report']:
                    raw['report']['summary']['investment_thesis']=[SOURCE['text']]
                    raw['citations'].append({'path':'/summary/investment_thesis/0','basis':'reported_fact','evidence':[{'sourceId':'S1','excerpt':SOURCE['text']}]})
                with self.db() as (_,cur):cur.execute('SELECT sources FROM stock_analysis_runs WHERE id=%s',(ident,));sid=cur.fetchone()['sources'][0]['id']
                for c in raw['citations']:
                    for e in c['evidence']:e['sourceId']=sid
                return raw
            return {'findings':[{'claimId':path,'status':'supported','reason':'Synthetic source review.'} for path in ('/summary/one_liner','/summary/investment_thesis/0')]}
        self.app.register_blueprint(company_research.create_blueprint(self.db,ask,lambda k:'synthetic',lambda:'synthetic-model',studio=True))
        def invoke(endpoint,method,body,params):
            with self.app.test_request_context('/internal',method=method,json=body):
                r=self.app.make_response(self.app.view_functions[endpoint](**params));return r.get_json(),r.status_code
        self.co.invoke=invoke
        with patch('company_research.threading.Thread') as thread,patch.dict('sys.modules',{'notegen':SimpleNamespace(extract_file_text=lambda d,max_chars:SOURCE['text'])}):
            self.start();self.collected();self.assertEqual(self.advance()['status'],'running')
            thread.call_args.kwargs['target'](*thread.call_args.kwargs['args'])
            r=self.advance();self.assertEqual(r['status'],'complete',r['error'])
            self.assertEqual(len(r['result']['artifacts']),3);self.assertTrue(r['result']['thesisDraftId'])
            n=len(calls);self.advance();self.assertEqual(len(calls),n);self.assertGreater(n,0)
    def test_quality_resume_archives_derived_outputs_and_repairs_only_on_request(self):
        original=self.report
        def unsupported():
            r=original();r['state']['citations']['/summary/investment_thesis/0']['review']='needs_review';return r
        self.report=unsupported;self.start();self.collected();r=self.advance()
        self.assertEqual(r['status'],'attention');self.assertTrue(r['result']['artifacts']);self.assertNotIn('thesisDraftId',r['result'])
        before=copy.deepcopy(r['result']['artifacts']);invoke=self.co.invoke;calls=[]
        def recording(endpoint,method,body,params):
            if endpoint=='stock_analysis.control':calls.append(body)
            return invoke(endpoint,method,body,params)
        self.co.invoke=recording
        self.advance();self.assertEqual(calls,[])
        response=self.client.post('/api/research/assignments/'+self.ident+'/resume',json={})
        self.assertEqual(response.status_code,200,response.json);self.assertTrue(calls[0]['repairThesis'])
        state=self.co.read(self.ident)['result'];self.assertNotIn('artifacts',state);self.assertEqual(state['artifactRevisions'][0],before)
        self.assertNotIn('artifactRevisions',self.co.public(self.co.read(self.ident))['result'])
        self.report=original;r=self.advance();self.assertEqual(r['status'],'complete');self.assertTrue(r['result']['thesisDraftId'])
    def test_legacy_risk_schema_gets_explicit_gap_only_in_pending_draft(self):
        from tests.unit.test_thesis_imports import sample
        old=sample(self.ticker);old['analysis']['threats'][0].pop('triggerPoints',None)
        with self.db(True) as (_,cur):cur.execute('INSERT INTO portfolio_analyses(ticker,company,analysis) VALUES(%s,%s,%s::jsonb)',(self.ticker,old['companyName'],json.dumps(old['analysis'])))
        self.start();self.collected();r=self.advance();self.assertEqual(r['status'],'complete',r['error'])
        draft=self.client.get('/api/thesis-imports/'+r['result']['thesisDraftId']).json
        self.assertEqual(draft['status'],'pending');risk=draft['package']['analysis']['threats'][0]
        self.assertEqual(risk['id'],old['analysis']['threats'][0]['id']);self.assertIn('Not recorded',risk['triggerPoints'])
        with self.db() as (_,cur):
            cur.execute('SELECT analysis FROM portfolio_analyses WHERE ticker=%s',(self.ticker,));self.assertEqual(cur.fetchone()['analysis'],old['analysis'])
