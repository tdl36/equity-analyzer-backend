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
    def test_configurable_collection_scope_is_frozen(self):
        self.assertEqual(plan({**P,'collectionLimit':40,'collectionOnly':True,'outputs':[]})['collectionLimit'],40)
        self.assertNotIn('collectionLimit',plan(P))
        for limit in (0,1001,True,'40'):
            with self.assertRaises(ValueError):plan({**P,'collectionLimit':limit})

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
        return dict(id=child(self.ident,'stock-analysis'),ticker=self.ticker,status=self.report_status,baselineStale=self.stale,input={},sources=[{'id':'S1','filename':'one.txt','originalHash':self.sha,'sourceUrl':'https://research.alpha-sense.com/doc/one'}],state={'completed':['a']*12,'report':{'summary':{'investment_thesis':[c['statement']],'one_liner':c['statement']},'risks':[{'risk':c['statement']}],'monitor':[{'metric':c['statement']}]},'citations':{p:copy.deepcopy(c) for p in ['/summary/investment_thesis/0','/summary/one_liner','/risks/0/risk','/monitor/0/metric']}})
    def start(self,outputs=None):
        p={**P,'ticker':self.ticker,'requestId':self.ident}
        if outputs:p['outputs']=outputs
        r=self.client.post('/api/research/assignments',json=p);self.assertEqual(r.status_code,202,r.json);return p
    def test_origin_rejection_and_exact_retry_after_baseline_changes(self):
        origin={'kind':'onepager','assignmentId':str(uuid.uuid4()),'runId':str(uuid.uuid4()),'ticker':self.ticker,'fingerprint':'a'*64,'fundamentalsDigest':'b'*64,'parentId':None}
        body={**P,'ticker':self.ticker,'requestId':self.ident,'origin':origin}
        with patch('research_assignment_origin.verify',side_effect=ValueError('Baseline changed')):
            self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,409)
        self.assertIsNone(self.co.read(self.ident))
        with self.db() as (_,cur):
            cur.execute('SELECT id FROM mp_jobs WHERE id=%s',(child(self.ident,'collection'),));self.assertIsNone(cur.fetchone())
        with patch('research_assignment_origin.verify') as checked:
            self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
            checked.assert_called_once()
        self.assertEqual(self.co.read(self.ident)['input']['origin'],origin)
        with patch('research_assignment_origin.verify',side_effect=AssertionError('Replay must retain frozen origin')):
            self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,200)
        changed={**body,'origin':{**origin,'parentId':str(uuid.uuid4())}}
        self.assertEqual(self.client.post('/api/research/assignments',json=changed).status_code,409)
    def collected(self,sha=None):
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET status='applied',result=%s::jsonb WHERE id=%s",(json.dumps({'refreshRequestId':self.ident}),child(self.ident,'collection')))
            v={'requests':[{'id':self.ident,'status':'complete','result':{'assignmentSources':[{'filename':'one.txt','sha256':sha or self.sha}]}}]}
            cur.execute("INSERT INTO app_settings(key,value) VALUES('collection_control_snapshot',%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",(json.dumps(v),))
    def advance(self):
        with self.app.app_context():self.co.advance(self.ident)
        return self.co.read(self.ident)
    def test_separate_proposal_link_checks_provenance_and_preserves_failed_run(self):
        self.start(['thesis']);self.collected();row=self.advance()
        state=row['result'];draft_id=state.pop('thesisDraftId')
        self.co.store(self.ident,state,'attention','No supported thesis statements were produced.')
        path='/api/research/assignments/'+self.ident+'/link-thesis'
        before=self.co.read(self.ident);calls=len(self.submits)
        original=self.co.invoke
        for change in ({'ticker':'WRONG'},{'stale':True},{'status':'dismissed'},
                       {'package':{'baseline':{'hash':'wrong'}}},
                       {'package':{'baseline':before['input']['thesisBaseline']['baseline'],'sourceRegister':[]}}):
            def changed(endpoint,method,body,params):
                data,code=original(endpoint,method,body,params)
                if endpoint=='thesis_imports.detail':data={**data,**change}
                return data,code
            with patch.object(self.co,'invoke',side_effect=changed):
                self.assertEqual(self.client.post(path,json={'draftId':draft_id}).status_code,409)
            self.assertEqual(self.co.read(self.ident),before)
        response=self.client.post(path,json={'draftId':draft_id})
        self.assertEqual(response.status_code,200,response.json)
        saved=self.co.read(self.ident)
        self.assertEqual(saved['status'],'attention');self.assertEqual(saved['error'],before['error'])
        self.assertEqual(saved['result']['separateThesisProposal']['sharedOriginals'],1)
        self.assertEqual(saved['result']['sources'],before['result']['sources'])
        self.assertEqual(self.client.post(path,json={'draftId':draft_id}).status_code,200)
        self.assertEqual(self.co.read(self.ident),saved)
        self.assertEqual(self.client.post(path,json={'draftId':str(uuid.uuid4())}).status_code,409)
        self.assertEqual(len(self.submits),calls)
        self.assertEqual(self.client.get('/api/thesis-imports/'+draft_id).json['status'],'pending')

    def test_financial_workbook_uses_saved_values_without_dispatch_or_mutation(self):
        from io import BytesIO
        from openpyxl import load_workbook
        from tests.unit.test_financial_workbook import WorkbookTests
        self.start(['visual']);self.collected();self.advance()
        path='/api/research/assignments/'+self.ident+'/financial-history'
        self.assertEqual(self.client.get(path).status_code,409)
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET result=jsonb_set(result,'{visualFundamentals}',%s::jsonb) WHERE id=%s",(json.dumps(WorkbookTests().fixture()),self.ident))
        before=self.co.read(self.ident);count=len(self.submits)
        response=self.client.get(path)
        self.assertEqual(response.status_code,200);self.assertEqual(response.headers['Cache-Control'],'no-store')
        book=load_workbook(BytesIO(response.data));self.assertEqual(book['Financial history']['C6'].value,1000)
        self.assertEqual(self.co.read(self.ident),before);self.assertEqual(len(self.submits),count)
        self.assertEqual(self.client.get(path.replace(self.ident,str(uuid.uuid4()))).status_code,404)

    def test_focused_read_keeps_identity_company_and_does_not_dispatch(self):
        from unittest.mock import Mock
        self.start();self.co.wake=Mock()
        path='/api/research/assignments?ticker='+self.ticker+'&requestId='+self.ident
        response=self.client.get(path)
        self.assertEqual(response.status_code,200)
        self.assertEqual([x['id'] for x in response.json['assignments']],[self.ident])
        self.assertEqual(response.headers['Cache-Control'],'no-store')
        self.assertEqual(self.client.get(path.replace(self.ticker,'OTHER')).json['assignments'],[])
        self.assertEqual(self.client.get(path.replace(self.ident,str(uuid.uuid4()))).json['assignments'],[])
        self.assertEqual(self.client.get(path.replace(self.ident,'not-an-id')).status_code,400)
        self.co.wake.assert_not_called()

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
    def start_visual_data_assignment(self):
        self.co.lab_note_defaults=lambda:{'modelKey':'fixture','promptTemplate':'Frozen fixture'}
        self.note_calls=[]
        def dispatch(ident,p,pack):
            self.note_calls.append((ident,copy.deepcopy(pack)))
            return {'noteId':child(ident,'full-lab-note'),'status':'draft'}
        self.co.lab_note_dispatch=dispatch
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['visual','summary_note','thesis'],
              'visualFundamentals':True,'labFullNote':{'allowPdfText':False}}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected()

    def test_visual_data_failure_retains_other_requested_drafts_without_fallback_visual(self):
        self.start_visual_data_assignment()
        with patch('stock_visual_fundamentals.collect',side_effect=RuntimeError('Private provider payload')):
            row=self.advance()
        self.assertEqual(row['status'],'attention')
        self.assertNotIn('visual',row['result']['artifacts'])
        self.assertIn('summary_note',row['result']['artifacts'])
        self.assertTrue(row['result']['thesisDraftId']);self.assertEqual(row['result']['labNoteStatus'],'draft')
        self.assertEqual(set(row['result']['outputErrors']),{'visual'})
        self.assertNotIn('Private',json.dumps(row['result'])+row['error'])
        self.assertEqual(self.client.get('/api/research/assignments/'+self.ident+'/artifact/visual').status_code,404)

    def test_visual_data_retry_preserves_saved_outputs_and_reuses_research(self):
        self.start_visual_data_assignment()
        with patch('stock_visual_fundamentals.collect',side_effect=TimeoutError):row=self.advance()
        before=copy.deepcopy(row['result']);original_note=note;original_invoke=self.co.invoke
        def render(run,kind,p):
            self.assertEqual(kind,'visual');return original_note(run,kind,p)
        def invoke(endpoint,method,body,params):
            if endpoint=='thesis_imports.drafts':raise AssertionError('Saved draft must not be recreated')
            return original_invoke(endpoint,method,body,params)
        self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/resume',json={}).status_code,200)
        f={'version':2,'asOf':'2026-10-10','annuals':[]}
        with patch('stock_visual_fundamentals.collect',return_value=f) as collect,patch('research_assignment_outputs.note',side_effect=render),patch.object(self.co,'invoke',side_effect=invoke):
            row=self.advance()
        self.assertEqual(row['status'],'complete',row['error']);self.assertEqual(collect.call_count,1)
        self.assertEqual(row['result']['artifacts']['summary_note'],before['artifacts']['summary_note'])
        self.assertEqual(row['result']['thesisDraftId'],before['thesisDraftId'])
        self.assertEqual(row['result']['outputErrors'],{});self.assertEqual(len(self.submits),1)
        self.assertEqual(self.note_calls[0],self.note_calls[1])

    def test_renderer_failure_does_not_hide_later_outputs_and_retry_only_renders_missing(self):
        self.start(['summary_note','stock_summary','thesis']);self.collected()
        original=note
        def render(run,kind,p):
            if kind=='summary_note':raise RuntimeError('Private licensed source body')
            return original(run,kind,p)
        with patch('research_assignment_outputs.note',side_effect=render):row=self.advance()
        self.assertEqual(row['status'],'attention');self.assertEqual(set(row['result']['outputErrors']),{'summary_note'})
        self.assertIn('stock_summary',row['result']['artifacts']);self.assertTrue(row['result']['thesisDraftId'])
        self.assertNotIn('Private',json.dumps(row['result'])+row['error'])
        saved=copy.deepcopy(row['result']['artifacts']['stock_summary'])
        self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/resume',json={}).status_code,200)
        with patch('research_assignment_outputs.note',wraps=original) as rendering:row=self.advance()
        self.assertEqual([c.args[1] for c in rendering.call_args_list],['summary_note'])
        self.assertEqual(row['status'],'complete');self.assertEqual(row['result']['artifacts']['stock_summary'],saved)
        self.assertEqual(len(self.submits),1)

    def test_combined_visual_and_thesis_errors_keep_thesis_repair_first(self):
        self.start_visual_data_assignment()
        with patch('stock_visual_fundamentals.collect',side_effect=TimeoutError),patch('research_assignment_outputs.thesis',side_effect=ValueError('No supported thesis statements: fixture')):
            row=self.advance()
        self.assertEqual(set(row['result']['outputErrors']),{'visual','thesis'})
        self.assertTrue(row['error'].startswith('No supported thesis statements'))
        self.assertEqual(row['result']['labNoteStatus'],'draft')

    def test_saved_visual_is_not_rebuilt_or_refetched_during_other_output_recovery(self):
        self.start_visual_data_assignment()
        with patch('stock_visual_fundamentals.collect',return_value={'version':2,'annuals':[]}):row=self.advance()
        self.assertEqual(row['status'],'complete');saved=copy.deepcopy(row['result']['artifacts'])
        state=row['result'];state.pop('visualFundamentals',None)
        self.co.store(self.ident,state,'running')
        with patch('stock_visual_fundamentals.collect',side_effect=AssertionError('Saved visual does not require new retrieval')),patch('research_assignment_outputs.note',side_effect=AssertionError('Saved drafts must not be rendered again')):
            row=self.advance()
        self.assertEqual(row['status'],'complete');self.assertEqual(row['result']['artifacts'],saved)

    def test_shared_report_attention_still_blocks_all_output_work(self):
        self.start_visual_data_assignment();self.report_status='attention'
        with patch('stock_visual_fundamentals.collect') as collect,patch('research_assignment_outputs.note') as render:
            row=self.advance()
        self.assertEqual(row['status'],'attention');collect.assert_not_called();render.assert_not_called()
        self.assertFalse(self.note_calls);self.assertFalse(row['result'].get('artifacts'))

    def test_visual_refresh_is_read_only_and_original_remains_available(self):
        self.start(['visual']);self.collected();self.advance()
        before=self.co.read(self.ident)
        path='/api/research/assignments/'+self.ident+'/artifact/visual'
        original=self.client.get(path+'?render=original')
        self.assertEqual(original.status_code,200)
        refreshed=self.client.get(path)
        self.assertIn(b'research-onepager-v5',refreshed.data)
        self.assertEqual(before,self.co.read(self.ident))
        self.assertEqual(len(self.submits),1)
        self.report_status='running'
        self.assertEqual(self.client.get(path).status_code,409)
        self.assertEqual(self.client.get(path+'?render=original').data,original.data)

    def test_original_visual_survives_report_and_renderer_outages_without_mutation(self):
        self.start(['visual']);self.collected();self.advance()
        before=self.co.read(self.ident)
        path='/api/research/assignments/'+self.ident+'/artifact/visual'
        original=before['result']['artifacts']['visual']['html'].encode()
        for failure in ('report_status','report_exception','renderer_exception'):
            with self.subTest(failure=failure):
                mock=patch.object(self.co,'invoke',return_value=({'error':'private diagnostic'},503)) if failure=='report_status' else patch.object(self.co,'invoke',side_effect=RuntimeError('private diagnostic')) if failure=='report_exception' else patch('research_onepager.render',side_effect=RuntimeError('private diagnostic'))
                with mock as dependency:
                    response=self.client.get(path)
                    self.assertEqual(response.status_code,409 if failure=='report_status' else 503)
                    self.assertIn('original saved output',response.json['error'])
                    self.assertNotIn('private diagnostic',response.get_data(as_text=True))
                    dependency.reset_mock()
                    saved=self.client.get(path+'?render=original')
                    self.assertEqual(saved.status_code,200);self.assertEqual(saved.data,original)
                    self.assertEqual(saved.headers['Cache-Control'],'no-store')
                    dependency.assert_not_called()
                self.assertEqual(self.co.read(self.ident),before)
                self.assertEqual(len(self.submits),1)

    def test_public_visual_supplement_preserves_research_and_replays_without_network(self):
        self.start(['visual']);self.collected();self.advance()
        before=self.co.read(self.ident)['result'];f={'version':2,'asOf':'2026-10-10','annuals':[]}
        path='/api/research/assignments/'+self.ident+'/visual-supplement'
        with patch('stock_visual_fundamentals.collect',return_value=f) as collect:
            self.assertEqual(self.client.post(path,json={}).status_code,200)
            self.assertEqual(self.client.post(path,json={}).status_code,200)
            self.assertEqual(collect.call_count,1)
        after=self.co.read(self.ident)['result']
        self.assertEqual(before['artifacts'],after['artifacts']);self.assertEqual(len(self.submits),1)
        self.assertEqual(after['visualFundamentals'],f)

    def test_visual_upgrade_preserves_previous_snapshot_on_failure_and_success(self):
        self.start(['visual']);self.collected();self.advance()
        row=self.co.read(self.ident);state=row['result']
        old={'version':1,'asOf':'2026-10-09','annuals':[]}
        state['visualFundamentals']=old;self.co.store(self.ident,state,row['status'],row.get('error'))
        path='/api/research/assignments/'+self.ident+'/visual-supplement'
        before=self.co.read(self.ident)
        with patch('stock_visual_fundamentals.collect',side_effect=ValueError('unavailable')):
            self.assertEqual(self.client.post(path,json={}).status_code,409)
        self.assertEqual(before,self.co.read(self.ident))
        new={'version':2,'asOf':'2026-10-10','annuals':[]}
        with patch('stock_visual_fundamentals.collect',return_value=new) as collect:
            self.assertEqual(self.client.post(path,json={}).status_code,200)
            self.assertEqual(self.client.post(path,json={}).status_code,200)
            self.assertEqual(collect.call_count,1)
        after=self.co.read(self.ident)['result']
        self.assertEqual(after['visualFundamentalRevisions'],[old])
        self.assertEqual(after['artifacts'],before['result']['artifacts'])
        self.assertEqual(len(self.submits),1)

    def test_incomplete_v2_retry_is_guarded_and_preserves_previous_on_failure(self):
        self.start(['visual']);self.collected();self.advance()
        row=self.co.read(self.ident);state=row['result']
        old={'version':2,'asOf':'2026-10-09','retrievedAt':'2026-10-09T12:00:00Z','gaps':['Missing quote']}
        state['visualFundamentals']=old;self.co.store(self.ident,state,row['status'],row.get('error'))
        before=self.co.read(self.ident);path='/api/research/assignments/'+self.ident+'/visual-supplement'
        with patch('stock_visual_fundamentals.collect',return_value=old) as collect:
            self.assertEqual(self.client.post(path,json={}).status_code,409)
            collect.assert_not_called()
            self.assertEqual(self.client.post(path,json={'expectedRetrievedAt':old['retrievedAt']}).status_code,409)
            self.assertEqual(before,self.co.read(self.ident))
        new={**old,'retrievedAt':'2026-10-10T12:00:00Z','gaps':[]}
        with patch('stock_visual_fundamentals.collect',return_value=new) as collect:
            for _ in range(2):self.assertEqual(self.client.post(path,json={'expectedRetrievedAt':old['retrievedAt']}).status_code,200)
            self.assertEqual(collect.call_count,1)
        after=self.co.read(self.ident)['result']
        self.assertEqual(after['visualFundamentals'],new)
        self.assertEqual(after['visualFundamentalRevisions'],[old])
        self.assertEqual(after['artifacts'],before['result']['artifacts']);self.assertEqual(len(self.submits),1)

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
    def test_queued_collection_reports_waiting_and_preserves_issue_without_research(self):
        self.start();self.collected()
        for status,step in [('queued','Waiting for browser collection worker'),('collecting','Collecting and verifying originals'),('unexpected','Waiting for verified collection progress')]:
            with self.subTest(status=status):
                collection={'id':self.ident,'status':status,'issue':'Another original export must finish first.','sourceProgress':{'documents':{},'tasks':[]}}
                with self.db(True) as (_,cur):
                    cur.execute("UPDATE app_settings SET value=%s,updated_at=NOW() WHERE key='collection_control_snapshot'",(json.dumps({'requests':[collection]}),))
                row=self.advance()
                self.assertEqual(row['status'],'running');self.assertEqual(row['result']['step'],step)
                public=self.co.public(row)
                self.assertEqual(public['result']['collection']['issue'],collection['issue'])
                self.assertEqual(public['result']['collection']['status'],status)
                self.assertFalse(row['result'].get('sources'));self.assertFalse(self.submits)

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
        self.resume_error=None;self.report_status='complete';self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/resume',json={'acknowledgeRetry':True,'expectedRevision':self.co.public(self.co.read(self.ident))['resumeRevision']}).status_code,200)
        self.assertEqual(self.advance()['status'],'complete');self.assertEqual(len(self.submits),1)
    def test_resume_receipt_blocks_stale_missing_and_cross_assignment_ack_before_child_work(self):
        self.start(['summary_note']);self.collected();row=self.advance()
        self.co.store(self.ident,row['result'],'attention','Synthetic uncertain outcome')
        row=self.co.read(self.ident);token=self.co.public(row)['resumeRevision']
        self.assertEqual(len(token),64)
        from research_assignments import resume_revision
        self.assertNotEqual(token,resume_revision({**row,'id':str(uuid.uuid4())}))
        from datetime import timedelta
        self.assertNotEqual(token,resume_revision({**row,'updated_at':row['updated_at']+timedelta(microseconds=1)}))
        self.co.store(self.ident,row['result'],'attention','New synthetic outcome')
        before=self.co.read(self.ident);path='/api/research/assignments/'+self.ident+'/resume'
        with patch.object(self.co,'invoke',side_effect=AssertionError('No child action')),patch.object(self.co,'wake') as wake:
            for body in ({'acknowledgeRetry':True},{'acknowledgeRetry':True,'expectedRevision':token},{'acknowledgeRetry':False,'expectedRevision':token},{'expectedRevision':None}):
                response=self.client.post(path,json=body)
                self.assertEqual(response.status_code,409,response.json)
                self.assertEqual(before,self.co.read(self.ident))
            wake.assert_not_called()
        current=self.co.public(before)['resumeRevision'];self.report_status='attention'
        with patch.object(self.co,'invoke',wraps=self.co.invoke) as invoke:
            self.assertEqual(self.client.post(path,json={'acknowledgeRetry':True,'expectedRevision':current}).status_code,200)
            actions=[call for call in invoke.call_args_list if call.args[0]=='stock_analysis.control']
            self.assertEqual(len(actions),1);self.assertIs(actions[0].args[2]['acknowledgeRetry'],True)
        with patch.object(self.co,'invoke',side_effect=AssertionError('No replay')):
            self.assertEqual(self.client.post(path,json={'acknowledgeRetry':True,'expectedRevision':current}).status_code,409)

    def test_stop_before_mac_receipt_creates_durable_cancel(self):
        self.start();self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/stop',json={}).status_code,200)
        self.assertEqual(self.advance()['status'],'cancelled');self.assertEqual(self.submits,{})
        with self.db() as (_,cur):cur.execute('SELECT input FROM mp_jobs WHERE id=%s',(child(self.ident,'cancel-collection'),));self.assertEqual(cur.fetchone()['input']['payload']['assignmentId'],self.ident)
    def stopping_assignment(self):
        self.start(['summary_note']);self.collected();row=self.advance()
        self.co.store(self.ident,row['result'],'attention','Synthetic delivery issue')
        return copy.deepcopy(row['result'])

    def test_failed_child_stop_fences_parent_and_preserves_outputs_until_retry(self):
        saved=self.stopping_assignment();original=self.co.invoke
        def fail(endpoint,method,body,params):
            if endpoint=='stock_analysis.control':return {'error':'Private provider detail'},503
            return original(endpoint,method,body,params)
        path='/api/research/assignments/'+self.ident
        with patch.object(self.co,'invoke',side_effect=fail):
            reply=self.client.post(path+'/stop',json={})
        self.assertEqual(reply.status_code,409);self.assertNotIn('Private',reply.json['error'])
        row=self.co.read(self.ident)
        self.assertEqual(row['status'],'attention');self.assertTrue(row['result']['stopRequested'])
        self.assertEqual(row['result']['artifacts'],saved['artifacts'])
        self.assertIn('Stop is not yet confirmed',row['error'])
        with patch.object(self.co,'invoke',side_effect=AssertionError('No dispatch while stop unresolved')):
            self.assertEqual(self.advance()['status'],'attention')
            self.assertEqual(self.client.post(path+'/resume',json={}).status_code,409)
        with self.db() as (_,cur):
            cur.execute('SELECT count(*) AS n FROM mp_jobs WHERE id=%s',(child(self.ident,'cancel-collection'),))
            self.assertEqual(cur.fetchone()['n'],1)
        self.assertEqual(self.client.post(path+'/stop',json={}).status_code,200)
        self.assertEqual(self.co.read(self.ident)['status'],'cancelled')
        self.assertEqual(self.co.read(self.ident)['result']['artifacts'],saved['artifacts'])
        self.assertEqual(len(self.submits),1)

    def test_lost_or_malformed_child_stop_receipt_never_claims_success(self):
        self.stopping_assignment();path='/api/research/assignments/'+self.ident+'/stop'
        for result in (({},200),({'ok':'true'},200),({'ok':True},503),RuntimeError('Private credentials')):
            with self.subTest(result=type(result).__name__):
                effect=result if isinstance(result,Exception) else lambda *a,r=result,**k:r
                with patch.object(self.co,'invoke',side_effect=effect):
                    reply=self.client.post(path,json={})
                self.assertEqual(reply.status_code,409)
                self.assertEqual(self.co.read(self.ident)['status'],'attention')
                self.assertNotIn('Private',reply.json['error'])

    def test_stop_accepts_verified_completed_child_not_foreign_receipt(self):
        self.stopping_assignment();report_id=child(self.ident,'stock-analysis')
        for delta in ({'id':str(uuid.uuid4())},{'ticker':'OTHER'},{'status':'running'},{}):
            report={**self.report(),**delta}
            def invoke(endpoint,*args):
                return ({'error':'No unfinished research found'},409) if endpoint=='stock_analysis.control' else (report,200)
            with patch.object(self.co,'invoke',side_effect=invoke):
                reply=self.client.post('/api/research/assignments/'+self.ident+'/stop',json={})
            self.assertEqual(reply.status_code,409 if delta else 200)
        self.assertEqual(self.co.read(self.ident)['status'],'cancelled')

    def test_stop_missing_child_requires_read_confirmation(self):
        self.stopping_assignment()
        with patch.object(self.co,'invoke',side_effect=[({'error':'missing'},409),({'error':'missing'},404)]):
            reply=self.client.post('/api/research/assignments/'+self.ident+'/stop',json={})
        self.assertEqual(reply.status_code,200)
        self.assertEqual(self.co.read(self.ident)['status'],'cancelled')

    def test_failed_report_stop_still_fences_note_and_missing_note_integration_holds(self):
        self.stopping_assignment()
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET input=input || %s::jsonb WHERE id=%s",(json.dumps({'labFullNote':{'allowPdfText':False}}),self.ident))
        stopped=[];self.co.lab_note_stop=lambda cur,note_id:stopped.append(note_id)
        with patch.object(self.co,'invoke',return_value=({'error':'unavailable'},503)):
            self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/stop',json={}).status_code,409)
        self.assertEqual(stopped,[child(self.ident,'full-lab-note')])
        self.co.lab_note_stop=None
        self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/stop',json={}).status_code,409)
        self.assertEqual(self.co.read(self.ident)['status'],'attention')
        self.co.lab_note_stop=lambda cur,note_id:None
        self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/stop',json={}).status_code,200)

    def test_note_stop_exception_retains_durable_parent_fence(self):
        self.stopping_assignment()
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET input=input || %s::jsonb WHERE id=%s",(json.dumps({'labFullNote':{'allowPdfText':False}}),self.ident))
        self.co.lab_note_stop=lambda *a:(_ for _ in ()).throw(RuntimeError('Private data'))
        reply=self.client.post('/api/research/assignments/'+self.ident+'/stop',json={})
        self.assertEqual(reply.status_code,503);self.assertNotIn('Private',reply.json['error'])
        self.assertTrue(self.co.read(self.ident)['result']['stopRequested'])
        self.assertEqual(self.advance()['status'],'attention')

    def test_changed_detailed_thesis_blocks_draft_without_losing_outputs(self):
        self.start();self.collected()
        with self.db(True) as (_,cur):cur.execute("INSERT INTO portfolio_analyses(ticker,company,analysis) VALUES(%s,'Synthetic','{}')",(self.ticker,))
        r=self.advance();self.assertEqual(r['status'],'attention');self.assertEqual(len(r['result']['artifacts']),3)

    def test_meeting_failure_preserves_thesis_and_only_missing_output_retried(self):
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['visual','thesis','meeting_prep'],'meeting':{'date':'2026-10-20','type':'management'}}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected()
        with patch('research_assignment_meeting.selection',side_effect=ValueError('No source-matched questions')):
            row=self.advance()
        self.assertEqual(row['status'],'attention');self.assertTrue(row['result']['thesisDraftId'])
        self.assertIn('visual',row['result']['artifacts'])
        self.assertEqual(row['result']['outputErrors'],{'meeting_prep':'No source-matched questions'})
        draft=row['result']['thesisDraftId'];original=self.co.invoke
        def invoke(endpoint,method,body,params):
            if endpoint=='research_meeting_handoff.handoff':return {'meetingId':31,'questionSetId':41},200
            if endpoint=='thesis_imports.drafts':raise AssertionError('Completed draft must not be recreated')
            return original(endpoint,method,body,params)
        self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/resume',json={}).status_code,200)
        with patch('research_assignment_meeting.selection',return_value={'frozen':'selection'}),patch.object(self.co,'invoke',side_effect=invoke):
            row=self.advance()
        self.assertEqual(row['status'],'complete',row['error']);self.assertEqual(row['result']['thesisDraftId'],draft)
        self.assertEqual(row['result']['meetingId'],31);self.assertEqual(row['result']['outputErrors'],{})

    def test_independent_output_failures_are_both_retained(self):
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['visual','thesis','meeting_prep'],'meeting':{'date':'2026-10-20','type':'management'}}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202);self.collected()
        with patch('research_assignment_meeting.selection',side_effect=ValueError('Missing questions')),patch('research_assignment_outputs.thesis',side_effect=ValueError('No supported thesis statements: synthetic')):
            row=self.advance()
        self.assertEqual(row['status'],'attention');self.assertIn('visual',row['result']['artifacts'])
        self.assertEqual(set(row['result']['outputErrors']),{'meeting_prep','thesis'})
        self.assertTrue(row['error'].startswith('No supported thesis statements'))

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
    def test_exhausted_quality_retry_exposes_manual_path_without_dispatch(self):
        original=self.report
        def exhausted():
            r=original();r['state']['citations']['/summary/investment_thesis/0']['review']='needs_review'
            r['state']['qualityRevisions']=[{'reason':'Prior explicit repair'}];return r
        self.report=exhausted;self.start();self.collected();r=self.advance()
        self.assertEqual(r['status'],'attention');self.assertIn('exhausted',r['error'])
        # Older saved errors also get accurate read-only recovery guidance.
        r['error']='No supported thesis statements were produced. Resume explicitly.'
        invoke=self.co.invoke;calls=[]
        def recording(endpoint,method,body,params):
            calls.append((endpoint,method));return invoke(endpoint,method,body,params)
        self.co.invoke=recording
        out=self.co.public(r)
        self.assertTrue(out['result']['manualThesisReviewRequired'])
        self.assertIn('no further automatic paid retry',out['error'])
        self.assertEqual(calls,[('stock_analysis.detail','GET')])
        self.assertTrue(out['result']['artifacts']);self.assertNotIn('thesisDraftId',out['result'])
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

    def test_event_purpose_survives_collection_dispatch_and_replay(self):
        purpose={'kind':'event_note','event':{'title':'Guidance update','date':'2026-09-30'}}
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['summary_note'],'researchPurpose':purpose,'labFullNote':{'allowPdfText':False}}
        self.co.lab_note_defaults=lambda:{'modelKey':'fixture','promptTemplate':'Holistic legacy prompt'}
        self.co.lab_note_dispatch=lambda ident,p,pack:{'noteId':str(uuid.uuid4()),'status':'draft','revision':1}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        frozen=self.co.read(self.ident)['input']
        self.assertIn('standalone professional event debrief',frozen['labNotePlan']['promptTemplate'])
        self.assertNotIn('Holistic legacy prompt',frozen['labNotePlan']['promptTemplate'])
        self.assertEqual(self.client.post('/api/research/assignments',json={**body,'researchPurpose':{'kind':'thesis_review'}}).status_code,409)
        self.collected();row=self.advance()
        self.assertEqual(row['status'],'complete')
        self.assertNotIn('thesisDraftId',row['result'])
        self.assertIn('Research purpose: event note',next(iter(self.submits.values()))['question'])
        self.assertIn('does not update the thesis',row['result']['artifacts']['summary_note']['html'])
        self.assertEqual(self.co.public(row)['input']['researchPurpose'],purpose)
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,200)
        self.assertEqual(len(self.submits),1)

    def test_source_only_collection_never_dispatches_research(self):
        self.co.has_key=lambda:False
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':[],'collectionOnly':True}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected();row=self.advance()
        self.assertEqual(row['status'],'complete');self.assertFalse(self.submits)
        self.assertNotIn('reportId',row['result']);self.assertEqual(len(row['result']['sources']),1)
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,200)

    def test_large_source_only_library_verifies_without_research(self):
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':[],
              'collectionOnly':True,'collectionLimit':25}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected()
        with self.db(True) as (_,cur):
            pack=[]
            for n in range(25):
                name=f'library-{n}.txt'
                cur.execute("INSERT INTO document_files(ticker,filename,file_data,metadata) SELECT ticker,%s,file_data,metadata FROM document_files WHERE ticker=%s AND filename='one.txt'",(name,self.ticker))
                pack.append({'filename':name,'sha256':self.sha})
            snapshot={'requests':[{'id':self.ident,'status':'complete','result':{'assignmentSources':pack}}]}
            cur.execute("UPDATE app_settings SET value=%s WHERE key='collection_control_snapshot'",(json.dumps(snapshot),))
        row=self.advance()
        self.assertEqual(row['status'],'complete',row.get('error'))
        self.assertEqual(len(row['result']['sources']),25)
        self.assertFalse(self.submits)
        self.assertNotIn('reportId',row['result'])

    def test_source_only_ambiguous_original_stays_attention_and_recovers_without_dispatch(self):
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':[],'collectionOnly':True}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected()
        other=base64.b64encode(b'Different synthetic original').decode()
        with self.db(True) as (_,cur):
            cur.execute('INSERT INTO document_files VALUES(%s,%s,%s,%s,%s::jsonb)',(self.ticker,'one.txt',other,'txt','{}'))
        with patch('command_thesis_bridge.file_hash',side_effect=AssertionError('Ambiguous bytes must not be hashed')):
            row=self.advance()
        self.assertEqual(row['status'],'attention');self.assertIn('Multiple stored originals',row['error'])
        self.assertNotIn('reportId',row['result']);self.assertNotIn('sources',row['result'])
        self.assertFalse(self.submits);self.assertEqual(row['input']['collectionOnly'],True)
        with self.db(True) as (_,cur):
            cur.execute('SELECT COUNT(*) AS n FROM document_files WHERE ticker=%s',(self.ticker,))
            self.assertEqual(cur.fetchone()['n'],2)
            # Simulate resolution of the disposable fixture only; production code never removes originals.
            cur.execute('DELETE FROM document_files WHERE ticker=%s AND file_data=%s',(self.ticker,other))
            cur.execute("UPDATE app_settings SET value='{}' WHERE key='collection_control_snapshot'")
        with patch.object(self.co,'invoke',side_effect=AssertionError('Collection recovery cannot call research')):
            self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/resume',json={}).status_code,200)
            recovered=self.advance()
        self.assertEqual(recovered['status'],'complete',recovered.get('error'))
        self.assertEqual(recovered['result']['collection'],row['result']['collection'])
        self.assertEqual(recovered['result']['sources'],[{'filename':'one.txt','sha256':self.sha}])
        self.assertFalse(self.submits)

    def test_source_only_same_byte_duplicate_identity_is_not_verified(self):
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':[],'collectionOnly':True}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected()
        with self.db(True) as (_,cur):
            cur.execute('INSERT INTO document_files SELECT * FROM document_files WHERE ticker=%s',(self.ticker,))
        row=self.advance()
        self.assertEqual(row['status'],'attention');self.assertIn('Multiple stored originals',row['error'])
        self.assertFalse(self.submits)

    def test_source_only_malformed_and_repeated_receipts_preserve_frozen_request(self):
        valid={'filename':'one.txt','sha256':self.sha}
        for pack in ({'filename':'one.txt'},['one.txt'],[{}],
                     [{**valid,'filename':' '}],[{**valid,'sha256':'invalid'}],[valid,valid]):
            with self.subTest(pack=pack):
                self.ident=str(uuid.uuid4());self.ticker='T'+uuid.uuid4().hex[:10].upper()
                body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':[],'collectionOnly':True}
                self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
                self.collected()
                with self.db(True) as (_,cur):
                    snapshot={'requests':[{'id':self.ident,'status':'complete','result':{'assignmentSources':pack}}]}
                    cur.execute("UPDATE app_settings SET value=%s WHERE key='collection_control_snapshot'",(json.dumps(snapshot),))
                row=self.advance()
                self.assertEqual(row['status'],'attention');self.assertIn('Collection receipts',row['error'])
                self.assertEqual(row['result']['collection'],snapshot['requests'][0])
                self.assertNotIn('reportId',row['result']);self.assertNotIn('sources',row['result'])
                self.assertFalse(self.submits)
                self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,200)
                self.assertEqual(self.co.read(self.ident),row)

    def test_completed_source_only_assignment_retains_legacy_frozen_receipts(self):
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':[],'collectionOnly':True}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected();row=self.advance()
        state=row['result'];state['sources'].append(copy.deepcopy(state['sources'][0]))
        self.co.store(self.ident,state,'complete');before=self.co.read(self.ident)
        self.assertEqual(self.advance(),before)
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,200)
        self.assertEqual(self.co.read(self.ident),before);self.assertFalse(self.submits)

    def test_stop_propagates_to_deterministic_full_note_child(self):
        stopped=[]
        self.co.lab_note_defaults=lambda:{'modelKey':'fixture','promptTemplate':'Fixture'}
        self.co.lab_note_stop=lambda cur,note_id:stopped.append(note_id)
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['summary_note'],'labFullNote':{'allowPdfText':False}}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/stop',json={}).status_code,200)
        self.assertEqual(stopped,[child(self.ident,'full-lab-note')])
        self.assertEqual(self.co.read(self.ident)['status'],'cancelled')

    def test_opt_in_full_note_waits_and_surfaces_downstream_failure(self):
        calls=[];note_id=str(uuid.uuid4());state={'status':'queued'}
        self.co.lab_note_defaults=lambda:{'modelKey':'fixture','promptTemplate':'Frozen fixture'}
        def dispatch(ident,p,pack):
            calls.append((ident,copy.deepcopy(p['labNotePlan']),copy.deepcopy(pack)))
            return {'noteId':note_id,**state}
        self.co.lab_note_dispatch=dispatch
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['summary_note'],'labFullNote':{'allowPdfText':False}}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected();row=self.advance()
        self.assertEqual(row['status'],'running');self.assertEqual(row['result']['labNoteId'],note_id)
        self.assertIn('summary_note',row['result']['artifacts'])
        state.update(status='attention',reason='provider_outcome_unknown')
        row=self.advance();self.assertEqual(row['status'],'attention');self.assertIn('provider_outcome_unknown',row['error'])
        state.update(status='draft',revision=1)
        self.co.store(self.ident,row['result'],'running');row=self.advance()
        self.assertEqual(row['status'],'complete');self.assertEqual(calls[0],calls[-1]);self.assertEqual(len(self.submits),1)
        self.assertNotIn('labNotePlan',self.co.public(row)['input'])

    def full_note_with_meeting(self, dispatch):
        self.co.lab_note_defaults=lambda:{'modelKey':'fixture','promptTemplate':'Frozen fixture'}
        self.co.lab_note_dispatch=dispatch
        body={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['visual','meeting_prep'],
              'meeting':{'date':'2026-10-20','type':'management'},'labFullNote':{'allowPdfText':False}}
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,202)
        self.collected()

    def test_meeting_failure_still_delivers_requested_full_note(self):
        calls=[]
        def dispatch(ident,p,pack):
            calls.append((ident,copy.deepcopy(p['labNotePlan']),copy.deepcopy(pack)))
            return {'noteId':child(ident,'full-lab-note'),'status':'draft','revision':1}
        self.full_note_with_meeting(dispatch)
        with patch('research_assignment_meeting.selection',side_effect=ValueError('Missing supported questions')):
            row=self.advance()
        self.assertEqual(row['status'],'attention')
        self.assertEqual(row['result']['labNoteStatus'],'draft')
        self.assertEqual(row['result']['labNoteId'],child(self.ident,'full-lab-note'))
        self.assertIn('visual',row['result']['artifacts'])
        self.assertEqual(row['result']['outputErrors'],{'meeting_prep':'Missing supported questions'})
        self.assertEqual(len(calls),1)

    def test_pending_note_with_failed_derivative_waits_for_explicit_resume(self):
        calls=[];status={'status':'queued'}
        def dispatch(ident,p,pack):
            calls.append((ident,copy.deepcopy(p),copy.deepcopy(pack)))
            return {'noteId':child(ident,'full-lab-note'),**status}
        self.full_note_with_meeting(dispatch)
        with patch('research_assignment_meeting.selection',side_effect=ValueError('Missing questions')):
            row=self.advance();self.advance()
        self.assertEqual(row['status'],'attention');self.assertIn('still in progress',row['result']['step'])
        self.assertEqual(len(calls),1)
        status['status']='draft';original=self.co.invoke
        def invoke(endpoint,method,body,params):
            if endpoint=='research_meeting_handoff.handoff':return {'meetingId':31,'questionSetId':41},200
            return original(endpoint,method,body,params)
        self.assertEqual(self.client.post('/api/research/assignments/'+self.ident+'/resume',json={}).status_code,200)
        with patch('research_assignment_meeting.selection',return_value={'frozen':'selection'}),patch.object(self.co,'invoke',side_effect=invoke):
            row=self.advance()
        self.assertEqual(row['status'],'complete',row['error'])
        self.assertEqual(row['result']['outputErrors'],{})
        self.assertEqual(calls[0],calls[1]);self.assertEqual(len(self.submits),1)

    def test_note_and_meeting_failures_remain_individually_visible(self):
        self.full_note_with_meeting(lambda *a:{'noteId':child(self.ident,'full-lab-note'),'status':'attention','reason':'provider_outcome_unknown'})
        with patch('research_assignment_meeting.selection',side_effect=ValueError('Missing questions')):
            row=self.advance()
        self.assertEqual(row['status'],'attention')
        self.assertEqual(set(row['result']['outputErrors']),{'meeting_prep','lab_note'})
        self.assertIn('provider_outcome_unknown',row['result']['outputErrors']['lab_note'])
        self.assertIn('visual',row['result']['artifacts'])
        self.assertEqual(row['result']['labNoteStatus'],'attention')

    def test_thesis_failure_keeps_repair_gate_and_independent_note_receipt(self):
        self.full_note_with_meeting(lambda *a:{'noteId':child(self.ident,'full-lab-note'),'status':'draft'})
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET input=jsonb_set(input,'{outputs}',%s::jsonb) WHERE id=%s",(json.dumps(['visual','thesis','meeting_prep']),self.ident))
        with patch('research_assignment_meeting.selection',side_effect=ValueError('Missing questions')),patch('research_assignment_outputs.thesis',side_effect=ValueError('No supported thesis statements: fixture')):
            row=self.advance()
        self.assertEqual(row['status'],'attention');self.assertTrue(row['error'].startswith('No supported thesis statements'))
        self.assertEqual(row['result']['labNoteStatus'],'draft')
        self.assertEqual(set(row['result']['outputErrors']),{'meeting_prep','thesis'})

    def test_full_note_unavailable_retains_derivative_error_and_no_false_receipt(self):
        self.full_note_with_meeting(None)
        with patch('research_assignment_meeting.selection',side_effect=ValueError('Missing questions')):
            row=self.advance()
        self.assertEqual(set(row['result']['outputErrors']),{'meeting_prep','lab_note'})
        self.assertNotIn('labNoteId',row['result']);self.assertEqual(row['status'],'attention')
        self.assertIn('unavailable',row['result']['outputErrors']['lab_note'])

    def test_invalid_original_still_blocks_full_note_before_any_dispatch(self):
        def forbidden(*a):raise AssertionError('Unverified originals must not dispatch')
        self.full_note_with_meeting(forbidden)
        with self.db(True) as (_,cur):
            cur.execute('UPDATE document_files SET file_data=%s WHERE ticker=%s',(base64.b64encode(b'changed').decode(),self.ticker))
        row=self.advance();self.assertEqual(row['status'],'attention')
        self.assertIn('hashes',row['error']);self.assertNotIn('labNoteId',row['result']);self.assertFalse(self.submits)

    def test_shared_meeting_output_recovers_lost_receipt_without_new_research(self):
        original_report=self.report
        def report():
            run=original_report();question='What supports the revenue outlook?'
            run['state']['report']['diligence_questions']=[question]
            run['state']['citations']['/diligence_questions/0']={**copy.deepcopy(run['state']['citations']['/summary/one_liner']),'statement':question}
            return run
        self.report=report
        requests=[];original_invoke=self.co.invoke
        def invoke(endpoint,method,body,params):
            if endpoint=='research_meeting_handoff.handoff':
                requests.append(copy.deepcopy(body))
                if len(requests)==1:return {'error':'Synthetic lost local meeting receipt'},503
                return {'meetingId':45,'questionSetId':46,'reused':True},200
            return original_invoke(endpoint,method,body,params)
        self.co.invoke=invoke
        p={**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['visual','meeting_prep'], 'meeting':{'date':'2026-10-20','type':'management'}}
        self.assertEqual(self.client.post('/api/research/assignments',json=p).status_code,202)
        self.collected();row=self.advance();self.assertEqual(row['status'],'attention')
        self.assertIn('visual',row['result']['artifacts']);self.assertIn('meetingSelection',row['result'])
        self.co.store(self.ident,row['result'],'running');row=self.advance()
        self.assertEqual(row['status'],'complete',row['error']);self.assertEqual(row['result']['meetingId'],45)
        self.assertEqual(requests[0],requests[1]);self.assertEqual(len(self.submits),1)
        self.assertNotIn('meetingSelection',self.co.public(row)['result'])
