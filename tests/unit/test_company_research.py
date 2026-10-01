import copy
import unittest
from company_research import (SECTIONS,GROUPS,validate_sections,apply_review,generate,
                              eligible,public_run,validate_handoff)
SOURCE={'id':'s','filename':'Synthetic original.txt','text':'The company reported revenue of 100 million dollars for fiscal 2025.'}

def response(group,with_claim=True):
    return {'sections':[{'id':sid,'claims':[{'statement':'Revenue was 100 million dollars in FY25.','basis':'reported_fact',
        'evidence':[{'sourceId':'s','excerpt':SOURCE['text']}]}] if with_claim else [],'gaps':['Consensus unavailable']} for sid,_ in group]}

class ResearchTests(unittest.TestCase):
    def test_sections_are_complete_unique_and_typed(self):
        output=validate_sections(response(GROUPS[0]),GROUPS[0],[SOURCE])
        self.assertTrue(output[0]['claims'][0]['passageMatched'])
        for mutation in ['missing','duplicate','source','basis']:
            raw=response(GROUPS[0])
            if mutation=='missing':raw['sections'].pop()
            if mutation=='duplicate':raw['sections'][1]=raw['sections'][0]
            if mutation=='source':raw['sections'][0]['claims'][0]['evidence'][0]['sourceId']='invented'
            if mutation=='basis':raw['sections'][0]['claims'][0]['basis']='consensus'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_sections(raw,GROUPS[0],[SOURCE])

    def test_unmatched_and_reviewer_disagreement_cannot_be_promoted(self):
        raw=response(GROUPS[0]);raw['sections'][0]['claims'][0]['evidence'][0]['excerpt']='This is an invented quote that does not occur in the original.'
        sections=validate_sections(raw,GROUPS[0],[SOURCE]);claims=[c for s in sections for c in s['claims']]
        review={'findings':[{'claimId':c['id'],'status':'supported','reason':'Synthetic finding'} for c in claims]}
        review['findings'][1]['status']='needs_review'
        result=apply_review(sections,review)
        self.assertEqual(result[0]['claims'][0]['review'],'needs_review')
        self.assertEqual(result[1]['claims'][0]['review'],'needs_review')
        self.assertEqual(result[2]['claims'][0]['review'],'supported')
        self.assertEqual(sections[2]['claims'][0]['review'],'pending')
        review['findings'].pop()
        with self.assertRaises(ValueError):apply_review(sections,review)

    def test_interruption_never_repeats_unknown_paid_attempt_automatically(self):
        saved=[];calls=[]
        def ask(prompt,tokens,stage):
            calls.append(stage)
            if stage=='group-1':raise TimeoutError('Synthetic provider timeout')
            return response(GROUPS[0])
        with self.assertRaises(TimeoutError):generate({},[SOURCE],{},ask,lambda s:saved.append(copy.deepcopy(s)),lambda:None)
        self.assertEqual(saved[-1]['completed'],['group-0'])
        self.assertEqual(saved[-1]['inFlight'],'group-1')
        with self.assertRaises(ValueError):generate(saved[-1],[SOURCE],{},ask,lambda s:None,lambda:None)
        self.assertEqual(calls,['group-0','group-1'])
        resumed=copy.deepcopy(saved[-1]);resumed.pop('inFlight');resumed['retries']=['group-1'];calls=[]
        def finish(prompt,tokens,stage):
            calls.append(stage)
            if stage.startswith('group-'):return response(GROUPS[int(stage.split('-')[1])],False)
            return {'findings':[{'claimId':sid+':0','status':'supported','reason':'Matches synthetic source'} for sid,_ in GROUPS[0]]}
        result=generate(resumed,[SOURCE],{},finish,lambda s:None,lambda:None)
        self.assertNotIn('group-0',calls);self.assertEqual(len(result['sections']),22)
        self.assertEqual(len(result['completed']),12)
        self.assertFalse(result.get('inFlight'))

    def test_stop_before_call_spends_nothing(self):
        calls=[]
        def stop():raise RuntimeError('Stopped')
        with self.assertRaises(RuntimeError):generate({},[SOURCE],{},lambda *a:calls.append(a),lambda s:None,stop)
        self.assertEqual(calls,[])

    def test_usage_restrictions_and_public_source_redaction(self):
        self.assertTrue(eligible({'usage':'research'}))
        for metadata in [{'usage':'reference_only'},{'nested':{'aiAllowed':False}},{'records':[{'usage':'reference_only'}]},'broken JSON']:
            self.assertFalse(eligible(metadata))
        result=public_run({'sources':[SOURCE],'payload_hash':'secret','state':{}})
        self.assertNotIn('text',result['sources'][0]);self.assertIn('text',SOURCE)

    def test_handoff_requires_same_company_baseline_and_original_hashes(self):
        run={'ticker':'ABC','status':'complete','baseline':{'revision':4},'input':{'hashes':{'original':'hash'}}}
        validate_handoff(run,'ABC',4,{'original':'hash'})
        for ticker,revision,hashes in [('OTHER',4,{'original':'hash'}),('ABC',5,{'original':'hash'}),('ABC',4,{'original':'changed'}),('ABC',4,{})]:
            with self.assertRaises(ValueError):validate_handoff(run,ticker,revision,hashes)

class ApiTests(unittest.TestCase):
    def setUp(self):
        import base64,json
        from contextlib import contextmanager
        from flask import Flask
        from company_research import create_blueprint
        from unittest.mock import patch
        self.rows={};self.revision=2;self.restricted=False
        outer=self
        class Cursor:
            def execute(self,sql,args=()):
                self.result=[]
                if sql.startswith(('CREATE','ALTER','SELECT pg_advisory')):return
                if 'to_regclass' in sql:self.result=[{'name':'investment_case_versions'}]
                elif 'FROM investment_case_versions' in sql:self.result=[{'revision':outer.revision,'body':{'thesis':'Synthetic baseline'}}]
                elif 'FROM document_files' in sql:self.result=[{'filename':'Original.txt','file_data':base64.b64encode(b'Synthetic original content').decode(),'metadata':{'usage':'reference_only' if outer.restricted else 'research'}}]
                elif sql.startswith('SELECT') and 'FROM company_research_runs' in sql:
                    if 'WHERE id=' in sql:self.result=[copy.deepcopy(outer.rows[args[0]])] if args[0] in outer.rows else []
                    else:
                        self.result=[copy.deepcopy(r) for r in outer.rows.values() if r['ticker']==args[0] and ("status IN" not in sql or r['status'] in ('queued','running','attention'))]
                elif sql.startswith('INSERT INTO company_research_runs'):
                    ident,ticker,signature,version,model,inputs,baseline=args
                    outer.rows[ident]={'id':ident,'ticker':ticker,'payload_hash':signature,'version':version,'model':model,'input':json.loads(inputs),'baseline':json.loads(baseline),'sources':[],'state':{},'status':'queued','cancel_requested':False,'owner':None,'created_at':'2026-10-01','updated_at':'2026-10-01'}
                elif sql.startswith('UPDATE company_research_runs SET cancel_requested'):
                    acquired,ident=args;r=outer.rows.get(ident)
                    if r and r['status']!='complete':
                        r['cancel_requested']=True
                        if acquired or r['status'] in ('queued','attention'):r['status']='cancelled'
                        self.result=[{'id':ident}]
                elif sql.startswith("UPDATE company_research_runs SET status='queued'"):
                    state,ident=args;r=outer.rows[ident];r.update(status='queued',owner=None,state=json.loads(state),error=None,cancel_requested=False)
                else:raise AssertionError(sql)
            def fetchone(self):return self.result[0] if self.result else None
            def fetchall(self):return self.result
        @contextmanager
        def db(commit=False):yield None,Cursor()
        @contextmanager
        def worker(*args):yield True
        self.thread=patch('company_research.threading.Thread').start();self.addCleanup(patch.stopall)
        patch('amendment_ownership.worker_session',worker).start()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,lambda *a:None,lambda key:'synthetic-key',lambda:'synthetic-model'))
        self.client=app.test_client()
    def payload(self):
        import uuid
        return {'requestId':str(uuid.uuid4()),'filenames':['Original.txt'],'revision':self.revision,'confirmed':True}
    def test_submission_retry_is_idempotent_and_freezes_case_and_hashes(self):
        p=self.payload();url='/api/research/company/ABC'
        self.assertEqual(self.client.post(url,json=p).status_code,202)
        self.assertTrue(self.client.post(url,json=p).json['replayed'])
        self.assertEqual(self.thread.call_count,1)
        row=self.rows[p['requestId']];self.assertEqual(row['baseline']['revision'],2)
        self.assertEqual(len(row['input']['hashes']['Original.txt']),64)
        self.assertEqual(self.client.post(url,json={**p,'revision':1}).status_code,409)
        self.assertEqual(self.client.post(url,json=self.payload()).status_code,409)
    def test_missing_confirmation_restrictions_and_changed_case_fail_before_dispatch(self):
        url='/api/research/company/ABC';p=self.payload()
        self.assertEqual(self.client.post(url,json={**p,'confirmed':False}).status_code,400)
        self.assertEqual(self.client.post(url,json={**p,'revision':0}).status_code,409)
        self.restricted=True
        self.assertEqual(self.client.post(url,json=p).status_code,409)
        self.assertEqual(self.thread.call_count,0)
    def test_restart_readback_and_explicit_retry_keep_frozen_baseline(self):
        p=self.payload();self.client.post('/api/research/company/ABC',json=p);ident=p['requestId'];row=self.rows[ident]
        row.update(status='running',state={'completed':['group-0'],'inFlight':'group-1'},sources=[SOURCE])
        self.revision=3
        detail=self.client.get('/api/research/company-run/'+ident).json
        self.assertTrue(detail['baselineStale']);self.assertNotIn('text',detail['sources'][0])
        url='/api/research/company-run/'+ident+'/resume'
        self.assertEqual(self.client.post(url,json={}).status_code,409)
        self.assertEqual(self.client.post(url,json={'acknowledgeRetry':True}).status_code,200)
        self.assertEqual(row['baseline']['revision'],2);self.assertEqual(row['state']['completed'],['group-0'])
        self.assertEqual(row['state']['retries'],['group-1'])
    def test_stop_recovers_orphaned_running_state_and_completed_work_is_immutable(self):
        p=self.payload();self.client.post('/api/research/company/ABC',json=p);ident=p['requestId'];row=self.rows[ident]
        row['status']='running';url='/api/research/company-run/'+ident+'/stop'
        self.assertEqual(self.client.post(url,json={}).status_code,200);self.assertEqual(row['status'],'cancelled')
        row['status']='complete'
        self.assertEqual(self.client.post(url,json={}).status_code,409)
        self.assertEqual(self.client.post('/api/research/company-run/'+ident+'/resume',json={}).status_code,409)


class ProviderAttemptTests(unittest.TestCase):
    def test_stream_passes_explicit_retry_policy_without_breaking_nonstream(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        tree=ast.parse(Path('app_v3.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_call_anthropic','_call_anthropic_stream')]
        sdk=MagicMock();ns={'anthropic':sdk}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'provider-test','exec'),ns)
        kwargs=dict(messages=[],system='',model='synthetic',max_tokens=5,timeout=2,api_key='synthetic')
        stream=sdk.Anthropic.return_value.messages.stream.return_value.__enter__.return_value
        stream.text_stream=['synthetic response'];stream.get_final_message.return_value=SimpleNamespace(usage=None,stop_reason='end_turn')
        ns['_call_anthropic_stream'](**kwargs,max_retries=0)
        self.assertEqual(sdk.Anthropic.call_args.kwargs['max_retries'],0)
        ns['_call_anthropic_stream'](**kwargs)
        self.assertEqual(sdk.Anthropic.call_args.kwargs['max_retries'],2)
        sdk.Anthropic.return_value.messages.create.return_value=SimpleNamespace(content=[],usage=SimpleNamespace(input_tokens=0,output_tokens=0),stop_reason='end_turn')
        ns['_call_anthropic'](**kwargs)
        self.assertNotIn('max_retries',sdk.Anthropic.call_args.kwargs)
