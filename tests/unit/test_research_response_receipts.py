import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from company_research import KnownResponseError, public_run
from research_response_receipts import response_scope, cached_response, preserve_response, archive_for_retry


class ReceiptTests(unittest.TestCase):
    def test_scope_filter_archive_and_public_redaction(self):
        state={'inFlight':'report-0','completed':['earlier']};saved=[]
        with response_scope('report-0',state,lambda v:saved.append(copy.deepcopy(v))):
            preserve_response({'text':'PRIVATE','stop_reason':'end_turn','headers':'SECRET'})
            self.assertEqual(cached_response()['text'],'PRIVATE')
        self.assertIsNone(cached_response())
        self.assertEqual(saved[0]['completed'],['earlier'])
        self.assertNotIn('SECRET',json.dumps(state))
        self.assertNotIn('PRIVATE',json.dumps(public_run({'state':state})))
        archive_for_retry(state,'report-0')
        self.assertEqual(state['previousReturnedResponses'][0]['response']['text'],'PRIVATE')
        self.assertNotIn('PRIVATE',json.dumps(public_run({'state':state})))

    def test_real_call_function_saves_before_parse_and_reuses_return(self):
        tree=ast.parse(Path('app_v3.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_company_research_call')
        calls=[];saved=[];state={}
        env=dict(resolve_picker_spec=lambda _:dict(provider='anthropic',model='synthetic'),PICKER_DEFAULT_MODEL='synthetic',
                 _call_anthropic_stream=lambda **kwargs:calls.append(kwargs) or {'text':'bad json','stop_reason':'end_turn'},
                 record_llm_usage=lambda *a,**kw:None,_extract_json=json.loads,company_research=SimpleNamespace(KnownResponseError=KnownResponseError))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'app_v3.py','exec'),env)
        with response_scope('report-0',state,lambda v:saved.append(copy.deepcopy(v))):
            for _ in range(2):
                with self.assertRaises(KnownResponseError):env['_company_research_call']('synthetic prompt','synthetic key',100,'run','report-0')
        self.assertEqual(len(calls),1)
        self.assertEqual(saved[0]['returnedResponses']['report-0']['text'],'bad json')

    def test_persistence_failure_prevents_parsing(self):
        def fail(_):raise RuntimeError('synthetic persistence failure')
        with response_scope('stage',{},fail),self.assertRaises(RuntimeError):
            preserve_response({'text':'returned'})

    def test_public_diagnostics_are_allowlisted_and_do_not_mutate_private_state(self):
        from research_response_receipts import receipt_metadata
        import hashlib
        text='private response — α'
        row={'state':{'returnedResponses':{'report-0':{'text':text,'stop_reason':'PRIVATE STOP','headers':'SECRET'}}}}
        out=public_run(row)
        item=out['responseRecovery']['receipts'][0]
        self.assertEqual(item,{'stage':'report-0','characters':len(text),'sha256':hashlib.sha256(text.encode()).hexdigest(),'stopReason':'unavailable'})
        self.assertNotIn(text,json.dumps(out));self.assertNotIn('PRIVATE STOP',json.dumps(out))
        self.assertEqual(row['state']['returnedResponses']['report-0']['text'],text)
        self.assertEqual(receipt_metadata('x',{'text':'','stop_reason':'max_tokens'})['characters'],0)
        self.assertIsNone(receipt_metadata('x',{'text':None})['sha256'])
