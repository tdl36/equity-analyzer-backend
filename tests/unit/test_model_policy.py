"""Hermetic migration, accounting, retirement and real SDK request-shape checks."""
import ast
import copy
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import httpx
import anthropic
import openai
import model_registry as registry

TREE = ast.parse(Path('app_v3.py').read_text())

def adapter(name, **scope):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope.update(model_registry=registry)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'fixture', 'exec'), scope)
    return scope[name]

class PolicyTests(unittest.TestCase):
    def test_registry_valid_and_research_retained(self):
        self.assertEqual(registry.validate(registry.REGISTRY), [])
        self.assertEqual(registry.role('research'), 'claude-opus-4-6')
        self.assertEqual(registry.role('summary_check'), 'claude-sonnet-5-5')

    def test_retirement_boundary_and_unknown_future_not_guessed(self):
        registry.ensure_active('o4-mini', date(2026,10,22))
        with self.assertRaisesRegex(ValueError, 'retired'): registry.ensure_active('o4-mini', date(2026,10,23))
        self.assertIsNone(registry.estimate('future-model', {'input_tokens':1000}))
        self.assertEqual(registry.request_options('future-model'), {})

    def test_options_are_copies_and_do_not_mutate_policy(self):
        opts=registry.request_options('claude-sonnet-5-5');opts['thinking']['type']='disabled'
        self.assertEqual(registry.request_options('claude-sonnet-5-5')['thinking'], {'type':'between_tools'})

    def migration(self, role='workhorse'):
        before=copy.deepcopy(registry.REGISTRY);after=copy.deepcopy(before)
        old=before['roles'][role];new='synthetic-successor'
        after['models'][new]=copy.deepcopy(before['models'][old]);after['roles'][role]=new
        proof={role:dict(officialSources=['https://platform.claude.com/docs/en/models/overview'],accountAvailable=True,compatibilityTested=True,qualityEvidence='Synthetic policy fixture only',tokenMultiplier=1,noAdditionalThinking=True,sameOrLowerTokenCaps=True)}
        return before,after,proof,new

    def test_equal_cost_requires_all_evidence(self):
        a,b,p,m=self.migration()
        self.assertEqual(registry.migration_errors(a,b,p),[])
        self.assertTrue(registry.migration_errors(a,b,{}))
        p['workhorse']['tokenMultiplier']=1.3
        self.assertTrue(registry.migration_errors(a,b,p))

    def test_nonfinite_multiplier_and_policy_weakening_are_rejected(self):
        a,b,p,m=self.migration();p['workhorse']['tokenMultiplier']=float('nan')
        self.assertTrue(registry.migration_errors(a,b,p))
        a,b,p,m=self.migration();b['protectedRoles']=[]
        self.assertTrue(registry.migration_errors(a,b,p))

    def test_research_and_hidden_cost_changes_blocked(self):
        a,b,p,m=self.migration('research')
        self.assertTrue(registry.migration_errors(a,b,p))
        a,b,p,m=self.migration();b['models'][m]['cacheRead']*=2
        self.assertTrue(registry.migration_errors(a,b,p))
        a,b,p,m=self.migration();p['workhorse']['noAdditionalThinking']=False
        self.assertTrue(registry.migration_errors(a,b,p))

    def test_cache_billing_provider_difference_and_long_context(self):
        u={'input_tokens':1000000,'output_tokens':100000,'cache_read_input_tokens':500000}
        # OpenAI input includes cached tokens. Long-context rates apply to the whole request.
        self.assertAlmostEqual(registry.estimate('gpt-6-luna',u),.185)
        u={'input_tokens':1000000,'output_tokens':100000,'cache_read_input_tokens':500000,'cache_creation_input_tokens':100000,'cache_creation_1h_input_tokens':50000}
        self.assertAlmostEqual(registry.estimate('claude-sonnet-5-5',u),3.425)

class AdapterTests(unittest.TestCase):
    def test_openai_native_pdf_reasoning_and_cache_with_real_sdk(self):
        captured=[]
        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200,json={'id':'fixture','object':'chat.completion','created':0,'model':'gpt-6-luna','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'source result'}}],'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'prompt_tokens_details':{'cached_tokens':50}}})
        client=openai.OpenAI(api_key='synthetic',http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        fn=adapter('_call_openai',openai=SimpleNamespace(OpenAI=lambda **k:client))
        result=fn(messages=[{'role':'user','content':[{'type':'document','source':{'type':'base64','media_type':'application/pdf','data':'UERG'}}]}],system='',model='gpt-6-luna',max_tokens=300,timeout=5,api_key='synthetic')
        self.assertEqual(captured[0]['reasoning_effort'],'none')
        self.assertEqual(captured[0]['max_completion_tokens'],300)
        self.assertNotIn('max_tokens',captured[0])
        self.assertEqual(captured[0]['messages'][0]['content'][0]['file']['file_data'],'data:application/pdf;base64,UERG')
        self.assertEqual(result['usage']['cache_read_input_tokens'],50)
        self.assertEqual(result['stop_reason'],'stop')
        with self.assertRaisesRegex(ValueError,'not discarded'):
            fn(messages=[{'role':'user','content':[{'type':'document','source':{'type':'url','url':'https://example.test/private'}}]}],system='',model='gpt-6-luna',max_tokens=300,timeout=5,api_key='synthetic')
        self.assertEqual(len(captured),1)

    def test_sonnet_options_serialized_and_thinking_block_not_read_as_text(self):
        captured=[]
        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200,json={'id':'fixture','type':'message','role':'assistant','model':'claude-sonnet-5-5','content':[{'type':'thinking','thinking':'','signature':'fixture'},{'type':'text','text':'answer'}],'stop_reason':'end_turn','usage':{'input_tokens':100,'output_tokens':20,'cache_read_input_tokens':30}})
        client=anthropic.Anthropic(api_key='synthetic',http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        fn=adapter('_call_anthropic',anthropic=SimpleNamespace(Anthropic=lambda **k:client))
        result=fn(messages=[{'role':'user','content':'fixture'}],system='',model='claude-sonnet-5-5',max_tokens=100,timeout=5,api_key='synthetic')
        self.assertEqual(result['text'],'answer')
        self.assertEqual(captured[0]['thinking'],{'type':'between_tools'})
        self.assertEqual(captured[0]['output_config']['effort'],'high')
        self.assertEqual(result['usage']['cache_read_input_tokens'],30)

    def test_saved_retired_picker_is_not_silently_replaced(self):
        spec={'key':'old','model':'o4-mini','provider':'openai'}
        fn=adapter('resolve_picker_spec',PICKER_DEFAULT_MODEL='default',PICKER_MODEL_BY_KEY={'old':spec,'default':{'model':'gpt-6-luna'}})
        with patch.object(registry,'ensure_active',side_effect=ValueError('retired')):
            with self.assertRaisesRegex(ValueError,'retired'):fn('old')

class MaintenanceApiTests(unittest.TestCase):
    def endpoint(self, row=None):
        from contextlib import contextmanager
        from flask import Flask, request, jsonify
        from datetime import datetime
        cur=Mock();cur.fetchone.return_value=row
        @contextmanager
        def db(**kw):yield Mock(),cur
        node=copy.deepcopy(next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='model_maintenance_status'))
        node.decorator_list=[]
        scope=dict(request=request,jsonify=jsonify,datetime=datetime,json=json,get_db=db,model_registry=registry)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'fixture','exec'),scope)
        app=Flask(__name__);app.add_url_rule('/maintenance',view_func=scope['model_maintenance_status'],methods=['GET','POST'])
        return app.test_client(),cur

    def test_status_saved_with_server_revision_and_retrieved(self):
        client,cur=self.endpoint()
        response=client.post('/maintenance',json={'status':'checked','summary':'No eligible changes','revision':'forged'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['revision'],registry.REVISION)
        stored=json.loads(cur.execute.call_args.args[1][1])
        self.assertEqual(stored['summary'],'No eligible changes')
        client,_=self.endpoint({'value':json.dumps(stored)})
        self.assertEqual(client.get('/maintenance').json['lastCheck'],stored)

    def test_bad_report_does_not_write(self):
        client,cur=self.endpoint()
        self.assertEqual(client.post('/maintenance',json={'status':'success'}).status_code,400)
        cur.execute.assert_not_called()
