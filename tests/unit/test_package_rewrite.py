import unittest
from research_package_rewrite import specification, validate, generate, preview


class RewriteTests(unittest.TestCase):
    def setUp(self):
        self.spec = specification({'fields':[{'id':'summary','label':'Summary','text':'Old','limit':300}], 'instructions':'Make specific'})

    def test_scope_and_bounds(self):
        for fields in ([], [self.spec['fields'][0]]*2, [{'id':'x','text':'x','label':'x','limit':999}]):
            with self.assertRaises(ValueError): specification({'fields':fields,'instructions':'Revise'})
        for raw in ({'fields':[]},{'fields':[{'id':'other','text':'New'}]}, {'fields':[{'id':'summary','text':'x'*301}]}):
            with self.assertRaises(ValueError): validate(raw,self.spec)

    def test_single_call_saved_before_validation_and_no_retry(self):
        saved=[]; calls=[]
        def ask(*args):
            calls.append(args)
            return {'fields':[{'id':'summary','text':'New'}]}
        inputs={'packageRewrite':self.spec,'writingPreferences':{}}
        result=generate({},[],ask,lambda s:saved.append(__import__('copy').deepcopy(s)),lambda:None,inputs)
        self.assertEqual(result['rewrite'],{'summary':'New'})
        self.assertEqual(len(calls),1)
        self.assertEqual(saved[0]['inFlight'],'package-rewrite')
        self.assertIn('rewriteResponse',saved[1]);self.assertNotIn('inFlight',saved[1])
        generate(result,[],ask,lambda s:None,lambda:None,inputs)
        self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError):generate({'inFlight':'package-rewrite'},[],ask,lambda s:None,lambda:None,inputs)
        self.assertEqual(len(calls),1)

    def test_invalid_response_retained_and_not_recharged(self):
        saved=[];calls=[]
        def ask(*args):calls.append(1);return {'fields':[]}
        inputs={'packageRewrite':self.spec,'writingPreferences':{}}
        with self.assertRaises(ValueError):generate({},[],ask,lambda s:saved.append(__import__('copy').deepcopy(s)),lambda:None,inputs)
        with self.assertRaises(ValueError):generate(saved[-1],[],ask,lambda s:None,lambda:None,inputs)
        self.assertEqual(len(calls),1)

    def test_quote_fences_preferences_sources_and_scope(self):
        from model_registry import MODELS
        model=next(k for k,v in MODELS.items() if 'input' in v and 'output' in v)
        a=preview(self.spec,[],{},model,'parent')
        b=preview(self.spec,[],{'rules':['No broker names']},model,'parent')
        self.assertNotEqual(a['fingerprint'],b['fingerprint'])
        self.assertGreater(a['estimatedUsd'],0)
        self.assertEqual(a['calls'],1)

if __name__=='__main__':unittest.main()

class RewriteRouteTests(unittest.TestCase):
    def test_authorization_changed_scope_and_exact_replay(self):
        import contextlib,json,uuid
        from unittest.mock import patch
        from flask import Flask
        from company_research import create_blueprint
        from model_registry import MODELS
        model=next(k for k,v in MODELS.items() if 'input' in v and 'output' in v)
        parent={'id':'parent','ticker':'ABT','status':'complete','input':{'filenames':['x'],'hashes':{'x':'hash'}},'sources':[{'id':'s','text':'Frozen excerpt'}],'baseline':{'revision':0}}
        runs={};result=None
        class Cursor:
            def execute(self,sql,args=()):
                nonlocal result
                result=None
                if 'SELECT id,payload_hash' in sql:result=runs.get(args[0])
                elif 'SELECT * FROM stock_analysis_runs' in sql:result=parent
                elif 'INSERT INTO stock_analysis_runs' in sql:
                    runs[args[0]]={'id':args[0],'payload_hash':args[2],'input':json.loads(args[5])}
            def fetchone(self):return result
        @contextlib.contextmanager
        def db(**kwargs):yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,lambda *a:None,lambda k:'key',lambda:model,studio=True))
        client=app.test_client();body={**self.make_spec(),'requestId':str(uuid.uuid4())}
        with patch('research_preferences.read',return_value={'rules':[]}),patch('company_research.threading.Thread') as thread:
            quote=client.post('/api/research/stock-analysis-run/parent/rewrite/prepare',json=body)
            self.assertEqual(quote.status_code,200);thread.assert_not_called()
            body['fingerprint']=quote.json['fingerprint']
            self.assertEqual(client.post('/api/research/stock-analysis-run/parent/rewrite/start',json=body).status_code,409)
            body['confirmed']=True
            changed={**body,'instructions':'different'}
            self.assertEqual(client.post('/api/research/stock-analysis-run/parent/rewrite/start',json=changed).status_code,409)
            self.assertEqual(client.post('/api/research/stock-analysis-run/parent/rewrite/start',json=body).status_code,202)
            self.assertEqual(thread.call_count,1)
            self.assertEqual(client.post('/api/research/stock-analysis-run/parent/rewrite/start',json=body).status_code,200)
            self.assertEqual(thread.call_count,1)
            self.assertEqual(client.post('/api/research/stock-analysis-run/parent/rewrite/start',json=changed).status_code,409)
            self.assertEqual(next(iter(runs.values()))['input']['packageRewrite']['fields'],body['fields'])

    def make_spec(self):
        return {'fields':[{'id':'summary','label':'Summary','text':'Old','limit':300}],'instructions':'Rewrite clearly'}
