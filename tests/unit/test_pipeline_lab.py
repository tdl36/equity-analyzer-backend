import json
import unittest
from contextlib import contextmanager
from flask import Flask
from pipeline_lab import config,create_blueprint

class LabTests(unittest.TestCase):
    def setUp(self):
        self.values={};outer=self
        class Cursor:
            def execute(self,sql,args):
                self.key=args[0]
                if sql.startswith('INSERT'):outer.values[args[0]]=args[1]
            def fetchone(self):return {'value':outer.values[self.key]} if self.key in outer.values else None
        @contextmanager
        def db(commit=False):yield None,Cursor()
        a=Flask(__name__);a.register_blueprint(create_blueprint(db));self.client=a.test_client()
    def test_defaults_and_explicit_empty_selection(self):
        r=self.client.get('/api/pipeline-lab/config/ABT');self.assertEqual(r.json['selection'],{'thesis':[],'note':[]});self.assertIn('no-store',r.headers['Cache-Control'])
        r=self.client.put('/api/pipeline-lab/config/ABT',json={'revision':0,'selection':{'thesis':[],'note':[]}});self.assertEqual(r.status_code,200)
    def test_separate_tasks_stocks_and_conflicting_devices(self):
        body={'revision':0,'selection':{'thesis':['charlie:thesis_1'],'note':[]}}
        r=self.client.put('/api/pipeline-lab/config/ABT',json=body);self.assertEqual(r.json['revision'],1)
        self.assertEqual(self.client.put('/api/pipeline-lab/config/ABT',json=body).json,r.json)
        body['selection']['note']=['other']
        self.assertEqual(self.client.put('/api/pipeline-lab/config/ABT',json=body).status_code,409)
        self.assertEqual(self.client.get('/api/pipeline-lab/config/UNH').json['revision'],0)
        self.assertEqual(set(self.values),{'pipeline_lab_v1:ABT'})
    def test_invalid_payloads_do_not_write(self):
        for rows in [None,['x','x'],['x']*1001,[True]]:
            self.assertEqual(self.client.put('/api/pipeline-lab/config/ABT',json={'revision':0,'selection':{'thesis':rows,'note':[]}}).status_code,400)
        self.assertEqual(self.values,{})
        self.assertEqual(self.client.put('/api/pipeline-lab/config/ABT',json={'revision':True,'selection':{'thesis':[],'note':[]}}).status_code,400)

class OriginalChecks(unittest.TestCase):
    def test_exact_bytes_and_task_binding(self):
        import base64
        from pipeline_lab import check_originals
        ref='charlie:thesis_1';docs={ref:{'filename':'one.pdf','file_data':base64.b64encode(b'original').decode()}}
        a=check_originals('ABT','thesis',1,[ref],docs)
        self.assertEqual(a['blockers'],[]);self.assertEqual(a['verified'][0]['bytes'],8)
        for ticker,task,rev in [('UNH','thesis',1),('ABT','note',1),('ABT','thesis',2)]:
            self.assertNotEqual(a['digest'],check_originals(ticker,task,rev,[ref],docs)['digest'])
        docs[ref]['file_data']=base64.b64encode(b'changed').decode()
        self.assertNotEqual(a['digest'],check_originals('ABT','thesis',1,[ref],docs)['digest'])
    def test_fail_closed_empty_missing_local_invalid_and_duplicate_names(self):
        from pipeline_lab import check_originals
        for refs,docs in [([],{}),(['charlie:thesis_1'],{}),(['icloud:["main","x"]'],{}),(['charlie:research_1'],{}),([str(n) for n in range(9)],{})]:
            r=check_originals('ABT','note',1,refs,docs);self.assertTrue(r['blockers']);self.assertIsNone(r['digest'])
        for data in ['',None,'%%%', 'a'*20000001]:
            r=check_originals('ABT','note',1,['charlie:thesis_1'],{'charlie:thesis_1':{'filename':'a','file_data':data}})
            self.assertTrue(r['blockers']);self.assertIsNone(r['digest'])
        docs={f'charlie:thesis_{n}':{'filename':'same','file_data':'YQ=='} for n in [1,2]}
        r=check_originals('ABT','note',1,list(docs),docs);self.assertTrue(r['blockers']);self.assertIsNone(r['digest'])
    def test_endpoint_saved_revision_and_ticker_scoped_read(self):
        calls=[]
        class Cursor:
            def execute(self,sql,args):calls.append((sql,args))
            def fetchone(self):return {'value':json.dumps({'revision':2,'selection':{'thesis':['charlie:thesis_5'],'note':[]}})}
            def fetchall(self):return [{'id':'5','filename':'a.pdf','file_data':'YQ=='}]
        @contextmanager
        def db(commit=False):
            self.assertFalse(commit);yield None,Cursor()
        a=Flask(__name__);a.register_blueprint(create_blueprint(db));client=a.test_client()
        self.assertEqual(client.post('/api/pipeline-lab/check/ABT',json={'revision':1,'task':'thesis'}).status_code,409)
        self.assertEqual(len(calls),1)
        r=client.post('/api/pipeline-lab/check/ABT',json={'revision':2,'task':'thesis'})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json['blockers'],[])
        self.assertFalse(r.json['generationAvailable']);self.assertEqual(calls[-1][1],('ABT',['5']))
        self.assertNotIn('file_data',r.get_data(as_text=True));self.assertIn('no-store',r.headers['Cache-Control'])
