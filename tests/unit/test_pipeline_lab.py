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
