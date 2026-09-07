import json
import unittest
import uuid
from contextlib import contextmanager
from flask import Flask
import collection_control as cc

class Commands:
    def __init__(self):self.jobs={};self.rows=[]
    @contextmanager
    def db(self,commit=False):yield None,self
    def execute(self,sql,args=()):
        self.rows=[]
        if 'pg_advisory' in sql:return
        if sql.startswith('SELECT stage,input,status'):
            row=self.jobs.get(args[0]);self.rows=[row] if row else []
        elif sql.startswith('INSERT INTO mp_jobs'):
            self.jobs[args[0]]={'stage':args[1],'input':json.loads(args[3]),'status':'queued'}
        else:raise AssertionError(sql)
    def fetchone(self):return self.rows[0] if self.rows else None

class ControlTests(unittest.TestCase):
    def setUp(self):
        self.db=Commands();app=Flask(__name__);app.testing=True;app.register_blueprint(cc.create_blueprint(self.db.db,lambda:False));self.client=app.test_client()
        self.data={'requestId':str(uuid.uuid4()),'action':'trigger','payload':{'ticker':'MDT'}}
    def test_idempotent_queue_and_conflicting_payload(self):
        self.assertEqual(self.client.post('/api/collection/control',json=self.data).status_code,202)
        self.assertEqual(self.client.post('/api/collection/control',json=self.data).status_code,200)
        self.assertEqual(len(self.db.jobs),1)
        self.assertEqual(self.client.post('/api/collection/control',json={**self.data,'payload':{'ticker':'DE'}}).status_code,409)
    def test_user_session_cannot_impersonate_agent(self):
        self.assertEqual(self.client.get('/api/agent/collection-control').status_code,403)
        self.assertEqual(self.client.post('/api/agent/collection-control',json={}).status_code,403)
    def test_reject_arbitrary_actions_and_invalid_tickers(self):
        for data in [{**self.data,'action':'shell'},{**self.data,'payload':{'ticker':'../MDT'}},{**self.data,'requestId':'bad'}]:
            with self.assertRaises(ValueError):cc.command(data)
    def test_policy_validation_and_metadata_removal(self):
        p={'ticker':'MDT','hours':24,'lookbackDays':7,'enabled':True,'workflow':'recap','topic':'MDT Earnings','kinds':['transcript'],'instructions':'Primary sources','lastSuccess':'not trusted'}
        value=cc.command({**self.data,'action':'save','payload':p});self.assertNotIn('lastSuccess',value['payload'])
        for key,val in [('hours',True),('topic','../bad'),('instructions','x'*3001),('kinds',[]),('enabled','yes')]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):cc.command({**self.data,'action':'save','payload':{**p,key:val}})
