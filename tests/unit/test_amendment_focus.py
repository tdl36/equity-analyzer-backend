"""Read-only deep links must never select a different company's/latest proposal."""
import json
import sqlite3
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch
from flask import Flask
from research_amendments import create_blueprint

JOB='11111111-1111-4111-8111-111111111111'
class FocusTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:');self.db.row_factory=sqlite3.Row
        self.db.execute('CREATE TABLE mp_jobs(id TEXT,ticker TEXT,stage TEXT,status TEXT,result TEXT,error TEXT,created_at TEXT,updated_at TEXT,input TEXT)')
        for i in range(12):self.add(f'new-{i}','ABT',f'2026-10-{i+1:02}',{})
        self.add(JOB,'ABT','2025-01-01',{})
        parent=self
        class Cursor:
            def execute(self,sql,args):
                assert sql.startswith('SELECT ')
                self.cursor=parent.db.execute(sql.replace('%s','?'),args)
            def fetchall(self):return self.cursor.fetchall()
            def fetchone(self):return self.cursor.fetchone()
        @contextmanager
        def db(commit=False):
            self.assertFalse(commit)
            yield None,Cursor()
        self.model=Mock();self.key=Mock()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,self.model,self.key))
        self.client=app.test_client()
    def tearDown(self):self.db.close()
    def add(self,id,ticker,date,body,stage='evidence_amendment'):
        self.db.execute('INSERT INTO mp_jobs VALUES(?,?,?,?,?,?,?,?,?)',(id,ticker,stage,'awaiting_approval','{}',None,date,date,json.dumps(body)))
    def read(self,ticker='ABT',identity=JOB):
        with patch('research_amendments.threading.Thread') as thread:
            r=self.client.get(f'/api/research/amendments/{ticker}?requestId={identity}')
            thread.assert_not_called();self.model.assert_not_called();self.key.assert_not_called()
        return r
    def test_old_proposal_is_read_beyond_recent_ten(self):
        recent=self.client.get('/api/research/amendments/ABT').json['jobs']
        self.assertEqual(len(recent),10);self.assertNotIn(JOB,[r['id'] for r in recent])
        r=self.read();self.assertEqual(r.status_code,200)
        self.assertEqual([(j['id'],j['ticker']) for j in r.json['jobs']],[(JOB,'ABT')])
        self.assertEqual(r.headers['Cache-Control'],'no-store')
    def test_missing_wrong_company_and_other_targets_never_fall_back(self):
        self.assertEqual(self.read('CVS').json['jobs'],[])
        for body,stage in [({'target':'investment_case'},'evidence_amendment'),({},'research_assignment')]:
            self.db.execute('UPDATE mp_jobs SET input=?,stage=? WHERE id=?',(json.dumps(body),stage,JOB))
            self.assertEqual(self.read().json['jobs'],[])
        self.assertEqual(self.read(identity='22222222-2222-4222-8222-222222222222').json['jobs'],[])
    def test_compact_saved_identity_is_not_rewritten(self):
        compact=JOB.replace('-','');self.add(compact,'ABT','2024-01-01',{})
        self.assertEqual(self.read(identity=compact).json['jobs'][0]['id'],compact)
    def test_malformed_identity_rejected(self):
        for value in ['','bad','%27%20OR%201=1']:
            self.assertEqual(self.read(identity=value).status_code,400)

    def test_followup_lookup_finds_older_links_and_separate_active_company_job(self):
        work='33333333-3333-4333-8333-333333333333'
        self.db.execute("UPDATE mp_jobs SET status='failed'")
        self.db.execute("UPDATE mp_jobs SET status='awaiting_approval' WHERE id=?",(JOB,))
        for id,ticker,body,stage in [
            ('old-linked','ABT',{'workEvidence':{'id':work,'revision':1}},'evidence_amendment'),
            ('other-company','CVS',{'workEvidence':{'id':work,'revision':1}},'evidence_amendment'),
            ('other-target','ABT',{'target':'investment_case','workEvidence':{'id':work,'revision':1}},'evidence_amendment'),
            ('other-stage','ABT',{'workEvidence':{'id':work,'revision':1}},'research_assignment')]:
            self.add(id,ticker,'2020-01-01',body,stage)
        r=self.client.get('/api/research/amendments/ABT?workId='+work)
        self.assertEqual(r.status_code,200);d=r.json
        self.assertEqual([j['id'] for j in d['jobs']],['old-linked'])
        self.assertEqual(d['workId'],work);self.assertFalse(d['truncated'])
        self.assertEqual(d['activeProposal']['id'],JOB)
        self.assertEqual(r.headers['Cache-Control'],'no-store')
        self.model.assert_not_called();self.key.assert_not_called()
    def test_followup_history_is_bounded_with_explicit_truncation(self):
        work='33333333-3333-4333-8333-333333333333'
        for i in range(31):self.add(f'linked-{i:02}','ABT','2020-01-01',{'workEvidence':{'id':work,'revision':i+1}})
        d=self.client.get('/api/research/amendments/ABT?workId='+work).json
        self.assertEqual(len(d['jobs']),30);self.assertTrue(d['truncated'])
        d=self.client.get('/api/research/amendments/ABT?workId='+JOB).json
        self.assertEqual(d['jobs'],[]);self.assertFalse(d['truncated'])
    def test_ambiguous_and_malformed_followup_filters_are_rejected(self):
        for query in ['workId=','workId=bad','workId='+JOB+'&requestId='+JOB]:
            self.assertEqual(self.client.get('/api/research/amendments/ABT?'+query).status_code,400)
