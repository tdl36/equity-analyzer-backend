"""Exact coordinated output reads, using disposable PostgreSQL only."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from flask import Flask, Blueprint
from research_assignment_meeting_read import register
from tests.unit import test_stock_analysis_postgres as pg

ID='674d1597-fd47-55fe-87b5-cdf7f12f254d'
RID='85b20347-e848-55d8-902b-5ea5324a36a0'

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class DeliveredMeeting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):
            c.execute('''CREATE TABLE mp_jobs(id UUID PRIMARY KEY,stage TEXT,ticker TEXT,result JSONB);
              CREATE TABLE mp_meetings(id INT PRIMARY KEY,company_id INT);
              CREATE TABLE mp_question_sets(id INT PRIMARY KEY,meeting_id INT,version INT,status TEXT,topics_json TEXT,created_at TIMESTAMP DEFAULT NOW(),generation_model TEXT);
              CREATE TABLE research_meeting_handoffs(run_id UUID,meeting_id INT,question_set_id INT);''')
    tearDownClass=classmethod(pg.StockPostgresTests.tearDownClass.__func__)
    def setUp(self):
        with self.db(True) as (_,c):
            c.execute('DELETE FROM mp_jobs; DELETE FROM mp_question_sets; DELETE FROM mp_meetings; DELETE FROM mp_companies; DELETE FROM research_meeting_handoffs')
            c.execute("INSERT INTO mp_companies(id,ticker) VALUES(1,'MDT'),(2,'ABT'); INSERT INTO mp_meetings VALUES(24,1),(26,2)")
            c.execute("INSERT INTO mp_jobs VALUES(%s,'research_assignment','MDT',%s::jsonb)",(ID,json.dumps({'reportId':RID,'meetingId':24,'meetingQuestionSetId':30})))
            c.execute('INSERT INTO research_meeting_handoffs VALUES(%s,24,30)',(RID,))
            for qid,mid,version in [(30,24,1),(31,24,2),(32,26,1)]:
                c.execute("INSERT INTO mp_question_sets VALUES(%s,%s,%s,'ready',%s,NOW(),'fixture')",(qid,mid,version,json.dumps([{'topic':'Synthetic','questions':[{'question':f'Question v{version}','research_origin':{'runId':RID},'supporting_quotes':[{'quote':'Synthetic evidence','filename':'fixture.txt'}]}]}])))
        bp=Blueprint('meeting_read',__name__);self.co=SimpleNamespace(bp=bp,db=self.db,wake=Mock(),invoke=Mock())
        register(self.co);app=Flask(__name__);app.testing=True;app.register_blueprint(bp);self.client=app.test_client()
    def read(self,ident=ID):return self.client.get(f'/api/research/assignments/{ident}/saved-meeting')
    def test_delivered_version_not_latest_and_no_writes_or_dispatch(self):
        with self.db() as (_,c):c.execute('SELECT result FROM mp_jobs WHERE id=%s',(ID,));before=c.fetchone()['result']
        for _ in range(2):
            r=self.read();self.assertEqual(r.status_code,200,r.json);q=r.json['questionSet']
            self.assertEqual((q['id'],q['version'],q['meeting_id'],q['ticker']),(30,1,24,'MDT'))
            self.assertEqual(r.json['reportId'],RID);self.assertEqual(r.headers['Cache-Control'],'private, no-store')
            self.assertEqual(q['topics'][0]['questions'][0]['supporting_quotes'][0]['quote'],'Synthetic evidence')
        self.co.wake.assert_not_called();self.co.invoke.assert_not_called()
        with self.db() as (_,c):
            c.execute('SELECT result FROM mp_jobs WHERE id=%s',(ID,));self.assertEqual(c.fetchone()['result'],before)
            c.execute('SELECT COUNT(*) AS n FROM mp_question_sets');self.assertEqual(c.fetchone()['n'],3)
    def test_unlinked_or_different_company_receipts_do_not_substitute(self):
        for patch in ({'meetingQuestionSetId':31},{'meetingQuestionSetId':32,'meetingId':26},{'reportId':ID},{'meetingId':26},{'meetingQuestionSetId':None}):
            with self.subTest(patch=patch):
                with self.db(True) as (_,c):c.execute('UPDATE mp_jobs SET result=%s::jsonb WHERE id=%s',(json.dumps({'reportId':RID,'meetingId':24,'meetingQuestionSetId':30,**patch}),ID))
                self.assertEqual(self.read().status_code,409)
    def test_missing_and_malformed_saved_version_remain_explicit(self):
        for raw in ('not-json','{}','null','[null]','[{"questions":[null]}]'):
            with self.db(True) as (_,c):c.execute('UPDATE mp_question_sets SET topics_json=%s WHERE id=30',(raw,))
            self.assertEqual(self.read().status_code,409)
        with self.db(True) as (_,c):c.execute('DELETE FROM mp_question_sets WHERE id=30')
        self.assertEqual(self.read().status_code,409)
        self.assertEqual(self.read(RID).status_code,404);self.assertEqual(self.read('invalid').status_code,400)
