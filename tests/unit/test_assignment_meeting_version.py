"""Read-only delivery checks against disposable PostgreSQL, no model calls."""
import json
import unittest
from flask import Flask, Blueprint
from assignment_workspace import routes
from tests.unit import test_stock_analysis_postgres as pg

CID='674d1597-fd47-55fe-87b5-cdf7f12f254d'
OTHER='85b20347-e848-55d8-902b-5ea5324a36a0'

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class SavedMeetingVersion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):
            c.execute('''CREATE TABLE mp_jobs(id UUID PRIMARY KEY,stage TEXT,ticker TEXT,input JSONB,status TEXT DEFAULT 'done',error TEXT,result JSONB DEFAULT '{}',created_at TIMESTAMP DEFAULT NOW(),updated_at TIMESTAMP DEFAULT NOW());
                CREATE TABLE app_settings(key TEXT,value JSONB);
                CREATE TABLE mp_documents(meeting_id INT,filename TEXT,doc_type TEXT,upload_order INT);
                CREATE TABLE mp_meetings(id INT PRIMARY KEY,company_id INT);
                CREATE TABLE mp_question_sets(id INT PRIMARY KEY,meeting_id INT,version INT,status TEXT,topics_json TEXT,created_at TIMESTAMP DEFAULT NOW(),generation_model TEXT);''')
    tearDownClass=classmethod(pg.StockPostgresTests.tearDownClass.__func__)
    def setUp(self):
        with self.db(True) as (_,c):
            c.execute('DELETE FROM mp_documents; DELETE FROM mp_jobs; DELETE FROM mp_question_sets; DELETE FROM mp_meetings; DELETE FROM mp_companies')
            c.execute("INSERT INTO mp_companies(id,ticker) VALUES(1,'MDT'),(2,'ABT'); INSERT INTO mp_meetings VALUES(24,1),(26,2)")
            c.execute("INSERT INTO mp_jobs(id,stage,ticker,input) VALUES(%s,'collection_control','MDT','{}'),(%s,'collection_control','ABT','{}'),('00000000-0000-4000-8000-000000000001','command_meeting','MDT',%s::jsonb)",(CID,OTHER,json.dumps({'commandId':CID,'meetingId':24})))
            c.execute("UPDATE mp_jobs SET input=%s::jsonb,result=%s::jsonb WHERE id=%s",(json.dumps({'payload':{'meetingPrep':{'duration':60}}}),json.dumps({'savedSources':True}),CID))
            c.execute("INSERT INTO mp_documents VALUES(24,'MDT fixture.pdf','broker-report',1),(26,'ABT fixture.pdf','broker-report',1)")
            for id,meeting,version in [(30,24,1),(31,24,2),(32,26,1)]:
                c.execute("INSERT INTO mp_question_sets(id,meeting_id,version,status,topics_json,generation_model) VALUES(%s,%s,%s,'ready',%s,'fixture')",(id,meeting,version,json.dumps([{'topic':'Synthetic','questions':[{'question':f'Question v{version}','supporting_quotes':[{'quote':'Synthetic passage','filename':'fixture.pdf'}]}]}])))
        app=Flask('saved-version');app.testing=True;bp=Blueprint('assignments',__name__);routes(bp,self.db);app.register_blueprint(bp);self.client=app.test_client()
    def read(self,id=30,cid=CID):
        return self.client.get(f'/api/research/assignments/{cid}/meeting-versions/{id}')
    def test_exact_older_version_retains_evidence_with_newer_present(self):
        r=self.read();self.assertEqual(r.status_code,200)
        self.assertEqual(r.json['assignmentId'],CID)
        q=r.json['questionSet'];self.assertEqual((q['id'],q['version'],q['ticker'],q['meeting_id']),(30,1,'MDT',24))
        self.assertEqual(q['topics'][0]['questions'][0]['supporting_quotes'][0]['quote'],'Synthetic passage')
        self.assertEqual(r.headers['Cache-Control'],'private, no-store')
        self.assertNotIn('topics_json',q)
        with self.db() as (_,c):c.execute('SELECT COUNT(*) AS n FROM mp_question_sets');self.assertEqual(c.fetchone()['n'],3)
    def test_wrong_assignment_meeting_and_missing_never_fall_back(self):
        for cid,id in [(OTHER,30),(CID,32),(CID,999)]:
            r=self.read(id,cid);self.assertEqual(r.status_code,404);self.assertIn('no-store',r.headers['Cache-Control'])
        self.assertEqual(self.read(cid='invalid').status_code,400)
    def test_mismatched_company_and_unlinked_command_refused(self):
        with self.db(True) as (_,c):c.execute("UPDATE mp_jobs SET ticker='ABT' WHERE id=%s",(CID,))
        self.assertEqual(self.read().status_code,404)
    def test_corrupt_content_does_not_substitute_newer(self):
        for raw in ('not-json','{}','null'):
            with self.db(True) as (_,c):c.execute('UPDATE mp_question_sets SET topics_json=%s WHERE id=30',(raw,))
            self.assertEqual(self.read().status_code,409)
            self.assertEqual(self.read(31).json['questionSet']['version'],2)

    def detail(self):
        return self.client.get(f'/api/research/assignments/{CID}')
    def test_detail_projects_only_verified_meeting_and_keeps_saved_status(self):
        r=self.detail();self.assertEqual(r.status_code,200);d=r.json
        self.assertEqual(d['command']['id'],CID);self.assertEqual(d['meetingId'],24)
        self.assertIsNone(d['meetingIssue']);self.assertEqual([x['id'] for x in d['versions']],[31,30])
        self.assertEqual([x['filename'] for x in d['documents']],['MDT fixture.pdf'])
        self.assertEqual(d['delivery']['state'],'delivered');self.assertEqual(r.headers['Cache-Control'],'private, no-store')
    def test_bad_meeting_references_retain_command_but_withhold_other_sources(self):
        for value in (26,999,None,True,'bad','024',2147483648):
            with self.subTest(value=value):
                with self.db(True) as (_,c):c.execute("UPDATE mp_jobs SET input=%s::jsonb WHERE stage='command_meeting'",(json.dumps({'commandId':CID,'meetingId':value}),))
                d=self.detail().json
                self.assertEqual(d['command']['id'],CID);self.assertIsNone(d['meetingId']);self.assertTrue(d['meetingIssue'])
                self.assertEqual(d['documents'],[]);self.assertEqual(d['versions'],[]);self.assertEqual(d['delivery']['state'],'attention')
                self.assertEqual(d['preparation']['status'],'done')
                self.assertNotIn('ABT fixture',json.dumps(d))
    def test_wrong_prep_company_refuses_both_detail_and_exact_version(self):
        with self.db(True) as (_,c):c.execute("UPDATE mp_jobs SET ticker='ABT' WHERE stage='command_meeting'")
        d=self.detail().json;self.assertIsNone(d['meetingId']);self.assertTrue(d['meetingIssue']);self.assertEqual(d['documents'],[])
        self.assertEqual(self.read().status_code,404)
    def test_ambiguous_preparations_are_not_arbitrarily_selected_or_mutated(self):
        with self.db(True) as (_,c):c.execute("INSERT INTO mp_jobs(id,stage,ticker,input) VALUES('00000000-0000-4000-8000-000000000002','command_meeting','MDT',%s::jsonb)",(json.dumps({'commandId':CID,'meetingId':24}),))
        d=self.detail().json;self.assertIn('Multiple preparation',d['meetingIssue'])
        self.assertIsNone(d['meetingId']);self.assertIsNone(d['preparation']);self.assertEqual(d['versions'],[])
        self.assertEqual(d['delivery']['state'],'attention');self.assertEqual(self.read().status_code,404)
        with self.db() as (_,c):
            c.execute("SELECT status FROM mp_jobs WHERE stage='command_meeting'");self.assertEqual([r['status'] for r in c.fetchall()],['done','done'])
