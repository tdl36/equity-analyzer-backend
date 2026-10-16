"""Synthetic recorded-answer writes in an isolated disposable database only."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from flask import Flask,jsonify,request
from meeting_answer_capture import save
from meeting_workspace import create_blueprint
from tests.unit import test_stock_analysis_postgres as pg

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class AnswerCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):c.execute('''CREATE TABLE mp_meetings(id INT PRIMARY KEY,company_id INT);
            CREATE TABLE mp_past_questions(id INT PRIMARY KEY,meeting_id INT,company_id INT,question TEXT,response_notes TEXT,status TEXT);
            INSERT INTO mp_meetings VALUES(24,1),(25,2);''')
    tearDownClass=classmethod(pg.StockPostgresTests.tearDownClass.__func__)
    def setUp(self):
        with self.db(True) as (_,c):c.execute("DELETE FROM mp_past_questions; INSERT INTO mp_past_questions VALUES(1,24,1,'Synthetic question?',NULL,'planned')")
        self.payload={'meetingId':24,'question':'Synthetic question?','responseNotes':'Synthetic recorded answer','status':'answered','expected':{'responseNotes':None,'status':'planned'}}
        app=Flask('answer-test');app.testing=True;app.register_blueprint(create_blueprint(self.db));self.client=app.test_client()
    def test_exact_read_and_save_receipt_with_safe_replay(self):
        r=self.client.get('/api/mp/meetings/24/answers/1');self.assertEqual(r.status_code,200);self.assertEqual(r.json['answer']['meeting_id'],24);self.assertIn('no-store',r.headers['Cache-Control'])
        self.assertEqual(self.client.get('/api/mp/meetings/25/answers/1').status_code,404)
        body,status=save(self.db,1,self.payload);self.assertEqual(status,200);self.assertFalse(body['replayed']);self.assertEqual(body['answer']['response_notes'],self.payload['responseNotes'])
        body,status=save(self.db,1,self.payload);self.assertEqual(status,200);self.assertTrue(body['replayed'])
    def test_missing_and_wrong_identity_never_claim_success(self):
        self.assertEqual(save(self.db,999,self.payload)[1],404)
        for extra in ({'meetingId':25},{'question':'Different question?'}):self.assertEqual(save(self.db,1,{**self.payload,**extra})[1],404)
        with self.db(True) as (_,c):c.execute('UPDATE mp_past_questions SET company_id=2 WHERE id=1')
        self.assertEqual(save(self.db,1,self.payload)[1],404)
    def test_competing_edits_only_one_wins_and_reviewed_baseline_can_save(self):
        p2={**self.payload,'responseNotes':'Other recorded answer'}
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda p:save(self.db,1,p),[self.payload,p2]))
        self.assertEqual(sorted(status for _,status in results),[200,409])
        conflict=next(body for body,status in results if status==409);row=conflict['answer']
        desired={**self.payload,'responseNotes':'Reviewed merged answer','expected':{'responseNotes':row['response_notes'],'status':row['status']}}
        self.assertEqual(save(self.db,1,desired)[1],200)
        self.assertEqual(save(self.db,1,self.payload)[1],409)
    def test_legacy_unbounded_and_invalid_edits_are_refused(self):
        for p in (None,{}, {'responseNotes':'No baseline'}, {**self.payload,'responseNotes':''},{**self.payload,'responseNotes':'x'*10001},{**self.payload,'status':'planned'},{**self.payload,'expected':{}},{**self.payload,'meetingId':True}):
            self.assertEqual(save(self.db,1,p)[1],400)
        r=self.client.get('/api/mp/meetings/24/answers/1');self.assertIsNone(r.json['answer']['response_notes'])
