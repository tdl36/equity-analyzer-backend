import json
import unittest
from flask import Flask
from meeting_workspace import create_blueprint
from tests.unit import test_stock_analysis_postgres as pg

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class MeetingContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):c.execute('''CREATE TABLE mp_question_sets(id INT PRIMARY KEY,meeting_id INT,version INT,topics_json TEXT,generation_model TEXT,created_at TIMESTAMP DEFAULT NOW());
            CREATE TABLE mp_documents(meeting_id INT,filename TEXT);
            CREATE TABLE mp_past_questions(id INT,meeting_id INT,question TEXT,response_notes TEXT,status TEXT);''')
    tearDownClass=classmethod(pg.StockPostgresTests.tearDownClass.__func__)
    def setUp(self):
        with self.db(True) as (_,c):
            c.execute('DELETE FROM mp_question_sets; DELETE FROM mp_documents; DELETE FROM mp_past_questions')
            for id,mid,version,count in [(1,24,1,1),(2,24,2,2),(3,25,1,3)]:
                c.execute('INSERT INTO mp_question_sets(id,meeting_id,version,topics_json) VALUES(%s,%s,%s,%s)',(id,mid,version,json.dumps([{'topic':'Synthetic','questions':[{'question':f'Question {n}?','priority':'high'} for n in range(count)]}])))
            c.execute("INSERT INTO mp_past_questions VALUES(1,24,'Question?',NULL,'planned')")
        app=Flask('context');app.testing=True;app.register_blueprint(create_blueprint(self.db));self.client=app.test_client()
    def test_quality_uses_exact_requested_version_not_latest(self):
        r=self.client.get('/api/mp/meetings/24/quality?questionSetId=1');self.assertEqual(r.status_code,200)
        self.assertEqual((r.json['meetingId'],r.json['questionSetId'],r.json['questions']),(24,1,1))
        latest=self.client.get('/api/mp/meetings/24/quality').json;self.assertEqual((latest['questionSetId'],latest['questions']),(2,2))
        self.assertEqual(self.client.get('/api/mp/meetings/24/quality?questionSetId=3').status_code,404)
        self.assertEqual(self.client.get('/api/mp/meetings/24/quality?questionSetId=99').status_code,404)
        for value in ('0','-1','abc','2147483648'):
            self.assertEqual(self.client.get('/api/mp/meetings/24/quality?questionSetId='+value).status_code,400)
    def test_histories_and_answers_disclose_scope_and_private_cache(self):
        for part in ('versions','answers'):
            r=self.client.get('/api/mp/meetings/24/'+part);self.assertEqual(r.json['meetingId'],24);self.assertEqual(r.headers['Cache-Control'],'private, no-store')
            self.assertTrue(all(x['meeting_id']==24 for x in r.json[part]))
    def test_history_truncation_is_visible_and_does_not_delete_old_versions(self):
        with self.db(True) as (_,c):c.execute("INSERT INTO mp_question_sets(id,meeting_id,version,topics_json) SELECT n,24,n,'[]' FROM generate_series(4,40) n")
        r=self.client.get('/api/mp/meetings/24/versions');self.assertTrue(r.json['truncated']);self.assertEqual(len(r.json['versions']),30)
        self.assertEqual(r.json['versions'][0]['id'],40)
        self.assertEqual(self.client.get('/api/mp/meetings/24/quality?questionSetId=1').status_code,200)
