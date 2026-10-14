import unittest
import uuid
from flask import Flask,request
from tests.unit import test_pipeline_lab_note_jobs as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_submit import schema,create_blueprint


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Submission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Jobs.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):schema(c)
    tearDownClass=classmethod(fixtures.Jobs.tearDownClass.__func__)
    def setUp(self):
        fixtures.Jobs.setUp(self)
        self.calls=[];self.note=str(uuid.uuid4())
        def builder(cur,data):
            self.calls.append(data)
            return {**self.plan,'noteId':data['requestId'],'ticker':data['ticker'],'allowPdfText':data['allowPdfText']}
        app=Flask('lab-note-submit')
        app.register_blueprint(create_blueprint(self.db,lambda:request.headers.get('X-Test-Identity')=='yes',builder))
        self.client=app.test_client();self.headers={'X-Test-Identity':'yes'}
        self.body={'requestId':self.note,'revision':1,'digest':'a'*64,'allowPdfText':False}
        self.url='/api/pipeline-lab/note-jobs/ABT'
    def test_authenticated_submission_replays_without_rebuilding_context(self):
        self.assertEqual(self.client.post(self.url,json=self.body).status_code,401)
        for _ in range(2):
            response=self.client.post(self.url,headers=self.headers,json=self.body)
            self.assertEqual(response.status_code,200,response.json)
            self.assertEqual(response.json['status'],'queued')
        self.assertEqual(len(self.calls),1)
        status=self.client.get('/api/pipeline-lab/note-job/'+self.note,headers=self.headers)
        self.assertEqual(status.json['noteId'],self.note);self.assertIn('no-store',status.headers['Cache-Control'])
    def test_request_identity_conflict_and_client_prompt_rejected(self):
        self.client.post(self.url,headers=self.headers,json=self.body)
        self.assertEqual(self.client.post(self.url,headers=self.headers,json={**self.body,'revision':2}).status_code,409)
        for field in ('promptTemplate','apiKey','sourcePack','modelKey'):
            self.assertEqual(self.client.post(self.url,headers=self.headers,json={**self.body,field:'not accepted'}).status_code,409)
        self.assertEqual(len(self.calls),1)
    def test_bounds_and_recovery_auth(self):
        self.assertEqual(self.client.post(self.url,headers=self.headers,json={'x':'a'*5000}).status_code,409)
        path='/api/pipeline-lab/note-job/'+self.note+'/recover'
        self.assertEqual(self.client.post(path,json={}).status_code,401)
        self.assertEqual(self.client.post(path,headers=self.headers,json={'approve':True}).status_code,409)
        self.assertEqual(self.calls,[])
