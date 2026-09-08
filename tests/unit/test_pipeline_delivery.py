import copy
import json
import unittest
from contextlib import contextmanager
from flask import Flask
from pipeline_delivery import create_blueprint, digest

JOB='11111111-1111-1111-1111-111111111111'
class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.job={'job_type':'synthesis','status':'failed','steps_detail':{'topic':'UNH filing','activityId':'activity'},'result':None}
        self.activity={'status':'failed','output':{'catalystJobId':JOB,'priorRuns':[{'synthesisMarkdown':'prior draft'}]}}
        self.result={'markdown':'Source-backed draft','topic':'UNH filing','fileCount':2}
        outer=self
        class Cursor:
            def execute(self,sql,args=()):
                self.sql=sql
                if sql.startswith('UPDATE research_pipeline_jobs'):
                    outer.job.update(status='complete',result=json.loads(args[0]))
                if sql.startswith('UPDATE analyst_activities'):
                    outer.activity.update(status='pending_review',output=json.loads(args[0]))
            def fetchone(self):return outer.activity if 'FROM analyst_activities' in self.sql else outer.job
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db));self.client=app.test_client()
    def send(self,result=None):return self.client.post('/api/pipeline/jobs/'+JOB+'/result',json={'status':'complete','result':result or self.result})
    def test_recovers_timed_out_delivery_and_keeps_prior_versions(self):
        r=self.send();self.assertEqual(r.status_code,200);self.assertEqual(r.json['resultHash'],digest(self.result))
        self.assertEqual(self.activity['status'],'pending_review');self.assertEqual(len(self.activity['output']['priorRuns']),1)
        self.assertEqual(self.job['status'],'complete')
    def test_same_receipt_is_idempotent(self):
        self.send();before=copy.deepcopy(self.activity);r=self.send()
        self.assertEqual(r.status_code,200);self.assertFalse(r.json['inboxLinked']);self.assertEqual(self.activity,before)
    def test_conflicting_receipt_cannot_replace_existing_result(self):
        self.send();self.assertEqual(self.send({**self.result,'markdown':'different'}).status_code,409)
        self.assertEqual(self.job['result'],self.result)
    def test_superseded_or_approved_inbox_is_not_overwritten(self):
        for change in ({'status':'approved'},{'output':{'catalystJobId':'new-job'}}):
            self.activity.update(change);before=copy.deepcopy(self.activity)
            self.assertEqual(self.send().status_code,200);self.assertEqual(self.activity,before)
    def test_wrong_topic_and_empty_draft_fail(self):
        for result in ({**self.result,'topic':'other'},{**self.result,'markdown':' '}):
            self.assertEqual(self.send(result).status_code,400)
        self.assertEqual(self.job['status'],'failed')
    def test_missing_and_cancelled_jobs_keep_local_receipt(self):
        self.job['status']='cancelled';self.assertEqual(self.send().status_code,409)
        self.job=None;self.assertEqual(self.send().status_code,404)
    def test_superseded_claim_cannot_deliver_late_output(self):
        self.job.update(agent_managed=True,agent_owner='current-owner')
        self.assertEqual(self.send().status_code,409)
        self.assertIsNone(self.job['result'])
    def test_actual_outbox_route_requires_matching_receipt(self):
        import tempfile
        from types import SimpleNamespace
        from catalyst_delivery import save_result,deliver
        def post(url,**kwargs):
            r=self.client.post(url.replace('https://test.invalid',''),json=kwargs['json'])
            return SimpleNamespace(ok=r.status_code==200,json=r.get_json)
        with tempfile.TemporaryDirectory() as tmp:
            path=save_result(JOB,self.result,tmp)
            self.assertTrue(deliver(path,'https://test.invalid',{},post=post))
            self.assertFalse(path.exists());self.assertEqual(self.activity['status'],'pending_review')
