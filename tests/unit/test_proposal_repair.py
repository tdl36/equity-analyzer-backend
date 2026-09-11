import copy
import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from flask import Flask
from proposal_repair import merge_repair
from research_amendments import create_blueprint

class RepairTests(unittest.TestCase):
    def setUp(self):
        self.changes=[{'id':'0','path':'assumptions.a.support','after':'Good','passageMatched':True,'reviewPassed':True},
                      {'id':'7','path':'assumptions.b.support','after':'Bad','passageMatched':False,'reviewPassed':False}]
    def test_preserves_siblings_and_ids(self):
        before=copy.deepcopy(self.changes)
        revised={**self.changes[1],'id':'0','after':'Supported','passageMatched':True,'reviewPassed':True}
        out=merge_repair(self.changes,[revised],[self.changes[1]])
        self.assertEqual(out[0],before[0]);self.assertEqual(out[1]['id'],'7')
        self.assertEqual(out[1]['repairOutcome'],'revised');self.assertEqual(self.changes,before)
    def test_no_evidence_keeps_draft_blocked(self):
        c=merge_repair(self.changes,[],[self.changes[1]])[1]
        self.assertFalse(c['passageMatched']);self.assertFalse(c['reviewPassed'])
        self.assertEqual(c['repairOutcome'],'no_supported_change')
    def test_failed_review_never_promoted(self):
        c=merge_repair(self.changes,[{**self.changes[1],'passageMatched':True}],[self.changes[1]])[1]
        self.assertFalse(c['reviewPassed']);self.assertEqual(c['repairOutcome'],'still_unsupported')
    def test_repair_cannot_rewrite_ready_sibling(self):
        with self.assertRaises(ValueError):merge_repair(self.changes,[self.changes[0]],[self.changes[1]])

class RepairRouteTests(RepairTests):
    def setUp(self):
        super().setUp()
        self.job={'ticker':'ABT','status':'awaiting_approval','input':{'target':'investment_case','baseline':{},'filenames':['source.pdf'],'sourceHashes':{'source.pdf':'hash'}},'result':{'changes':self.changes,'checkpoint':{'draft':{'changes':[]}}}}
        job=self.job
        class Cursor:
            def execute(self,sql,args):
                self.sql=sql
                if sql.startswith("UPDATE mp_jobs SET status='queued'"):
                    job.update(status='queued',input=json.loads(args[0]),result=json.loads(args[1]))
                if sql.startswith("UPDATE mp_jobs SET status='running'"):job['status']='running'
                if sql.startswith("UPDATE mp_jobs SET status=%s"):
                    job.update(status=args[0],result={**job['result'],**json.loads(args[1])},error=args[2])
            def fetchall(self):return [{'filename':'source.pdf','file_data':'dummy','file_type':'pdf'}]
            def fetchone(self):return {'locked':True} if 'pg_try' in self.sql else copy.deepcopy(job)
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        app=Flask(__name__);app.config['TESTING']=True
        app.register_blueprint(create_blueprint(db,lambda *a:None,lambda k:'test-key'))
        self.client=app.test_client()
        self.worker=patch('research_amendments.threading.Thread');self.thread=self.worker.start();self.addCleanup(self.worker.stop)
        self.current=patch('research_amendments.baseline_current',return_value=True);self.check=self.current.start();self.addCleanup(self.current.stop)
    def send(self,attempt=0):return self.client.post('/api/research/amendment/job/repair',json={'attempt':attempt})
    def test_queues_and_archives_without_replacing_changes(self):
        before=copy.deepcopy(self.job['result'])
        self.assertEqual(self.send().status_code,202)
        self.assertEqual(self.job['result']['changes'],before['changes'])
        self.assertNotIn('checkpoint',self.job['result'])
        self.assertEqual(self.job['result']['repairHistory'][0]['checkpoint'],before['checkpoint'])
        self.assertEqual(self.job['input']['repair']['targets'],[self.changes[1]])
        self.assertEqual(self.send().status_code,200);self.assertEqual(self.thread.call_count,1)
    def test_worker_repair_reextracts_reviews_and_merges(self):
        from research_evidence import source_catalog
        text='The company reported operating margin of twenty percent in fiscal 2025.'
        self.job['input']['baseline']={'_investmentCase':{'revision':1,'body':{'assumptions':[{'id':'a','support':'Old a'},{'id':'b','support':'Old b'}]}}}
        self.assertEqual(self.send().status_code,202)
        target=self.thread.call_args.kwargs['target'];args=self.thread.call_args.kwargs['args']
        source=source_catalog([{'filename':'source.pdf','file_data':'dummy','extracted_text':text}])[0]
        draft={'changes':[{'path':'assumptions.b.support','after':text,'reason':'Margin update.','source_id':source['id'],'source_excerpt':text}]}
        review={'checks':[{'id':'0','verdict':'pass','issue':''}]}
        @contextmanager
        def ownership(*args):yield True
        # Model and DB are synthetic; execute the actual worker and verification pipeline.
        with patch('amendment_ownership.worker_session',ownership),patch('notegen.extract_pdf_text',return_value=text),patch('command_thesis_bridge.file_hash',return_value='hash'),patch('amendment_checkpoints.Checkpoints') as checkpoints:
            checkpoints.return_value.load.side_effect=[draft,review]
            target(*args)
        self.assertEqual(self.job['status'],'awaiting_approval',self.job.get('error'))
        self.assertEqual(self.job['result']['changes'][0],self.changes[0])
        c=self.job['result']['changes'][1]
        self.assertEqual(c['id'],'7');self.assertTrue(c['passageMatched']);self.assertTrue(c['reviewPassed'])
        self.assertEqual(c['after'],text)

    def test_stale_case_rejected_without_mutation(self):
        self.check.return_value=False;before=copy.deepcopy(self.job)
        self.assertEqual(self.send().status_code,409);self.assertEqual(self.job,before);self.thread.assert_not_called()
    def test_closed_case_rejected(self):
        self.job['status']='dismissed';self.assertEqual(self.send().status_code,409)
    def test_old_attempt_rejected(self):
        self.job['result']['repairHistory']=[{}]
        self.assertEqual(self.send().status_code,409);self.thread.assert_not_called()
    def test_bound_repeated_unsupported_repairs(self):
        self.job['result']['repairHistory']=[{},{}]
        self.assertEqual(self.send(2).status_code,409);self.thread.assert_not_called()

if __name__=='__main__':unittest.main()
