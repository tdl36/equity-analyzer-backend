import unittest
import uuid
from tests.unit import test_pipeline_lab_note_jobs as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_worker import process_one
from pipeline_lab_note_jobs import status
from pipeline_lab_notes import append


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Worker(unittest.TestCase):
    setUpClass=classmethod(fixtures.Jobs.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.Jobs.tearDownClass.__func__)
    setUp=fixtures.Jobs.setUp
    def test_claim_commit_and_draft_receipt_are_settled(self):
        def executor(db,note,originals,provider,planner):
            with db() as (_,c):self.assertEqual(status(c,note)['status'],'running')
            self.assertEqual(originals({}),{'fixture':True})
            with db(True) as (_,c):append(c,note,'ABT',str(uuid.uuid4()),0,self.payload)
            return {'status':'draft','noteId':note,'revision':1}
        result=process_one(self.db,lambda plan:{'fixture':True},lambda **kw:self.fail('No provider needed'),executor=executor)
        self.assertEqual(result['status'],'draft')
        self.assertIsNone(process_one(self.db,None,None,executor=executor))
    def test_expired_lease_blocks_next_provider_call(self):
        calls=[]
        def executor(db,note,originals,provider,planner):
            with db(True) as (_,c):c.execute("UPDATE pipeline_lab_note_jobs SET lease_until=NOW()-INTERVAL '1 second' WHERE note_id=%s",(note,))
            provider(messages=[])
        result=process_one(self.db,lambda p:{},lambda **kw:calls.append(True),executor=executor)
        self.assertEqual(result['reason'],'worker_interrupted');self.assertEqual(calls,[])
        # Next polling iteration reaps the expired claim, not redispatches it.
        self.assertIsNone(process_one(self.db,None,None,executor=executor))
        with self.db() as (_,c):self.assertEqual(status(c,self.note)['status'],'attention')
    def test_source_failure_settles_attention_without_private_error(self):
        def executor(db,note,originals,provider,planner):raise ValueError('Private source details')
        result=process_one(self.db,None,None,executor=executor)
        self.assertEqual(result['reason'],'source_check_failed')
        self.assertNotIn('Private',str(result))
    def test_uncertain_result_is_preserved_and_not_requeued(self):
        def executor(*args):return {'status':'attention','reason':'provider_outcome_unknown'}
        result=process_one(self.db,None,None,executor=executor)
        self.assertEqual(result['reason'],'provider_outcome_unknown')
        self.assertIsNone(process_one(self.db,None,None,executor=executor))
