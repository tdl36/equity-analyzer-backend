import unittest
import uuid
from tests.unit import test_pipeline_lab_note_plan as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_jobs import schema,enqueue,claim,renew,settle,status
from pipeline_lab_notes import append


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Jobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Plans.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):schema(c)
    tearDownClass=classmethod(fixtures.Plans.tearDownClass.__func__)
    def setUp(self):
        fixtures.Plans.setUp(self)
        with self.db(True) as (_,c):
            c.execute('DELETE FROM pipeline_lab_note_jobs')
            enqueue(c,self.plan)
    def test_enqueue_replay_does_not_reset_running_or_attention(self):
        with self.db(True) as (_,c):
            lease=claim(c);self.assertIsNotNone(lease)
            self.assertEqual(enqueue(c,self.plan)['status'],'running')
            renew(c,self.note,lease['leaseId'])
            settle(c,self.note,lease['leaseId'],{'status':'attention','reason':'provider_outcome_unknown'})
            self.assertEqual(enqueue(c,self.plan)['status'],'attention')
            self.assertIsNone(claim(c))
    def test_expired_lease_requires_attention_and_old_worker_cannot_write(self):
        with self.db(True) as (_,c):
            lease=claim(c)
            c.execute("UPDATE pipeline_lab_note_jobs SET lease_until=NOW()-INTERVAL '1 second' WHERE note_id=%s",(self.note,))
            self.assertIsNone(claim(c))
            self.assertEqual(status(c,self.note)['reason'],'worker_interrupted')
        for operation in (lambda c:renew(c,self.note,lease['leaseId']),lambda c:settle(c,self.note,lease['leaseId'],{'status':'attention','reason':'source_check_failed'})):
            with self.assertRaises(ValueError),self.db(True) as (_,c):operation(c)
    def test_parallel_claims_have_one_owner(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        gate=Barrier(2)
        def take(_):
            gate.wait()
            with self.db(True) as (_,c):return claim(c)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(r is not None for r in pool.map(take,[1,2])),1)
    def test_saved_draft_required_and_lease_identity_fenced(self):
        with self.db(True) as (_,c):lease=claim(c)
        receipt={'status':'draft','noteId':self.note,'revision':1}
        with self.assertRaises(ValueError),self.db(True) as (_,c):settle(c,self.note,lease['leaseId'],receipt)
        with self.db(True) as (_,c):append(c,self.note,'ABT',str(uuid.uuid4()),0,self.payload)
        with self.assertRaises(ValueError),self.db(True) as (_,c):settle(c,self.note,str(uuid.uuid4()),receipt)
        with self.db(True) as (_,c):
            self.assertEqual(settle(c,self.note,lease['leaseId'],receipt)['status'],'draft')
            self.assertEqual(enqueue(c,self.plan)['status'],'draft')
