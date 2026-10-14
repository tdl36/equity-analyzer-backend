import unittest
import uuid
from tests.unit import test_pipeline_lab_note_jobs as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_jobs import recover,claim,status
from pipeline_lab_note_calls import schema,reserve,finish
from pipeline_lab_notes import append


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Recovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Jobs.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):schema(c)
    tearDownClass=classmethod(fixtures.Jobs.tearDownClass.__func__)
    setUp=fixtures.Jobs.setUp
    def call(self,c,text=None):
        request={'noteId':self.note,'batch':1,'sourcePack':self.plan['sourcePack'],'contextSha256':'a'*64,
                 'promptSha256':'b'*64,'modelKey':'fixture','maxTokens':1000}
        intent=reserve(c,request)
        if text is not None:
            finish(c,self.note,1,intent['requestHash'],{'text':text,'provider':'fixture','model':'fixture','stopReason':'end_turn'})
    def test_live_worker_cannot_be_stolen_but_unstarted_expired_job_can_resume(self):
        with self.db(True) as (_,c):claim(c)
        with self.assertRaises(ValueError),self.db(True) as (_,c):recover(c,self.note)
        with self.db(True) as (_,c):
            c.execute("UPDATE pipeline_lab_note_jobs SET lease_until=NOW()-INTERVAL '1 second' WHERE note_id=%s",(self.note,))
            self.assertEqual(recover(c,self.note)['status'],'queued')
    def test_unknown_outcome_never_requeues(self):
        with self.db(True) as (_,c):
            self.call(c)
            result=recover(c,self.note)
            self.assertEqual(result['reason'],'provider_outcome_unknown')
            self.assertEqual(result['status'],'attention')
            self.assertEqual(recover(c,self.note)['status'],'attention')
            self.assertIsNone(claim(c))
    def test_saved_final_reply_recovers_draft_without_provider(self):
        with self.db(True) as (_,c):
            self.call(c,'===NOTE_START===Synthetic draft [1].===NOTE_END===')
            result=recover(c,self.note)
            self.assertEqual(result['status'],'draft');self.assertEqual(result['revision'],1)
            self.assertEqual(recover(c,self.note)['revision'],1)
    def test_malformed_reply_stays_attention_and_existing_draft_is_reconciled(self):
        with self.db(True) as (_,c):
            self.call(c,'Malformed saved response')
            self.assertEqual(recover(c,self.note)['reason'],'saved_reply_needs_review')
            append(c,self.note,'ABT',str(uuid.uuid4()),0,self.payload)
            self.assertEqual(recover(c,self.note)['status'],'draft')
    def test_saved_call_from_different_plan_cannot_be_materialized(self):
        with self.db(True) as (_,c):
            self.call(c,'===NOTE_START===Synthetic draft [1].===NOTE_END===')
            c.execute("UPDATE pipeline_lab_note_calls SET request=jsonb_set(request,'{modelKey}','\"different-model\"'::jsonb) WHERE note_id=%s",(self.note,))
            result=recover(c,self.note)
            self.assertEqual(result['status'],'attention')
            self.assertEqual(result['reason'],'saved_reply_needs_review')
