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
        from pipeline_lab_notes import read
        with self.db(True) as (_,c):
            self.call(c,'===NOTE_START===Synthetic draft [1].===NOTE_END===')
            result=recover(c,self.note)
            self.assertEqual(result['status'],'draft');self.assertEqual(result['revision'],1)
            self.assertEqual(recover(c,self.note)['revision'],1)
            self.assertNotIn('researchProvenance',read(c,self.note))

    def test_saved_final_reply_recovery_retains_purpose_and_frozen_source_context(self):
        from pipeline_lab_note_jobs import enqueue
        from pipeline_lab_notes import read
        from pipeline_lab_note_review import export_text
        for purpose,label in (({'kind':'event_note','event':{'title':'Fixture earnings','date':'2025-03-01'}},'does not update the thesis'),
                              ({'kind':'thesis_review'},'Longitudinal thesis review')):
            with self.subTest(purpose=purpose):
                self.note=str(uuid.uuid4());source=self.plan['sourcePack'][0]
                provenance={'version':1,'purpose':purpose,'sourceEvidence':{'version':1,'sources':[
                    {'id':source['reference'][len('charlie:thesis_'):],'filename':source['filename'],'sha256':source['sha256'],
                     'role':'supplemental','publishedDate':'2024-03-01','publisher':'Frozen fixture publisher',
                     'sourceKind':'transcript','referencedBySavedThesis':False}]}}
                self.plan={**self.plan,'noteId':self.note,'researchProvenance':provenance,
                           'warnings':[{'reference':source['reference'],'issue':'Synthetic omitted source context.'}]}
                with self.db(True) as (_,c):
                    enqueue(c,self.plan)
                    self.call(c,'===NOTE_START===Synthetic recovered draft.===NOTE_END===')
                    receipt=recover(c,self.note);self.assertEqual(receipt['status'],'draft')
                    draft=read(c,self.note)
                    self.assertEqual(draft.get('researchProvenance'),provenance)
                    self.assertIn(label,export_text(draft))
                    self.assertIn(r'2024\-03\-01',export_text(draft))
                    self.assertFalse(draft['investorApproved']);self.assertEqual(draft['status'],'draft')
                    self.assertTrue(any('Synthetic omitted source context.' in issue for issue in draft['qualityIssues']))
                    self.assertEqual(recover(c,self.note)['revision'],1)
                    self.assertEqual(read(c,self.note),draft)
                    c.execute('SELECT COUNT(*) AS n FROM pipeline_lab_note_calls WHERE note_id=%s',(self.note,))
                    self.assertEqual(c.fetchone()['n'],1)
                    c.execute('SELECT COUNT(*) AS n FROM pipeline_lab_note_versions WHERE note_id=%s',(self.note,))
                    self.assertEqual(c.fetchone()['n'],1)
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
