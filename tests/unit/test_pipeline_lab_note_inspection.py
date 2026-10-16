"""Exact job reads retain uncertain calls and never invoke plan building or recovery."""
import json
import unittest
import uuid
from pipeline_lab_note_calls import reserve,finish
from pipeline_lab_notes import append
from tests.unit import test_pipeline_lab_note_submit as fixtures
from tests.unit import test_stock_analysis_postgres as pg

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Inspection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Submission.setUpClass.__func__(cls)
        from pipeline_lab_note_service import initialize
        with cls.db(True) as (_,c):initialize(c)
    tearDownClass=classmethod(fixtures.Submission.tearDownClass.__func__)
    setUp=fixtures.Submission.setUp
    def create(self):
        self.plan['batchCount']=2
        self.assertEqual(self.client.post(self.url,headers=self.headers,json=self.body).status_code,200)
        self.path='/api/pipeline-lab/note-job/'+self.note+'/inspection?ticker=ABT'
        with self.db(True) as (_,c):
            for batch in (1,2):
                request={'noteId':self.note,'batch':batch,'sourcePack':self.plan['sourcePack'],'contextSha256':'a'*64,'promptSha256':'b'*64,'modelKey':'fixture','maxTokens':1000}
                saved=reserve(c,request)
                if batch==1:finish(c,self.note,batch,saved['requestHash'],{'text':'Private synthetic reply body','provider':'fixture','model':'fixture','stopReason':'end_turn'})
            c.execute("UPDATE pipeline_lab_note_calls SET status='uncertain' WHERE note_id=%s AND batch=2",(self.note,))
            c.execute("UPDATE pipeline_lab_note_jobs SET status='attention',reason='provider_outcome_unknown',created_at=NOW()-INTERVAL '90 days' WHERE note_id=%s",(self.note,))
    def test_old_exact_progress_preserves_call_outcomes_without_private_bodies_or_writes(self):
        self.create()
        with self.db() as (_,c):c.execute('SELECT * FROM pipeline_lab_note_jobs WHERE note_id=%s',(self.note,));before=dict(c.fetchone())
        calls=len(self.calls)
        for _ in range(2):
            r=self.client.get(self.path,headers=self.headers);self.assertEqual(r.status_code,200,r.json)
            d=r.json;self.assertEqual((d['noteId'],d['ticker'],d['status']),(self.note,'ABT','attention'))
            self.assertTrue(d['readOnly']);self.assertEqual(d['batchCount'],2)
            self.assertEqual([(b['batch'],b['status']) for b in d['batches']],[(1,'complete'),(2,'uncertain')])
            self.assertIsNone(d['revision']);self.assertIn('no-store',r.headers['Cache-Control'])
            for private in ('Private synthetic reply body','sourcePack','request_hash','lease_id'):
                self.assertNotIn(private,r.get_data(as_text=True))
        self.assertEqual(len(self.calls),calls)
        with self.db() as (_,c):
            c.execute('SELECT * FROM pipeline_lab_note_jobs WHERE note_id=%s',(self.note,));self.assertEqual(dict(c.fetchone()),before)
            c.execute('SELECT COUNT(*) AS n FROM pipeline_lab_note_calls WHERE note_id=%s',(self.note,));self.assertEqual(c.fetchone()['n'],2)
    def test_wrong_company_missing_bad_id_and_extra_filters_blocked(self):
        self.create()
        for path,code in [(self.path.replace('ticker=ABT','ticker=UNH'),404),(self.path.replace(self.note,str(uuid.uuid4())),404),(self.path.replace(self.note,'bad'),400),(self.path+'&ticker=ABT',400),(self.path+'&retry=1',400),(self.path.split('?')[0],400)]:
            self.assertEqual(self.client.get(path,headers=self.headers).status_code,code)
        self.assertEqual(self.client.get(self.path).status_code,401)
    def test_latest_revision_and_older_acceptance_are_independent_of_job_receipt(self):
        self.create()
        with self.db(True) as (_,c):
            append(c,self.note,'ABT',str(uuid.uuid4()),0,self.payload)
            c.execute("INSERT INTO pipeline_lab_note_acceptances(note_id,revision,investor) VALUES(%s,1,'fixture')",(self.note,))
            append(c,self.note,'ABT',str(uuid.uuid4()),1,{**self.payload,'markdown':'Newer synthetic draft'})
        d=self.client.get(self.path,headers=self.headers).json
        self.assertEqual((d['revision'],d['acceptedRevision'],d['status']),(2,1,'attention'))
