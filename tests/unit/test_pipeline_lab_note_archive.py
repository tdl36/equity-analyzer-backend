"""Archive traversal is read-only and keeps older revision receipts discoverable."""
import base64
import json
import unittest
import uuid
from pipeline_lab_note_jobs import enqueue
from pipeline_lab_notes import append
from tests.unit import test_pipeline_lab_note_review as fixtures
from tests.unit import test_stock_analysis_postgres as pg

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Archive(unittest.TestCase):
    setUpClass=classmethod(fixtures.Review.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.Review.tearDownClass.__func__)
    setUp=fixtures.Review.setUp
    def page(self,cursor=None,ticker='ABT'):
        return self.client.get('/api/pipeline-lab/note-jobs/'+ticker,query_string={} if cursor is None else {'cursor':cursor},headers={'X-Auth':'machine'})
    def populate(self):
        with self.db(True) as (_,c):
            for i in range(64):
                ident=str(uuid.uuid4())
                enqueue(c,{**self.plan,'noteId':ident})
                if i==0:
                    self.older=ident
                    append(c,ident,'ABT',str(uuid.uuid4()),0,self.payload)
                    c.execute("INSERT INTO pipeline_lab_note_acceptances(note_id,revision,investor) VALUES(%s,1,'synthetic investor')",(ident,))
                    append(c,ident,'ABT',str(uuid.uuid4()),1,{**self.payload,'title':'Edited older note'})
            c.execute("UPDATE pipeline_lab_note_jobs SET created_at='2026-01-01T00:00:00Z',status='attention',reason='provider_outcome_unknown'")
            c.execute('SELECT note_id::text AS id FROM pipeline_lab_note_jobs ORDER BY created_at DESC,note_id DESC');self.ids=[r['id'] for r in c.fetchall()]
    def test_all_pages_same_timestamp_and_new_insertion_preserve_archive(self):
        self.populate();first=self.page();self.assertEqual(first.status_code,200);self.assertEqual(len(first.json['jobs']),30)
        self.assertIn('no-store',first.headers['Cache-Control'])
        with self.db(True) as (_,c):enqueue(c,{**self.plan,'noteId':str(uuid.uuid4())})
        rows=first.json['jobs'];cursor=first.json['nextCursor']
        while cursor:
            response=self.page(cursor);self.assertEqual(response.status_code,200,response.json)
            rows+=response.json['jobs'];cursor=response.json['nextCursor']
        self.assertEqual([r['noteId'] for r in rows],self.ids)
        old=next(r for r in rows if r['noteId']==self.older)
        self.assertEqual((old['title'],old['revision'],old['acceptedRevision']),('Edited older note',2,1))
        self.assertEqual(old['status'],'attention');self.assertEqual(old['reason'],'provider_outcome_unknown')
        self.assertNotIn('markdown',old);self.assertNotIn('sourcePack',old)
        with self.db() as (_,c):
            c.execute("SELECT COUNT(*) AS n FROM pipeline_lab_note_jobs WHERE status='attention'");self.assertEqual(c.fetchone()['n'],65)
            c.execute('SELECT COUNT(*) AS n FROM pipeline_lab_note_calls');self.assertEqual(c.fetchone()['n'],0)
    def test_bad_cursor_wrong_company_and_duplicate_filters_rejected(self):
        self.populate();cursor=self.page().json['nextCursor']
        self.assertEqual(self.page(cursor,ticker='UNH').status_code,400)
        for value in ('','not-json','x'*513,base64.urlsafe_b64encode(json.dumps(['2026-01-01',self.note,'ABT']).encode()).decode()):
            self.assertEqual(self.page(value).status_code,400)
        for query in ('cursor=x&cursor=y','limit=1000'):
            self.assertEqual(self.client.get('/api/pipeline-lab/note-jobs/ABT?'+query,headers={'X-Auth':'machine'}).status_code,400)
        self.assertEqual(self.client.get('/api/pipeline-lab/note-jobs/ABT').status_code,401)
        empty=self.page(ticker='UNH').json;self.assertEqual(empty['jobs'],[]);self.assertIsNone(empty['nextCursor']);self.assertTrue(empty['readOnly'])
