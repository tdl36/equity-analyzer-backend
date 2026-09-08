import copy
import json
import unittest
from contextlib import contextmanager
from flask import Flask
from research_restore import create_blueprint,fingerprint,metadata

REQUEST='11111111-1111-1111-1111-111111111111'
class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.source={'id':'old','ticker':'DE','created_at':'2026-01-01','note_markdown':'Historical note','note_docx':'original DOCX','charts':[{'id':'chart'}],'metadata':{},'status':'published','version':'1.0'}
        self.current={**self.source,'id':'new','created_at':'2026-02-01','note_markdown':'Current note'}
        self.receipt=None;self.inserts=[];self.calls=[];outer=self
        class Cursor:
            def execute(self,sql,args=()):
                self.sql=sql;outer.calls.append((sql,args))
                if sql.startswith('INSERT INTO research_notes') or sql.startswith('INSERT INTO investment_reviews'):outer.inserts.append((sql,args))
                if sql.startswith('INSERT INTO mp_jobs'):outer.receipt={'stage':'research_restore','input':json.loads(args[2]),'result':json.loads(args[3])}
            def fetchone(self):
                if 'FROM mp_jobs' in self.sql:return outer.receipt
                return outer.current if 'ORDER BY' in self.sql else outer.source
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,lambda tk,state,mode:('rendered markdown','rendered html','rendered PDF')));self.client=app.test_client()
    def body(self,kind='note'):
        return {'requestId':REQUEST,'kind':kind,'sourceId':self.source['id'],'sourceHash':fingerprint(kind,self.source),'latestId':self.current['id'],'latestHash':fingerprint(kind,self.current)}
    def post(self,body=None):return self.client.post('/api/research/restorations',json=body or self.body())
    def test_restore_creates_draft_and_preserves_original_and_charts(self):
        original=copy.deepcopy(self.source);r=self.post()
        self.assertEqual(r.status_code,201);self.assertEqual(r.json['status'],'draft');self.assertEqual(self.source,original)
        sql,args=self.inserts[0];self.assertIn("'draft'",sql);self.assertEqual(args[3],'Historical note');self.assertEqual(json.loads(args[7]),[{'id':'chart'}])
    def test_retry_is_idempotent_even_after_current_changes(self):
        body=self.body();first=self.post(body);self.current['note_markdown']='changed after save'
        again=self.post(body);self.assertEqual(again.status_code,200);self.assertEqual(first.json,again.json);self.assertEqual(len(self.inserts),1)
    def test_changed_source_latest_content_or_latest_identity_blocks(self):
        for row,key,value in [(self.source,'note_markdown','changed'),(self.current,'note_markdown','changed'),(self.current,'id','another')]:
            body=self.body();before=row[key];row[key]=value
            self.assertEqual(self.post(body).status_code,409);row[key]=before
        self.assertEqual(self.inserts,[])
    def test_request_identity_conflict_fails(self):
        body=self.body();self.post(body);body['sourceId']='other'
        self.assertEqual(self.post(body).status_code,409);self.assertEqual(len(self.inserts),1)
    def test_preview_is_bounded_and_not_a_mutation(self):
        self.source['note_markdown']='x'*130000
        r=self.client.get('/api/research/restoration-preview/note/old')
        self.assertEqual(r.status_code,200);self.assertTrue(r.json['source']['previewTruncated']);self.assertEqual(len(r.json['source']['markdown']),120000);self.assertFalse(self.inserts)
    def test_review_keeps_as_of_and_invalidates_current_readiness(self):
        self.source.update(state={'as_of':'2025-12-31','price':120},review_markdown='Historical review',metadata={'evidence':{'old':'evidence'},'readiness':{'status':'ready'}})
        self.current={**self.source,'id':'new'}
        r=self.post(self.body('review'));self.assertEqual(r.status_code,201)
        args=self.inserts[0][1];self.assertEqual(json.loads(args[3])['as_of'],'2025-12-31');meta=json.loads(args[-1]);self.assertEqual(meta['readiness']['status'],'needs_review');self.assertEqual(meta['historicalEvidence'],{'old':'evidence'})
    def test_latest_version_cannot_be_restored_over_itself(self):
        self.current=self.source;self.assertEqual(self.post().status_code,400);self.assertFalse(self.inserts)
    def test_malformed_kind_and_id_do_not_mutate(self):
        self.assertEqual(self.post({'kind':'table injection','requestId':REQUEST}).status_code,400)
        body=self.body();body['requestId']='bad';self.assertEqual(self.post(body).status_code,400);self.assertFalse(self.inserts)
    def test_historical_date_is_visible_in_export_formats(self):
        import io
        import investment_review as ir
        from PyPDF2 import PdfReader
        state=ir.ReviewState(ticker='DE',as_of='2025-12-31',historical_as_of='2025-12-31')
        pdf=ir.render_pdf(state)
        text=' '.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(pdf)).pages)
        for value in (ir.render_markdown(state),ir.render_html(state),text):
            self.assertIn('Historical restoration',value);self.assertIn('2025-12-31',value)
