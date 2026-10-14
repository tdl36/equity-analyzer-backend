"""Isolated PostgreSQL and synthetic research only; never production or paid APIs."""
import base64
import json
import unittest
import uuid
from unittest.mock import patch
from tests.unit import test_research_assignments as assignment_tests
PG=assignment_tests.PG
P=assignment_tests.P
from research_assignments import child,plan
from pipeline_lab import check_originals
from lab_source_identity import validate,verify

class IdentityTests(unittest.TestCase):
    def test_exact_identity_hash_and_names(self):
        import hashlib
        expected=[{'id':'1','filename':'a.txt','sha256':hashlib.sha256(b'a').hexdigest()}]
        self.assertEqual(validate(expected,['a.txt']),expected)
        verify(expected,[{'id':1,'filename':'a.txt','file_data':'YQ=='}])
        for row in [{'id':2,'filename':'a.txt','file_data':'YQ=='},{'id':1,'filename':'a.txt','file_data':'Yg=='}]:
            with self.assertRaises(ValueError):verify(expected,[row])
        for pack in [[],expected*2,[{**expected[0],'sha256':'bad'}]]:
            with self.assertRaises(ValueError):validate(pack,['a.txt'])
    def test_separate_task_contract(self):
        sel={'task':'note','revision':1,'digest':'a'*64}
        with self.assertRaises(ValueError):plan({**P,'labSelection':sel})
        self.assertEqual(plan({**P,'outputs':['summary_note'],'labSelection':sel})['labSelection'],sel)

@unittest.skipUnless((PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class LabAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assignment_tests.AssignmentPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,cur):cur.execute('ALTER TABLE document_files ADD COLUMN id SERIAL')
    tearDownClass=classmethod(assignment_tests.AssignmentPostgresTests.tearDownClass.__func__)
    setUp=assignment_tests.AssignmentPostgresTests.setUp
    invoke=assignment_tests.AssignmentPostgresTests.invoke
    report=assignment_tests.AssignmentPostgresTests.report
    def prepare(self):
        with self.db(True) as (_,cur):
            cur.execute('SELECT id,file_data FROM document_files WHERE ticker=%s',(self.ticker,));doc=cur.fetchone()
            ref='charlie:thesis_'+str(doc['id'])
            cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s)',('pipeline_lab_v1:'+self.ticker,json.dumps({'revision':1,'selection':{'thesis':[ref],'note':[ref]}})))
        checked=check_originals(self.ticker,'note',1,[ref],{ref:{'filename':'one.txt','file_data':doc['file_data']}})
        return {**P,'ticker':self.ticker,'requestId':self.ident,'outputs':['summary_note'],'labSelection':{'revision':1,'task':'note','digest':checked['digest']}}
    def test_saved_pack_replay_no_collection_and_draft_delivery(self):
        body=self.prepare();r=self.client.post('/api/research/assignments',json=body);self.assertEqual(r.status_code,202,r.json)
        original=self.co.read(self.ident)['input']['labPack']
        with self.db(True) as (_,cur):
            cur.execute('SELECT id FROM mp_jobs WHERE id=%s',(child(self.ident,'collection'),));self.assertIsNone(cur.fetchone())
            cur.execute('UPDATE app_settings SET value=%s WHERE key=%s',(json.dumps({'revision':2,'selection':{'thesis':[],'note':[]}}),'pipeline_lab_v1:'+self.ticker))
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,200)
        self.assertEqual(self.co.read(self.ident)['input']['labPack'],original)
        self.co.advance(self.ident)
        row=self.co.read(self.ident);self.assertEqual(row['status'],'complete',row.get('error'))
        self.assertIn('summary_note',row['result']['artifacts'])
        research=row['result']['researchInput']
        self.assertEqual(research['expectedOriginals'][0]['sha256'],self.sha)
    def test_stale_bytes_and_revision_reject_before_reservation(self):
        body=self.prepare();body['labSelection']['digest']='0'*64
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,409)
        self.assertIsNone(self.co.read(self.ident))
        body['labSelection']['revision']=2
        self.assertEqual(self.client.post('/api/research/assignments',json=body).status_code,409)
        self.assertIsNone(self.co.read(self.ident))

class MetadataTests(unittest.TestCase):
    def test_only_allowlisted_valid_metadata(self):
        from pipeline_source_metadata import fields
        self.assertEqual(fields('bad')['publishedDate'],None)
        result=fields({'published':'2026-10-01','publisher':'Broker A','source':'alphasense_collection','sha256':'a'*64,'privateBody':'secret'})
        self.assertTrue(result['originalReceiptRecorded']);self.assertEqual(result['publishedDate'],'2026-10-01')
        self.assertNotIn('privateBody',result)
        self.assertIsNone(fields({'published':'yesterday'})['publishedDate'])

from tests.unit import test_stock_analysis_postgres as stock_tests
@unittest.skipUnless((PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class LabWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stock_tests.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,cur):cur.execute('ALTER TABLE document_files ADD COLUMN id SERIAL')
    tearDownClass=classmethod(stock_tests.StockPostgresTests.tearDownClass.__func__)
    setUp=stock_tests.StockPostgresTests.setUp
    start=stock_tests.StockPostgresTests.start
    def pack(self):
        from command_thesis_bridge import file_hash
        with self.db() as (_,cur):
            cur.execute('SELECT id,filename,file_data FROM document_files WHERE ticker=%s',(self.ticker,));d=cur.fetchone()
        return [{'id':str(d['id']),'filename':d['filename'],'sha256':file_hash(d)}]
    def test_exact_pack_completes_and_replays_without_new_calls(self):
        self.payload['expectedOriginals']=self.pack()
        r=self.start();self.assertEqual(r['status'],'complete',r.get('error'))
        count=len(self.calls);self.assertEqual(self.client.post(self.url,json=self.payload).status_code,200);self.assertEqual(len(self.calls),count)
        other={**self.payload,'expectedOriginals':[{**self.pack()[0],'id':'999999'}]}
        self.assertEqual(self.client.post(self.url,json=other).status_code,409)
    def test_changed_pack_rejected_before_work(self):
        self.payload['expectedOriginals']=[{**self.pack()[0],'sha256':'0'*64}]
        self.assertEqual(self.client.post(self.url,json=self.payload).status_code,409)
        self.assertEqual(self.calls,[]);self.thread.assert_not_called()
    def test_identity_change_after_submission_stops_before_paid_call(self):
        self.payload['expectedOriginals']=self.pack()
        self.assertEqual(self.client.post(self.url,json=self.payload).status_code,202)
        with self.db(True) as (_,cur):cur.execute('UPDATE document_files SET id=id+10000 WHERE ticker=%s',(self.ticker,))
        self.thread.call_args.kwargs['target'](*self.thread.call_args.kwargs['args'])
        r=self.client.get('/api/research/stock-analysis-run/'+self.payload['requestId']).json
        self.assertEqual(r['status'],'attention');self.assertEqual(self.calls,[])
