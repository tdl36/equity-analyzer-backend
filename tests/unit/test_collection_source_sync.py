import json
import unittest
from tests.unit import test_collection_refresh as refresh
from collection_source_sync import reconcile
class SourceSyncTests(unittest.TestCase):
    setUp=refresh.RefreshTests.setUp
    def test_only_sealed_source_pauses_resume_and_defaults_freeze(self):
        p={'mode':'review','rules':[]};reconcile(self.m,p,[])
        self.m.save(self.cfg);self.clock+=86401;r=self.m.claim();self.assertEqual(r['config']['sourcePolicy'],p)
        self.m.mark(r['id'],r['owner'],'attention','Source selection: choose documents')
        s={'commandId':r['config']['sourceReviewId'],'ticker':'MDT','sealed':False,'candidates':[{'decision':'include'}]}
        self.assertEqual(reconcile(self.m,p,[s]),[])
        s['sealed']=True;self.assertEqual(reconcile(self.m,p,[s]),[r['id']])
        changed={'mode':'auto','rules':[]};reconcile(self.m,changed,[])
        saved=json.loads(self.c.db.execute('SELECT config FROM refresh_requests WHERE id=?',(r['id'],)).fetchone()['config'])
        self.assertEqual(saved['sourcePolicy'],p)
        claimed=self.m.claim();self.m.mark(r['id'],claimed['owner'],'attention','Research dispatch needs inspection')
        self.assertEqual(reconcile(self.m,p,[s]),[])
    def test_catalyst_customer_purchase_is_review_only(self):
        from catalyst_watch import candidate
        d=candidate('NVDA',{'headline':'Quantum Cyber Acquires NVIDIA A100 AI Compute Cluster','url':'https://example.com/release','datetime':10},0,11)
        self.assertTrue(d['requiresReview']);self.assertIn('customer event',d['triageReason'])
