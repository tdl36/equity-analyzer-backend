import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from pipeline_lab_icloud_agent import process_one

class AgentImport(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.stocks=self.root/'STOCKS';self.catalysts=self.root/'CATALYSTS'
        (self.stocks/'ABT').mkdir(parents=True);self.path=self.stocks/'ABT'/'a.txt';self.path.write_bytes(b'original')
        self.ref='icloud:'+json.dumps(['main','a.txt']);self.job={'requestId':'fixture','input':{'ticker':'ABT'},'status':'queued','references':[self.ref]};self.calls=[]
    def get(self,path):
        self.calls.append(('GET',path))
        if path.endswith('imports'):return {'imports':[copy.deepcopy(self.job)] if self.job['status']!='complete' else []}
        return copy.deepcopy(self.job)
    def post(self,path,body):
        self.calls.append(('POST',path))
        if path.endswith('freeze'):self.job.update(status='frozen',manifest=body['manifest'])
        else:self.job.update(status='complete',receipts=[{**r,'target':'charlie:thesis_1'} for r in self.job['manifest']])
        return copy.deepcopy(self.job)
    def run_import(self,post=None):return process_one(self.get,post or self.post,self.stocks,self.catalysts)
    def test_freeze_before_upload_and_completion_stops_reads(self):
        self.assertEqual(self.run_import(),'complete');self.path.unlink()
        self.assertIsNone(self.run_import())
        self.assertTrue(self.calls[2][1].endswith('freeze'));self.assertTrue(self.calls[3][1].endswith('complete'))
    def test_lost_freeze_response_resumes_frozen_hash(self):
        def lost(path,body):
            self.post(path,body);raise ConnectionError('synthetic lost response')
        with self.assertRaises(ConnectionError):self.run_import(lost)
        self.assertEqual(self.job['status'],'frozen')
        self.assertEqual(self.run_import(),'complete')
    def test_changed_original_after_lost_freeze_is_never_uploaded(self):
        def lost(path,body):self.post(path,body);raise ConnectionError()
        with self.assertRaises(ConnectionError):self.run_import(lost)
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'frozen'):self.run_import()
        self.assertFalse(any(p.endswith('complete') for _,p in self.calls))
    def test_lost_complete_response_reconciles_without_rereading(self):
        def lost(path,body):
            result=self.post(path,body)
            if path.endswith('complete'):raise ConnectionError()
            return result
        with self.assertRaises(ConnectionError):self.run_import(lost)
        self.path.unlink();self.assertIsNone(self.run_import())
    def test_cancellation_between_listing_and_read_stops(self):
        original_get=self.get
        def cancelled(path):
            result=original_get(path)
            if path.endswith('imports'):self.job['status']='cancelled'
            return result
        self.path.unlink()
        self.assertEqual(process_one(cancelled,self.post,self.stocks,self.catalysts),'cancelled')
        self.assertFalse(any(method=='POST' for method,_ in self.calls))
    def test_mismatched_receipt_never_claims_success(self):
        def wrong(path,body):
            result=self.post(path,body)
            if path.endswith('complete'):result['receipts'][0]['sha256']='0'*64
            return result
        with self.assertRaisesRegex(ValueError,'completion receipt'):self.run_import(wrong)
