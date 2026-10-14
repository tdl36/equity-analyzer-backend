import base64,hashlib,json,unittest,uuid
from tests.unit import test_pipeline_lab_import as import_fixtures
from tests.unit import test_stock_analysis_postgres as fixtures
from pipeline_lab_icloud import enqueue,freeze,complete,cancel,load,PREFIX

@unittest.skipUnless((fixtures.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class ICloudTransactions(unittest.TestCase):
    setUpClass=classmethod(import_fixtures.Imports.setUpClass.__func__)
    tearDownClass=classmethod(import_fixtures.Imports.tearDownClass.__func__)
    def setUp(self):
        self.tk='T'+uuid.uuid4().hex[:8].upper();self.request=str(uuid.uuid4());self.cfg='pipeline_lab_v1:'+self.tk
        self.refs=['icloud:'+json.dumps(['main','a.txt']),'icloud:'+json.dumps(['Reports','b.txt'])]
        self.manifest=[dict(reference=r,filename=n,sha256=hashlib.sha256(n.encode()).hexdigest(),bytes=len(n)) for r,n in zip(self.refs,['a.txt','b.txt'])]
        self.bodies=[dict(reference=r,fileData=base64.b64encode(n.encode()).decode()) for r,n in zip(self.refs,['a.txt','b.txt'])]
        with self.db(True) as (_,c):c.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s)',(self.cfg,json.dumps({'revision':1,'selection':{'thesis':self.refs,'note':self.refs[:1]}})))
    def invoke(self,fn,*args):
        with self.db(True) as (_,c):return fn(c,*args)
    def start(self):
        self.invoke(enqueue,self.tk,'thesis',1,self.request);return self.invoke(freeze,self.request,self.manifest)
    def test_queue_replay_freeze_restart_and_atomic_completion(self):
        q=self.invoke(enqueue,self.tk,'thesis',1,self.request)
        self.assertEqual(self.invoke(enqueue,self.tk,'thesis',1,self.request),q)
        self.invoke(freeze,self.request,self.manifest)
        self.assertEqual(self.invoke(freeze,self.request,list(reversed(self.manifest)))['status'],'frozen')
        receipt=self.invoke(complete,self.request,self.bodies);self.assertFalse(receipt['researchStarted']);self.assertEqual(receipt['status'],'complete')
        with self.db(True) as (_,c):
            cfg=load(c,self.cfg);self.assertEqual(cfg['revision'],2);self.assertEqual(cfg['selection']['note'],self.refs[:1])
            self.assertTrue(all(r.startswith('charlie:thesis_') for r in cfg['selection']['thesis']))
            c.execute('UPDATE app_settings SET value=%s WHERE key=%s',(json.dumps({'revision':3,'selection':{'thesis':[],'note':[]}}),self.cfg))
        self.assertEqual(self.invoke(complete,self.request,[]),receipt)
        with self.db() as (_,c):self.assertEqual(load(c,self.cfg)['revision'],3)
    def test_concurrent_identity_and_changed_manifest_rejected(self):
        self.start()
        with self.assertRaises(ValueError):self.invoke(enqueue,self.tk,'thesis',1,str(uuid.uuid4()))
        with self.assertRaises(ValueError):self.invoke(enqueue,self.tk,'note',1,self.request)
        changed=[{**m,'sha256':'f'*64} for m in self.manifest]
        with self.assertRaises(ValueError):self.invoke(freeze,self.request,changed)
        self.invoke(cancel,self.request)
        with self.assertRaises(ValueError):self.invoke(complete,self.request,self.bodies)
        self.assertEqual(self.invoke(enqueue,self.tk,'thesis',1,str(uuid.uuid4()))['status'],'queued')
    def test_byte_and_filename_conflicts_rollback(self):
        self.start()
        bad=[{**b,'fileData':'Yw=='} for b in self.bodies]
        with self.assertRaises(ValueError):self.invoke(complete,self.request,bad)
        with self.db(True) as (_,c):c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s)',(self.tk,'b.txt','Yw==','{}'))
        with self.assertRaises(ValueError):self.invoke(complete,self.request,self.bodies)
        with self.db() as (_,c):
            c.execute('SELECT COUNT(*) AS n FROM document_files WHERE ticker=%s',(self.tk,));self.assertEqual(c.fetchone()['n'],1)
            self.assertEqual(load(c,self.cfg)['revision'],1);self.assertEqual(load(c,PREFIX+self.request)['status'],'frozen')
    def test_stale_settings_never_replaced(self):
        self.start()
        with self.db(True) as (_,c):c.execute('UPDATE app_settings SET value=%s WHERE key=%s',(json.dumps({'revision':2,'selection':{'thesis':[],'note':[]}}),self.cfg))
        with self.assertRaises(ValueError):self.invoke(complete,self.request,self.bodies)
        with self.db() as (_,c):
            c.execute('SELECT COUNT(*) AS n FROM document_files WHERE ticker=%s',(self.tk,));self.assertEqual(c.fetchone()['n'],0)
    def test_duplicate_names_and_foreign_reference_block_freeze(self):
        self.invoke(enqueue,self.tk,'thesis',1,self.request)
        bad=[{**m,'reference':'icloud:["Other","a.txt"]','filename':'a.txt'} for m in self.manifest]
        with self.assertRaises(ValueError):self.invoke(freeze,self.request,bad)

    def test_two_request_ids_cannot_queue_competing_stock_imports(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        barrier=Barrier(2)
        def submit(request_id):
            barrier.wait()
            try:return self.invoke(enqueue,self.tk,'thesis',1,request_id)['status']
            except ValueError:return 'blocked'
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes=list(pool.map(submit,[self.request,str(uuid.uuid4())]))
        self.assertEqual(sorted(outcomes),['blocked','queued'])
