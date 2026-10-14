import unittest
from tests.unit import test_pipeline_lab_notes as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_plan import schema,freeze,load,verify_preparation


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Plans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DraftVersions.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):schema(c)
    tearDownClass=classmethod(fixtures.DraftVersions.tearDownClass.__func__)
    def setUp(self):
        fixtures.DraftVersions.setUp(self)
        self.plan={'noteId':self.note,'ticker':'ABT','title':'Fixture draft','context':'Frozen context',
                   'promptTemplate':'Full note fixture','modelKey':'fixture','maxTokens':1000,
                   'sourcePack':self.payload['sourcePack'],'representationSha256':'a'*64,
                   'batchCount':1,'warnings':[],'allowPdfText':False}
    def test_replay_and_read_keep_frozen_context(self):
        with self.db(True) as (_,c):
            self.assertFalse(freeze(c,self.plan)['replayed'])
            self.assertTrue(freeze(c,self.plan)['replayed'])
            self.assertEqual(load(c,self.note),self.plan)
        for field,value in [('context','New context'),('ticker','UNH'),('maxTokens',2000),('batchCount',2)]:
            with self.assertRaises(ValueError),self.db(True) as (_,c):freeze(c,{**self.plan,field:value})
    def test_parallel_different_plans_have_only_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        gate=Barrier(2)
        def save(n):
            gate.wait()
            try:
                with self.db(True) as (_,c):freeze(c,{**self.plan,'context':str(n)})
                return 'saved'
            except ValueError:return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:self.assertCountEqual(list(pool.map(save,[1,2])),['saved','conflict'])
    def test_invalid_plan_and_representation_drift_block(self):
        for field,value in [('context','x'*500001),('batchCount',True),('allowPdfText',1),('warnings',[{'reference':'charlie:thesis_2','issue':'Wrong source'}])]:
            with self.assertRaises(ValueError),self.db(True) as (_,c):freeze(c,{**self.plan,field:value})
        prepared={'sourcePack':self.plan['sourcePack'],'representationSha256':'a'*64,'batches':[[]],'warnings':[]}
        verify_preparation(self.plan,prepared)
        for field,value in [('representationSha256','b'*64),('batches',[[],[]]),('warnings',[{}])]:
            with self.assertRaises(ValueError):verify_preparation(self.plan,{**prepared,field:value})
