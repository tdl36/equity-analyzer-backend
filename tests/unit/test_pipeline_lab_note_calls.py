import unittest
import uuid
from tests.unit import test_pipeline_lab_notes as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_calls import schema,reserve,run_once,finish


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class Calls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DraftVersions.setUpClass.__func__(cls)
        with cls.db(True) as (_,c): schema(c)
    tearDownClass=classmethod(fixtures.DraftVersions.tearDownClass.__func__)

    def setUp(self):
        fixtures.DraftVersions.setUp(self)
        self.request={'noteId':self.note,'batch':1,'sourcePack':self.payload['sourcePack'],
                      'contextSha256':'b'*64,'promptSha256':'c'*64,'modelKey':'synthetic-model','maxTokens':1000}
        self.response={'text':'Synthetic reply','provider':'fixture','model':'synthetic-model','stopReason':'end_turn'}

    def test_intent_is_committed_before_dispatch_and_reply_reused(self):
        calls=[]
        def invoke():
            with self.db() as (_,c):
                c.execute('SELECT status FROM pipeline_lab_note_calls WHERE note_id=%s',(self.note,))
                self.assertEqual(c.fetchone()['status'],'dispatched')
            calls.append(True);return self.response
        first=run_once(self.db,self.request,invoke)
        self.assertEqual(run_once(self.db,self.request,invoke),first)
        self.assertEqual(calls,[True])
        with self.assertRaises(ValueError):run_once(self.db,{**self.request,'promptSha256':'d'*64},invoke)

    def test_failure_and_interrupted_intent_never_retry(self):
        calls=[]
        def invoke():calls.append(True);raise RuntimeError('private material')
        result=run_once(self.db,self.request,invoke)
        self.assertEqual(result,{'status':'attention','reason':'provider_outcome_unknown'})
        self.assertEqual(run_once(self.db,self.request,invoke),result)
        self.assertEqual(calls,[True])
        other={**self.request,'noteId':str(uuid.uuid4())}
        with self.db(True) as (_,c):reserve(c,other)
        self.assertEqual(run_once(self.db,other,invoke),result)
        self.assertEqual(calls,[True])

    def test_concurrent_reservations_have_one_dispatch(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        gate=Barrier(2)
        def claim(_):
            gate.wait()
            with self.db(True) as (_,c):return reserve(c,self.request)['dispatch']
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(claim,[1,2])),[True,False])

    def test_receipt_is_bound_and_immutable(self):
        with self.db(True) as (_,c):intent=reserve(c,self.request)
        with self.assertRaises(ValueError),self.db(True) as (_,c):
            finish(c,self.note,1,'d'*64,self.response)
        with self.db(True) as (_,c):finish(c,self.note,1,intent['requestHash'],self.response)
        with self.assertRaises(ValueError),self.db(True) as (_,c):
            finish(c,self.note,1,intent['requestHash'],{**self.response,'text':'Different'})
    def test_reply_save_failure_does_not_repeat_provider_call(self):
        from contextlib import contextmanager
        writes=[];calls=[]
        @contextmanager
        def failing_db(commit=False):
            with self.db(commit) as pair:
                if commit:writes.append(True)
                yield pair
                if commit and len(writes)==2:
                    raise RuntimeError('Synthetic receipt transaction failure')
        def invoke():calls.append(True);return self.response
        result=run_once(failing_db,self.request,invoke)
        self.assertEqual(result['status'],'attention')
        self.assertEqual(run_once(self.db,self.request,invoke)['status'],'attention')
        self.assertEqual(calls,[True])
