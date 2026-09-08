import unittest,json
from contextlib import contextmanager
from unittest.mock import Mock,patch
from manual_meeting_recovery import execute

class RecoveryTests(unittest.TestCase):
    def setup_db(self,status='running',attempts=0):
        cur=Mock();cur.fetchone.return_value={'status':status,'input':{'recoveryAttempts':attempts},'result':{'analyses':[{'done':True},None]}}
        @contextmanager
        def db(**kwargs):yield None,cur
        return db,cur
    @contextmanager
    def lock(self,*args):yield True
    def test_recovery_retains_partial_checkpoint_and_assigns_owner(self):
        db,cur=self.setup_db();run=Mock()
        with patch('amendment_ownership.worker_session',self.lock):self.assertTrue(execute(db,'job',run,True))
        inp,cp=run.call_args.args
        self.assertEqual(inp['recoveryAttempts'],1);self.assertTrue(inp['workerToken'])
        self.assertEqual(cp['analyses'],[{'done':True},None])
    def test_completed_job_is_never_replayed(self):
        db,cur=self.setup_db('done');run=Mock()
        with patch('amendment_ownership.worker_session',self.lock):self.assertFalse(execute(db,'job',run,True))
        run.assert_not_called()
    def test_recovery_is_bounded(self):
        db,cur=self.setup_db(attempts=2);run=Mock()
        with patch('amendment_ownership.worker_session',self.lock):self.assertFalse(execute(db,'job',run,True))
        run.assert_not_called();self.assertIn('Automatic recovery limit',cur.execute.call_args.args[0])
    def test_active_owner_prevents_duplicate_model_work(self):
        db,cur=self.setup_db();run=Mock()
        @contextmanager
        def busy(*args):yield False
        with patch('amendment_ownership.worker_session',busy):self.assertFalse(execute(db,'job',run,True))
        run.assert_not_called();cur.execute.assert_not_called()
    def test_local_worker_guard_survives_unreliable_database_lock(self):
        import threading
        db,cur=self.setup_db();entered=threading.Event();release=threading.Event();errors=[]
        def active(*args):entered.set();release.wait(5)
        def first():
            try:execute(db,'local-guard-fixture',active)
            except Exception as e:errors.append(e)
        with patch('amendment_ownership.worker_session',self.lock):
            t=threading.Thread(target=first);t.start();self.assertTrue(entered.wait(2))
            try:self.assertFalse(execute(db,'local-guard-fixture',Mock(),True))
            finally:release.set();t.join(5)
        self.assertFalse(errors)
    def test_valid_lease_blocks_recovery_even_without_session_lock(self):
        db,cur=self.setup_db();cur.fetchone.return_value['lease_current']=True;run=Mock()
        with patch('amendment_ownership.worker_session',self.lock):self.assertFalse(execute(db,'lease-fixture',run,True))
        run.assert_not_called()
