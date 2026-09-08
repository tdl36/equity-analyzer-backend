import unittest
from contextlib import contextmanager
from unittest.mock import Mock,patch
from amendment_ownership import worker_session
from amendment_checkpoints import Checkpoints

class OwnershipTests(unittest.TestCase):
    def test_lock_released_even_when_work_raises(self):
        conn=Mock();cur=Mock();cur.fetchone.return_value={'locked':True}
        @contextmanager
        def db():yield conn,cur
        with self.assertRaises(ValueError):
            with worker_session(db,'job') as acquired:
                self.assertTrue(acquired);raise ValueError('work failed')
        self.assertIn('pg_advisory_unlock',cur.execute.call_args.args[0])
        conn.close.assert_not_called()
    def test_unavailable_lock_does_not_unlock_other_owner(self):
        conn=Mock();cur=Mock();cur.fetchone.return_value={'locked':False}
        @contextmanager
        def db():yield conn,cur
        with worker_session(db,'job') as acquired:self.assertFalse(acquired)
        self.assertEqual(cur.execute.call_count,1)
    def test_unlock_error_discards_pool_connection(self):
        conn=Mock();cur=Mock();cur.fetchone.return_value={'locked':True};cur.execute.side_effect=[None,RuntimeError('lost session')]
        @contextmanager
        def db():yield conn,cur
        with self.assertRaises(RuntimeError):
            with worker_session(db,'job'):pass
        conn.close.assert_called_once()
    def test_stale_token_blocks_checkpoint_write(self):
        cur=Mock();cur.fetchone.return_value={'status':'running','owner':'new','result':{}}
        @contextmanager
        def db(commit=False):yield None,cur
        cp=Checkpoints(db,'job','old')
        with self.assertRaises(ValueError):cp.load('key','draft')
        with self.assertRaises(ValueError):cp.save('key','draft',{})
        self.assertFalse(any(c.args[0].startswith('UPDATE') for c in cur.execute.call_args_list))

class RecoveryTests(unittest.TestCase):
    def client(self,locked=True,attempts=0,changed=False,key='configured'):
        from flask import Flask
        from research_amendments import create_blueprint
        cur=Mock();baseline={'thesis':{'summary':'saved'}}
        cur.fetchall.return_value=[{'id':'job','ticker':'UNH','input':{'baseline':baseline,'filenames':['doc.pdf'],'workerToken':'old','autoRecoveryAttempts':attempts}}]
        cur.fetchone.side_effect=[{'locked':locked},{'analysis':{} if changed else baseline}]
        @contextmanager
        def db(commit=False):yield None,cur
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,Mock(),lambda supplied:key))
        return app.test_client(),cur
    def test_live_owner_or_missing_key_never_restarted(self):
        for options in ({'locked':False},{'key':''}):
            client,cur=self.client(**options)
            with patch('research_amendments.threading.Thread') as thread:
                r=client.post('/api/agent/recover-amendments')
                self.assertEqual(r.json['recovered'],[]);thread.assert_not_called()
                self.assertFalse(any(c.args[0].startswith('UPDATE') for c in cur.execute.call_args_list))
    def test_abandoned_job_requeued_with_new_attempt_and_cleared_owner(self):
        import json
        client,cur=self.client()
        with patch('research_amendments.threading.Thread') as thread:
            r=client.post('/api/agent/recover-amendments')
            self.assertEqual(r.json['recovered'],['job']);thread.assert_called_once()
            update=next(c for c in cur.execute.call_args_list if c.args[0].startswith('UPDATE'))
            saved=json.loads(update.args[1][0]);self.assertNotIn('workerToken',saved);self.assertEqual(saved['autoRecoveryAttempts'],1)
            self.assertIn("status='running'",cur.execute.call_args_list[0].args[0])
    def test_limit_or_changed_thesis_fails_without_provider_dispatch(self):
        for options in ({'attempts':2},{'changed':True}):
            client,cur=self.client(**options)
            with patch('research_amendments.threading.Thread') as thread:
                r=client.post('/api/agent/recover-amendments')
                self.assertEqual(r.json['failed'],['job']);thread.assert_not_called()

    def test_live_owners_do_not_hide_abandoned_work_and_recovery_stays_bounded(self):
        client,cur=self.client()
        import copy
        template=cur.fetchall.return_value[0]
        cur.fetchall.return_value=[{**copy.deepcopy(template),'id':f'job-{i}'} for i in range(8)]
        baseline={'analysis':template['input']['baseline']}
        cur.fetchone.side_effect=[{'locked':False}]*3+[item for _ in range(3) for item in ({'locked':True},baseline)]
        with patch('research_amendments.threading.Thread') as thread:
            result=client.post('/api/agent/recover-amendments').json
            self.assertEqual(result['recovered'],['job-3','job-4','job-5'])
            self.assertEqual(thread.call_count,3)
            self.assertIn('LIMIT 20',cur.execute.call_args_list[0].args[0])
