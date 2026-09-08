import unittest
from contextlib import contextmanager
from unittest.mock import Mock
from amendment_checkpoints import Checkpoints,identity

class CheckpointTests(unittest.TestCase):
    def checkpoint(self,row):
        cur=Mock();cur.fetchone.return_value=row
        @contextmanager
        def db(commit=False):yield None,cur
        return Checkpoints(db,'job'),cur
    def test_prompt_and_source_changes_invalidate_identity(self):
        first=identity('prompt',{'source':'one'})
        self.assertNotEqual(identity('prompt',{},'model-a'),identity('prompt',{},'model-b'))
        self.assertNotEqual(first,identity('prompt',{'source':'two'}))
        self.assertNotEqual(first,identity('changed',{'source':'one'}))
    def test_completed_stage_reused(self):
        cp,_=self.checkpoint({'status':'running','result':{'checkpoint':{'key':'same','draft':{'changes':[]}}}})
        self.assertEqual(cp.load('same','draft'),{'changes':[]})
        self.assertIsNone(cp.load('same','review'))
        with self.assertRaises(ValueError):cp.load('changed','draft')
    def test_late_cancelled_or_failed_worker_cannot_continue(self):
        for status in ('dismissed','failed','applied'):
            cp,cur=self.checkpoint({'status':status,'result':{}})
            with self.assertRaises(ValueError):cp.load('key','draft')
            with self.assertRaises(ValueError):cp.save('key','draft',{})
            self.assertFalse(any(c.args[0].startswith('UPDATE') for c in cur.execute.call_args_list))
    def test_conflicting_stage_never_overwrites_saved_work(self):
        cp,cur=self.checkpoint({'status':'running','result':{'checkpoint':{'key':'same','draft':{'old':1}}}})
        with self.assertRaises(ValueError):cp.save('same','draft',{'new':2})
        self.assertFalse(any(c.args[0].startswith('UPDATE') for c in cur.execute.call_args_list))
    def test_checkpoint_update_preserves_previous_stage(self):
        import json
        cp,cur=self.checkpoint({'status':'running','result':{'checkpoint':{'key':'same','draft':{'changes':[]}}}})
        cp.save('same','review',{'checks':[]})
        result=json.loads(cur.execute.call_args.args[1][0])
        self.assertEqual(result['checkpoint']['draft'],{'changes':[]})
        self.assertEqual(result['checkpoint']['review'],{'checks':[]})

class ResumeRouteTests(unittest.TestCase):
    def client(self,attempts=0,changed=False,other=False,locked=True):
        from flask import Flask
        from research_amendments import create_blueprint
        baseline={'thesis':{'summary':'saved'}}
        cur=Mock();cur.fetchone.side_effect=[{'ticker':'UNH'},
            {'ticker':'UNH','status':'failed','input':{'baseline':baseline,'filenames':['source.pdf'],'resumeAttempts':attempts},'result':{'checkpoint':{'key':'saved'}}},{'locked':locked},
            {'id':'other'} if other else None,{'analysis':{'thesis':{'summary':'changed'}} if changed else baseline}]
        @contextmanager
        def db(commit=False):yield None,cur
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,Mock(),lambda key:'test-key'))
        return app.test_client(),cur
    def test_resume_queues_same_job_and_increments_bound(self):
        import json
        from unittest.mock import patch
        client,cur=self.client()
        with patch('research_amendments.threading.Thread') as thread:
            response=client.post('/api/research/amendment/job/resume',json={})
            self.assertEqual(response.status_code,202)
            self.assertEqual(thread.call_args.kwargs['args'][0],'job')
            update=next(c for c in cur.execute.call_args_list if c.args[0].startswith('UPDATE'))
            self.assertEqual(json.loads(update.args[1][0])['resumeAttempts'],1)
    def test_exhausted_changed_baseline_or_other_active_proposal_blocks(self):
        from unittest.mock import patch
        for kwargs in ({'attempts':2},{'changed':True},{'other':True},{'locked':False}):
            client,cur=self.client(**kwargs)
            with patch('research_amendments.threading.Thread') as thread:
                self.assertEqual(client.post('/api/research/amendment/job/resume',json={}).status_code,409)
                thread.assert_not_called()
