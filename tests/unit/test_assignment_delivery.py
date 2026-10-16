import unittest
from assignment_workspace import delivery_contract

class DeliveryTests(unittest.TestCase):
    def run_contract(self,**overrides):
        args=dict(payload={},command={'status':'done'},collection={'status':'complete'},source_reuse=False,prep=None,versions=[],reports=[],proposal=None,dispatch=None)
        return delivery_contract(**{**args,**overrides})
    def test_dispatched_is_not_delivered(self):
        self.assertEqual(self.run_contract()['state'],'in_progress')
        self.assertEqual(self.run_contract()['delivered'],1)
    def test_meeting_requires_saved_ready_version_and_finished_job(self):
        args=dict(payload={'meetingPrep':{'duration':60}},reports=[{'has_report':True}])
        self.assertEqual(self.run_contract(**args,prep={'status':'done'})['state'],'in_progress')
        self.assertEqual(self.run_contract(**args,prep={'status':'done'},versions=[{'status':'ready'}])['state'],'delivered')
        self.assertEqual(self.run_contract(**args,prep={'status':'failed','error':'Passage missing'})['state'],'attention')
    def test_proposal_dispatch_is_not_saved_proposal(self):
        args=dict(payload={'autoProposal':True},reports=[{'has_report':True}])
        self.assertEqual(self.run_contract(**args,dispatch={'status':'done'})['state'],'in_progress')
        self.assertEqual(self.run_contract(**args,proposal={'status':'awaiting_approval'})['state'],'delivered')
        self.assertEqual(self.run_contract(**args,dispatch={'status':'failed'})['state'],'attention')
    def test_unrequested_questions_not_required(self):
        self.assertEqual(self.run_contract(reports=[{'has_report':True}])['expected'],2)
    def test_failed_regeneration_does_not_hide_behind_old_output(self):
        result=self.run_contract(reports=[{'status':'failed','has_report':True}])
        self.assertEqual(result['state'],'attention')
    def test_saved_source_pack_does_not_wait_for_unrequested_recap(self):
        result=self.run_contract(source_reuse=True,payload={'meetingPrep':{'duration':60}},prep={'status':'done'},versions=[{'status':'ready'}])
        self.assertEqual(result['state'],'delivered')
        self.assertEqual(result['expected'],2)
    def test_blocked_dispatch_requires_attention(self):
        self.assertEqual(self.run_contract(payload={'autoProposal':True},dispatch={'status':'blocked'})['state'],'attention')

    def test_actual_collection_attention_and_auth_states_are_not_pending(self):
        for status in ('attention', 'needs_auth', 'needs_attention', 'stopped'):
            with self.subTest(status=status):
                result=self.run_contract(collection={'status':status,'issue':'Preserved collector reason'})
                self.assertEqual(result['state'],'attention')
                self.assertEqual(result['steps'][0]['state'],'attention')
                self.assertEqual(result['steps'][0]['detail'],'Preserved collector reason')
                self.assertEqual(result['delivered'],0)

    def test_newer_unfinished_recap_cannot_inherit_an_older_saved_report(self):
        for status in ('queued','running','pending'):
            with self.subTest(status=status):
                result=self.run_contract(reports=[{'status':status,'has_report':False},
                                                 {'status':'completed','has_report':True}])
                self.assertEqual(result['state'],'in_progress')
                self.assertEqual(result['steps'][1]['state'],'pending')
                self.assertIn('earlier report remains available',result['steps'][1]['detail'])
                self.assertEqual(result['delivered'],1)

    def test_latest_saved_recap_and_attention_remain_distinct(self):
        result=self.run_contract(reports=[{'status':'completed','has_report':True},
                                         {'status':'failed','has_report':False}])
        self.assertEqual(result['state'],'delivered')
        for status in ('attention','needs_auth','stopped'):
            result=self.run_contract(reports=[{'status':status,'has_report':True}])
            self.assertEqual(result['steps'][1]['state'],'attention')

    def test_latest_question_version_controls_delivery_without_losing_history(self):
        args=dict(source_reuse=True,payload={'meetingPrep':{'duration':60}},prep={'status':'done'})
        for status in ('pending','generating','failed','attention'):
            versions=[{'status':status},{'status':'ready'}]
            with self.subTest(status=status):
                result=self.run_contract(**args,versions=versions)
                expected='attention' if status in ('failed','attention') else 'pending'
                self.assertEqual(result['steps'][1]['state'],expected)
                self.assertEqual(result['delivered'],1)
                self.assertEqual(versions,[{'status':status},{'status':'ready'}])
        self.assertEqual(self.run_contract(**args,versions=[{'status':'ready'},{'status':'failed'}])['state'],'delivered')

    def test_missing_collector_issue_exposes_auth_or_command_failure(self):
        result=self.run_contract(collection={'status':'needs_auth'})
        self.assertIn('AlphaSense sign-in',result['steps'][0]['detail'])
        result=self.run_contract(collection=None,command={'status':'failed','error':'Command was not acknowledged'})
        self.assertEqual(result['state'],'attention')
        self.assertEqual(result['steps'][0]['detail'],'Command was not acknowledged')

    def test_missing_acknowledged_collection_is_unknown_not_waiting_or_complete(self):
        result=self.run_contract(collection=None,command={'status':'applied'},reports=[{'has_report':True}])
        self.assertEqual(result['state'],'unknown')
        self.assertEqual(result['steps'][0]['state'],'unknown')
        self.assertIn('cannot be inferred',result['steps'][0]['detail'])
        self.assertEqual(result['delivered'],1)
        self.assertEqual(self.run_contract(collection=None,command={'status':'queued'})['steps'][0]['state'],'pending')
