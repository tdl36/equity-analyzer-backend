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
