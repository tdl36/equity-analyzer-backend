import unittest
from meeting_assignment_verification import verify

class AssignmentVerificationTests(unittest.TestCase):
    def inputs(self):
        question=dict(question='What changed?',context='Reported revenue.',source='release.htm',follow_up_angle='What reverses this?',priority='high',source_filenames=['release.htm'])
        return [dict(prep_status='done',meeting_id='1',ticker='ABT',topic='Meeting',job_id='job'),
                dict(status='complete',result={'importedOriginals':1}),
                dict(status='complete',ticker='ABT',detail={'topic':'Meeting'},result={'markdown':'Recap','evidenceSnapshot':{'sources':[{'filename':'release.htm'}]}}),
                dict(meeting={'id':1,'ticker':'ABT'},documents=[{'filename':'release.htm'}],questionSet={'id':2,'topics':[{'topic':'Growth','questions':[question]}]}),
                dict(status='done',id='job',ticker='ABT',result={'questionSetId':2})]
    def test_complete_saved_chain_passes(self):
        self.assertEqual(verify(*self.inputs())['status'],'passed')
    def test_wrong_source_set_or_unsaved_questions_cannot_pass(self):
        values=self.inputs();values[3]['documents']=[{'filename':'other.htm'}]
        self.assertEqual(verify(*values)['status'],'incomplete')
        values=self.inputs();values[4]['result']['questionSetId']=3
        self.assertEqual(verify(*values)['status'],'incomplete')
    def test_pending_job_is_not_success(self):
        values=self.inputs();values[4]['status']='running'
        self.assertEqual(verify(*values)['status'],'incomplete')
