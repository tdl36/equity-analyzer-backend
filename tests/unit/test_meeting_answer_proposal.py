import json
import unittest
from contextlib import contextmanager
from unittest.mock import Mock,patch
from flask import Flask
from research_amendments import create_blueprint,STAGE

WORK='11111111-1111-4111-8111-111111111111'
JOB='22222222-2222-4222-8222-222222222222'
class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.cur=Mock();self.cur.fetchall.return_value=[]
        @contextmanager
        def db(**kwargs):yield None,self.cur
        self.model=Mock(return_value={'changes':[]})
        self.app=Flask(__name__);self.app.register_blueprint(create_blueprint(db,self.model,lambda key:'synthetic-key'))
        self.client=self.app.test_client()
        self.ref={'id':WORK,'revision':2}
        self.evidence={'workId':WORK,'workRevision':2,'answer':{'id':1,'meeting_id':2,'meeting_date':'2026-01-01','question':'Outlook?','response_notes':'Recorded demand remains uncertain; no public guidance change was confirmed.'},'interpretation':'Analyst inference, not evidence','scope':'Recorded notes'}
        self.payload={'requestId':JOB,'workEvidence':self.ref}
        self.baseline={'thesis':{'summary':'Saved view'}}
    def test_submission_freezes_server_evidence_and_worker_reuses_existing_review_path(self):
        self.cur.fetchone.side_effect=[None,None,{'analysis':self.baseline}]
        with patch('meeting_answer_evidence.resolve',return_value=self.evidence),patch('research_amendments.threading.Thread') as thread:
            response=self.client.post('/api/research/amendments/ABT',json=self.payload)
            self.assertEqual(response.status_code,202)
            calls=[c for c in self.cur.execute.call_args_list if c.args[0].startswith('INSERT INTO mp_jobs')]
            frozen=json.loads(calls[0].args[1][-1]);self.assertEqual(frozen['meetingAnswerEvidence'],self.evidence)
            self.assertEqual(frozen['filenames'],[]);self.model.assert_not_called()
            worker=thread.call_args.kwargs
        self.cur.reset_mock();self.cur.fetchall.return_value=[]
        self.cur.fetchone.side_effect=[{'id':JOB},{'input':frozen,'result':{}}]
        @contextmanager
        def session(*args):yield True
        with patch('amendment_ownership.worker_session',session),patch('meeting_answer_evidence.resolve',return_value=self.evidence),patch('amendment_checkpoints.Checkpoints') as checkpoints,patch('meeting_answer_evidence.reserve_call') as reserve:
            checkpoints.return_value.load.return_value=None
            worker['target'](*worker['args'])
        reserve.assert_called_once()
        self.assertEqual(reserve.call_args.args[-1],'draft')
        self.model.assert_called_once()
        prompt=self.model.call_args.args[0]
        self.assertIn('not a verified transcript',prompt)
        self.assertIn('Analyst inference, not evidence',prompt)
        completion=[c for c in self.cur.execute.call_args_list if c.args[0].startswith('UPDATE mp_jobs SET status=%s')][-1]
        self.assertEqual(completion.args[1][0],'awaiting_approval')
        self.assertEqual(json.loads(completion.args[1][1])['meetingAnswerEvidence'],self.evidence)
    def test_same_request_replays_but_changed_reference_conflicts(self):
        existing={'stage':STAGE,'ticker':'ABT','input':{'filenames':[],'workEvidence':self.ref}}
        self.cur.fetchone.side_effect=None;self.cur.fetchone.return_value=existing
        with patch('research_amendments.threading.Thread') as thread:
            self.assertEqual(self.client.post('/api/research/amendments/ABT',json=self.payload).status_code,200)
            self.assertEqual(self.client.post('/api/research/amendments/ABT',json={**self.payload,'workEvidence':{**self.ref,'revision':3}}).status_code,409)
            thread.assert_not_called()
    def test_mixed_sources_rejected_before_database_or_model(self):
        for extra in ({'filenames':['source.pdf']},{'commandId':'command'},{'researchRunId':'run'}):
            self.assertEqual(self.client.post('/api/research/amendments/ABT',json={**self.payload,**extra}).status_code,400)
        self.cur.execute.assert_not_called();self.model.assert_not_called()
    def test_changed_answer_blocks_application_without_writes(self):
        job={'ticker':'ABT','status':'awaiting_approval','input':{'workEvidence':self.ref,'meetingAnswerEvidence':self.evidence,'baseline':self.baseline},'result':{'changes':[]}}
        self.cur.fetchone.side_effect=[job,{'analysis':self.baseline}]
        with patch('meeting_answer_evidence.resolve',side_effect=ValueError('Answer changed')):
            response=self.client.post('/api/research/amendment/'+JOB+'/decide',json={'action':'apply','acceptedIds':[]})
        self.assertEqual(response.status_code,409)
        self.assertFalse(any(c.args[0].startswith('UPDATE') for c in self.cur.execute.call_args_list))
