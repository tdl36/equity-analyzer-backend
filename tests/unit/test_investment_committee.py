import copy
import unittest
from investment_committee import generate,ROLES
from test_company_research import SOURCE,response
import test_company_research as research_tests

class CommitteeTests(unittest.TestCase):
    def answer(self,prompt,tokens,stage):
        self.calls.append((prompt,tokens,stage))
        if stage.startswith('assessment-'):
            role=stage.removeprefix('assessment-')
            # Previous assessments must not leak into independent initial prompts.
            self.assertNotIn('ASSESSMENTS:',prompt)
            self.assertNotIn('ROLE-SPECIFIC-OUTPUT',prompt)
            r=response([(role,dict(ROLES)[role])]);r['sections'][0]['claims'][0]['statement']='ROLE-SPECIFIC-OUTPUT '+role
            return r
        if stage.startswith('review-'):
            return {'findings':[{'claimId':stage.removeprefix('review-')+':0','status':'supported','reason':'Synthetic source support'}]}
        if stage=='challenges':return {'challenges':[{'claimIds':['upside:0','downside:0'],'question':'What resolves the differing interpretations?'}]}
        return {'responses':[{'challengeId':'challenge-0','status':'contested','reason':'Evidence remains incomplete','nextTest':'Obtain next reported quarter','proposedChange':'No change until reviewed'}]}
    def test_independent_passes_preserved_dissent_and_bounded_rounds(self):
        self.calls=[];saved=[]
        result=generate({},[SOURCE],{'revision':3},self.answer,lambda s:saved.append(copy.deepcopy(s)),lambda:None)
        self.assertEqual(len(self.calls),12)
        self.assertTrue(all(tokens<=6000 for _,tokens,_ in self.calls))
        self.assertEqual([s['id'] for s in result['sections']],[r for r,_ in ROLES])
        self.assertEqual(result['responses'][0]['status'],'contested')
        self.assertEqual(result['sections'][1]['claims'][0]['statement'],'ROLE-SPECIFIC-OUTPUT upside')
        self.assertNotIn('inFlight',result)
        self.assertEqual(generate(result,[SOURCE],{},lambda *a:self.fail('Repeated paid call'),lambda s:None,lambda:None),result)
    def test_unknown_outcome_requires_ack_and_resumes_only_unfinished_stage(self):
        self.calls=[];saved=[]
        def fail(prompt,tokens,stage):
            if stage=='assessment-downside':raise TimeoutError()
            return self.answer(prompt,tokens,stage)
        with self.assertRaises(TimeoutError):generate({},[SOURCE],{},fail,lambda s:saved.append(copy.deepcopy(s)),lambda:None)
        checkpoint=saved[-1]
        self.assertEqual(checkpoint['inFlight'],'assessment-downside')
        with self.assertRaises(ValueError):generate(checkpoint,[SOURCE],{},self.answer,lambda s:None,lambda:None)
        checkpoint.pop('inFlight');self.calls=[]
        generate(checkpoint,[SOURCE],{},self.answer,lambda s:None,lambda:None)
        self.assertNotIn('assessment-lead',[x[2] for x in self.calls])
    def test_invalid_challenge_cannot_reference_invented_evidence(self):
        self.calls=[]
        def bad(prompt,tokens,stage):
            if stage=='challenges':return {'challenges':[{'claimIds':['invented:0'],'question':'Invented'}]}
            return self.answer(prompt,tokens,stage)
        with self.assertRaises(ValueError):generate({},[SOURCE],{},bad,lambda s:None,lambda:None)
    def test_stop_or_budget_check_prevents_paid_call(self):
        def stop():raise ValueError('Budget reached')
        with self.assertRaises(ValueError):generate({},[SOURCE],{},lambda *a:self.fail('Paid call'),lambda s:None,stop)

class CommitteeApiTests(unittest.TestCase):
    committee=True
    def setUp(self):
        self.tables=[];research_tests.ApiTests.setUp(self)
    payload=research_tests.ApiTests.payload
    def test_separate_durable_ledger_idempotency_and_recovery(self):
        p=self.payload();url='/api/research/committee/ABC'
        self.assertEqual(self.client.post(url,json=p).status_code,202)
        self.assertTrue(self.client.post(url,json=p).json['replayed'])
        row=self.rows[p['requestId']]
        self.assertEqual(row['version'],'investment-committee-v1')
        self.assertTrue(any('INSERT INTO investment_committee_runs' in sql for sql in self.tables))
        row.update(status='running',state={'completed':['assessment-lead'],'inFlight':'assessment-upside'})
        self.revision=3
        self.assertTrue(self.client.get('/api/research/committee-run/'+p['requestId']).json['baselineStale'])
        resume='/api/research/committee-run/'+p['requestId']+'/resume'
        self.assertEqual(self.client.post(resume,json={}).status_code,409)
        self.assertEqual(self.client.post(resume,json={'acknowledgeRetry':True}).status_code,200)
        self.assertEqual(row['state']['completed'],['assessment-lead'])
    def test_restricted_sources_stale_baseline_and_duplicate_jobs_block_dispatch(self):
        p=self.payload();url='/api/research/committee/ABC'
        self.restricted=True
        self.assertEqual(self.client.post(url,json=p).status_code,409)
        self.restricted=False
        self.assertEqual(self.client.post(url,json={**p,'revision':0}).status_code,409)
        self.assertEqual(self.thread.call_count,0)
        self.assertEqual(self.client.post(url,json=p).status_code,202)
        self.assertEqual(self.client.post(url,json=self.payload()).status_code,409)
        row=self.rows[p['requestId']];row['status']='complete'
        self.assertEqual(self.client.post('/api/research/committee-run/'+p['requestId']+'/resume',json={}).status_code,409)
