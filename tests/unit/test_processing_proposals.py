"""Synthetic isolated-Postgres approval handoff; never paid or production work."""
import base64
import copy
import json
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from research_assignments import child
from processing_proposals import assignment_id
from tests.unit.test_research_assignments import AssignmentPostgresTests, PG, P


@unittest.skipUnless((PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class ProcessingProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AssignmentPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_, cur):
            cur.execute('ALTER TABLE document_files ADD COLUMN id BIGSERIAL PRIMARY KEY')

    @classmethod
    def tearDownClass(cls):
        AssignmentPostgresTests.tearDownClass.__func__(cls)

    setUp = AssignmentPostgresTests.setUp
    invoke = AssignmentPostgresTests.invoke
    report = AssignmentPostgresTests.report

    def parent(self):
        with self.db(True) as (_, cur):
            cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result) VALUES(%s,'research_assignment',%s,'complete',%s::jsonb,%s::jsonb)", (self.ident, self.ticker, json.dumps({**P, 'ticker': self.ticker, 'outputs': [], 'collectionOnly': True}), json.dumps({'sources': [{'filename': 'one.txt', 'sha256': self.sha}]})))
        self.url = '/api/research/assignments/' + self.ident + '/processing-proposals'
        self.body = {'requestId': str(uuid.uuid4()), 'filenames': ['one.txt'], 'plan': {'researchPurpose': {'kind': 'event_note', 'event': {'title': 'Earnings', 'date': '2026-10-09'}}, 'outputs': ['summary_note']}}

    def save(self):
        self.parent()
        response = self.client.post(self.url, json=self.body)
        self.assertEqual(response.status_code, 201, response.json)
        return response.json

    def decision(self, proposal, action='approve', client=None):
        return (client or self.client).post('/api/research/processing-proposals/' + proposal['id'] + '/' + action, json={'revision': proposal['revision'], 'digest': proposal['digest']})

    def row(self, ident):
        with self.db() as (_, cur):
            cur.execute('SELECT * FROM mp_jobs WHERE id=%s', (ident,))
            return cur.fetchone()

    def test_save_read_and_exact_replay_never_dispatch(self):
        proposal = self.save()
        self.assertEqual(proposal['status'], 'pending')
        self.assertFalse(self.submits)
        self.assertIsNone(self.row(assignment_id(proposal['id'])))
        self.assertEqual(self.client.get(self.url).json['proposals'][0]['digest'], proposal['digest'])
        self.assertEqual(self.client.post(self.url, json=self.body).status_code, 200)
        changed = copy.deepcopy(self.body);changed['plan']['instruction'] = 'Different question'
        self.assertEqual(self.client.post(self.url, json=changed).status_code, 409)

    def test_approval_reuses_coordinator_and_preserves_parent(self):
        proposal = self.save();parent = self.row(self.ident)
        response = self.decision(proposal)
        self.assertEqual(response.status_code, 202, response.json)
        aid = assignment_id(proposal['id'])
        self.assertEqual(response.json['id'], aid)
        self.assertIsNone(self.row(child(aid, 'collection')))
        self.assertEqual(self.row(self.ident), parent)
        self.assertEqual(self.row(proposal['id'])['status'], 'dispatched')
        self.assertEqual(self.decision(proposal).status_code, 200)
        self.assertFalse(self.submits)
        with self.app.app_context(): self.co.advance(aid)
        self.assertEqual(len(self.submits), 1)
        research = next(iter(self.submits.values()))
        self.assertEqual(research['expectedOriginals'][0]['sha256'], self.sha)
        self.assertEqual(self.co.read(aid)['status'], 'complete')
        self.assertNotIn('thesisDraftId', self.co.read(aid)['result'])

    def test_changed_bytes_block_both_save_and_approval(self):
        proposal = self.save()
        with self.db(True) as (_, cur):
            cur.execute('UPDATE document_files SET file_data=%s WHERE ticker=%s', (base64.b64encode(b'changed').decode(), self.ticker))
        self.assertEqual(self.decision(proposal).status_code, 409)
        self.assertEqual(self.row(proposal['id'])['status'], 'pending')
        self.body['requestId'] = str(uuid.uuid4())
        self.assertEqual(self.client.post(self.url, json=self.body).status_code, 409)
        self.assertFalse(self.submits)

    def test_changed_permission_and_baseline_block_approval(self):
        proposal = self.save()
        with self.db(True) as (_, cur):
            cur.execute("UPDATE document_files SET metadata=%s::jsonb WHERE ticker=%s", (json.dumps({'usage': 'reference_only'}), self.ticker))
        self.assertEqual(self.decision(proposal).status_code, 409)
        with self.db(True) as (_, cur):
            cur.execute("UPDATE document_files SET metadata='{}'::jsonb WHERE ticker=%s", (self.ticker,))
            cur.execute("INSERT INTO portfolio_analyses(ticker,company,analysis) VALUES(%s,'Changed',%s::jsonb)", (self.ticker, json.dumps({'summary': {'investment_thesis': ['Changed thesis']}})))
        self.assertEqual(self.decision(proposal).status_code, 409)
        self.assertEqual(self.row(proposal['id'])['status'], 'pending')

    def test_cancelled_proposal_cannot_dispatch_or_be_revised(self):
        proposal = self.save()
        self.assertEqual(self.decision(proposal, 'cancel').status_code, 200)
        self.assertEqual(self.decision(proposal).status_code, 409)
        self.assertEqual(self.row(proposal['id'])['status'], 'cancelled')
        self.assertIsNone(self.row(assignment_id(proposal['id'])))

    def test_wrong_digest_or_changed_plan_is_rejected(self):
        proposal = self.save();bad = dict(proposal, digest='0'*64)
        self.assertEqual(self.decision(bad).status_code, 409)
        body = dict(proposal['plan'], requestId=assignment_id(proposal['id']), processingProposal={'id': proposal['id'], 'revision': 1, 'digest': proposal['digest']}, instruction='Tampered')
        self.assertEqual(self.client.post('/api/research/assignments', json=body).status_code, 409)
        body['instruction'] = '';body['requestId'] = str(uuid.uuid4())
        self.assertEqual(self.client.post('/api/research/assignments', json=body).status_code, 409)
        self.assertEqual(self.row(proposal['id'])['status'], 'pending')

    def test_concurrent_approvals_create_one_assignment(self):
        proposal = self.save()
        def approve(_):
            with self.app.test_client() as client:return self.decision(proposal, client=client).status_code
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(approve, range(2))), [200, 202])
        with self.db() as (_, cur):
            cur.execute("SELECT COUNT(*) AS n FROM mp_jobs WHERE ticker=%s AND stage='research_assignment'", (self.ticker,))
            self.assertEqual(cur.fetchone()['n'], 2)

    def test_output_and_source_boundaries(self):
        self.parent()
        for change in ({'filenames': []}, {'filenames': ['missing.txt']}, {'filenames': ['one.txt', 'one.txt']}, {'plan': {'outputs': ['summary_note']}}, {'plan': {**self.body['plan'], 'outputs': ['thesis']}}, {'plan': {**self.body['plan'], 'collectionOnly': True}}):
            with self.subTest(change=change):
                self.assertEqual(self.client.post(self.url, json={**self.body, **change}).status_code, 409)
        self.assertFalse(self.submits)

    def test_incomplete_or_generation_parent_is_not_a_source_only_handoff(self):
        self.parent()
        with self.db(True) as (_, cur):
            cur.execute("UPDATE mp_jobs SET status='running' WHERE id=%s", (self.ident,))
        self.assertEqual(self.client.get(self.url).status_code, 409)
        with self.db(True) as (_, cur):
            cur.execute("UPDATE mp_jobs SET status='complete',input=%s::jsonb WHERE id=%s", (json.dumps({**P, 'ticker': self.ticker}), self.ident))
        self.assertEqual(self.client.post(self.url, json=self.body).status_code, 409)

    def test_active_ticker_and_no_key_leave_proposal_pending(self):
        proposal = self.save()
        self.co.has_key = lambda: False
        self.assertEqual(self.decision(proposal).status_code, 400)
        self.co.has_key = lambda: True
        with self.db(True) as (_, cur):
            cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,'research_assignment',%s,'attention','{}')", (str(uuid.uuid4()), self.ticker))
        self.assertEqual(self.decision(proposal).status_code, 409)
        self.assertEqual(self.row(proposal['id'])['status'], 'pending')
