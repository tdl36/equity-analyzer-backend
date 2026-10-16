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
        self.assertNotIn('sourceEvidence',research)  # Legacy frozen proposal retains its old request contract.
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

    def extra(self, filename='annual-2024.txt', raw=b'Historical foundation', ticker=None, metadata=None):
        with self.db(True) as (_, cur):
            cur.execute('INSERT INTO document_files(ticker,filename,file_data,file_type,metadata) VALUES(%s,%s,%s,%s,%s::jsonb) RETURNING id', (ticker or self.ticker, filename, base64.b64encode(raw).decode(), 'txt', json.dumps(metadata or {'published':'2024-03-01'})))
            return str(cur.fetchone()['id'])

    def collected_pair(self, raw):
        import hashlib
        self.extra(filename='two.txt',raw=raw)
        self.body['filenames']=['one.txt','two.txt']
        with self.db(True) as (_,cur):
            cur.execute('UPDATE mp_jobs SET result=%s::jsonb WHERE id=%s',(json.dumps({'sources':[{'filename':'one.txt','sha256':self.sha},{'filename':'two.txt','sha256':hashlib.sha256(raw).hexdigest()}]}),self.ident))

    def test_collected_only_duplicate_bytes_block_new_save_without_optional_field(self):
        self.parent();self.collected_pair(self.raw);before=self.row(self.ident)
        response=self.client.post(self.url,json=self.body)
        self.assertEqual(response.status_code,409,response.json);self.assertIn('identical bytes',response.json['error'])
        self.assertIsNone(self.row(self.body['requestId']));self.assertEqual(self.row(self.ident),before);self.assertFalse(self.submits)
        with self.db() as (_,cur):
            cur.execute('SELECT filename,file_data FROM document_files WHERE ticker=%s ORDER BY filename',(self.ticker,))
            rows=list(cur.fetchall());self.assertEqual([r['filename'] for r in rows],['one.txt','two.txt'])
            self.assertEqual(rows[0]['file_data'],rows[1]['file_data'])

    def test_collected_only_distinct_originals_remain_eligible_without_optional_field(self):
        self.parent();self.collected_pair(b'Different evidence')
        response=self.client.post(self.url,json=self.body)
        self.assertEqual(response.status_code,201,response.json);self.assertEqual(len(response.json['sources']),2)
        self.assertFalse(self.submits)

    def test_existing_frozen_duplicate_proposal_exact_replay_is_preserved(self):
        from processing_proposals import digest
        proposal=self.save();stored=self.row(proposal['id']);frozen=stored['input']
        self.collected_pair(self.raw)
        extra_id=None
        with self.db() as (_,cur):
            cur.execute("SELECT id FROM document_files WHERE ticker=%s AND filename='two.txt'",(self.ticker,));extra_id=str(cur.fetchone()['id'])
        frozen['sources'].append({**frozen['sources'][0],'filename':'two.txt','reference':'charlie:thesis_'+extra_id,'documentId':extra_id})
        frozen['intentHash']=digest({'collectionId':self.ident,'plan':frozen['plan'],'filenames':sorted(self.body['filenames'])})
        frozen.pop('digest');frozen['digest']=digest(frozen)
        with self.db(True) as (_,cur):cur.execute('UPDATE mp_jobs SET input=%s::jsonb WHERE id=%s',(json.dumps(frozen),proposal['id']))
        before=self.row(proposal['id']);response=self.client.post(self.url,json=self.body)
        self.assertEqual(response.status_code,200,response.json);self.assertEqual(response.json['digest'],frozen['digest'])
        self.assertEqual(self.row(proposal['id']),before);self.assertFalse(self.submits)

    def test_supplemental_history_freezes_roles_and_dispatches_exact_pack(self):
        self.parent();extra=self.extra()
        self.body['supplementalIds']=[extra]
        response=self.client.post(self.url,json=self.body)
        self.assertEqual(response.status_code,201,response.json)
        proposal=response.json
        self.assertEqual(len(proposal['sources']),2)
        self.assertEqual(proposal['sourceContext'][extra]['role'],'supplemental')
        self.assertEqual(proposal['sourceContext'][extra]['publishedDate'],'2024-03-01')
        self.assertFalse(proposal['sourceContext'][extra]['referencedBySavedThesis'])
        self.assertEqual(self.decision(proposal).status_code,202)
        aid=assignment_id(proposal['id']);self.report_status='running'
        with self.app.app_context():self.co.advance(aid)
        research=next(iter(self.submits.values()))
        self.assertEqual(len(research['expectedOriginals']),2)
        self.assertIn('supplemental stored history',research['question'])
        evidence=research['sourceEvidence'];self.assertEqual(evidence['version'],1)
        historical=next(s for s in evidence['sources'] if s['id']==extra)
        self.assertEqual(historical['role'],'supplemental');self.assertEqual(historical['publishedDate'],'2024-03-01')
        self.assertEqual(self.row(proposal['id'])['input']['sourceEvidence'],evidence)
        before=copy.deepcopy(research)
        with self.db(True) as (_,cur):cur.execute("UPDATE document_files SET metadata='{}'::jsonb WHERE id=%s",(extra,))
        with self.app.app_context():self.co.advance(aid)
        self.assertEqual(next(iter(self.submits.values())),before)
        self.assertEqual(self.co.read(aid)['input']['processingSourceContext'],proposal['sourceContext'])
        self.assertIsNone(self.row(child(aid,'collection')))

    def test_supplements_reject_other_company_overlap_and_duplicate_bytes(self):
        self.parent()
        other=self.extra(ticker='DIFFERENT')
        copy_id=self.extra(raw=self.raw)
        with self.db() as (_,cur):
            cur.execute('SELECT id FROM document_files WHERE ticker=%s AND filename=%s',(self.ticker,'one.txt'));collected=str(cur.fetchone()['id'])
        for ids in ([other],[collected],[copy_id],['-1'],[True],['1','1']):
            with self.subTest(ids=ids):
                response=self.client.post(self.url,json={**self.body,'supplementalIds':ids})
                self.assertEqual(response.status_code,409,response.json)
        self.assertFalse(self.submits)

    def test_supplemental_bytes_and_permissions_rechecked_before_dispatch(self):
        self.parent();extra=self.extra()
        self.body['supplementalIds']=[extra]
        proposal=self.client.post(self.url,json=self.body).json
        with self.db(True) as (_,cur):
            cur.execute('UPDATE document_files SET file_data=%s WHERE id=%s',(base64.b64encode(b'new historical source').decode(),int(extra)))
        self.assertEqual(self.decision(proposal).status_code,409)
        self.assertEqual(self.row(proposal['id'])['status'],'pending')
        self.body['requestId']=str(uuid.uuid4())
        proposal=self.client.post(self.url,json=self.body).json
        with self.db(True) as (_,cur):cur.execute('UPDATE document_files SET metadata=%s::jsonb WHERE id=%s',(json.dumps({'usage':'reference_only'}),int(extra)))
        self.assertEqual(self.decision(proposal).status_code,409)

    def test_library_inventory_is_scoped_paged_and_body_free(self):
        self.parent();self.extra(ticker='OTHER')
        with self.db(True) as (_,cur):
            cur.execute("INSERT INTO document_files(ticker,filename,metadata) SELECT %s,'source-'||n||'.txt',%s::jsonb FROM generate_series(1,201) n",(self.ticker,json.dumps({'published':'2023-12-31','private':'SECRET_FIXTURE','usage':'reference_only'})))
        url=self.url.replace('processing-proposals','processing-library')
        response=self.client.get(url);self.assertEqual(response.status_code,200)
        data=response.json;self.assertEqual(len(data['documents']),200)
        self.assertEqual((data['ticker'],data['collectionId'],data['after']),(self.ticker,self.ident,None))
        self.assertEqual(response.headers['Cache-Control'],'private, no-store')
        self.assertNotIn('SECRET_FIXTURE',response.text)
        self.assertNotIn('file_data',response.text)
        self.assertTrue(all(not r['eligible'] for r in data['documents']))
        more=self.client.get(url+'?after='+data['nextCursor']).json
        self.assertEqual(len(more['documents']),1)
        self.assertEqual((more['ticker'],more['collectionId'],more['after']),(self.ticker,self.ident,data['nextCursor']))
        self.assertEqual([int(r['id'][7:]) for r in data['documents']],sorted([int(r['id'][7:]) for r in data['documents']],reverse=True))
        self.assertLess(int(more['documents'][0]['id'][7:]),int(data['nextCursor']))
        for query in ('?after=1&after=2','?ticker=OTHER','?collectionId=other'):
            self.assertEqual(self.client.get(url+query).status_code,400)
        self.assertIsNone(more['nextCursor'])
        self.assertEqual(self.client.get(url+'?after=99999999999999999999').status_code,400)
        self.assertFalse(self.submits)

    def test_saved_thesis_reference_requires_exact_bytes_not_filename(self):
        self.parent();extra=self.extra()
        from processing_proposal_sources import context
        with self.db() as (_,cur):
            rows=context(cur,self.ticker,[{'documentId':extra,'filename':'annual-2024.txt','sha256':'a'*64}],[],{'sourceRegister':[{'filename':'annual-2024.txt','sha256':'b'*64}]})
            self.assertFalse(rows[extra]['referencedBySavedThesis'])
            rows=context(cur,self.ticker,[{'documentId':extra,'filename':'annual-2024.txt','sha256':'a'*64}],[],{'sourceRegister':[{'filename':'different-name.txt','sha256':'a'*64}]})
            self.assertTrue(rows[extra]['referencedBySavedThesis'])

    def note_hooks(self):
        self.note_calls = []
        self.note_id = str(uuid.uuid4())
        self.co.lab_note_defaults = lambda: {'modelKey': 'fixture-v1', 'promptTemplate': 'Holistic fixture instructions'}
        def dispatch(ident, plan, pack):
            self.note_calls.append(copy.deepcopy(plan['labNotePlan']))
            return {'noteId': self.note_id, 'status': 'draft', 'revision': 1}
        self.co.lab_note_dispatch = dispatch
        self.co.lab_note_stop = lambda cur, ident: None

    def test_full_note_settings_freeze_at_save_and_replay_without_defaults(self):
        self.parent();self.note_hooks()
        self.body['plan'].update(labFullNote={'allowPdfText': False}, instruction='Explain guidance changes')
        response = self.client.post(self.url, json=self.body)
        self.assertEqual(response.status_code, 201, response.json)
        proposal = response.json
        frozen = self.row(proposal['id'])['input']['labNotePlan']
        self.assertIn('standalone professional event debrief', frozen['promptTemplate'])
        self.assertIn('Explain guidance changes', frozen['promptTemplate'])
        self.assertNotIn('Holistic fixture', frozen['promptTemplate'])
        self.assertNotIn('promptTemplate', proposal['fullNote'])
        self.assertEqual(proposal['fullNote']['maxCalls'], 8)
        self.assertFalse(self.note_calls);self.assertFalse(self.submits)
        self.co.lab_note_defaults = lambda: (_ for _ in ()).throw(AssertionError('Mutable defaults consulted'))
        self.assertEqual(self.client.post(self.url, json=self.body).json, proposal)
        approved = self.decision(proposal)
        self.assertEqual(approved.status_code, 202, approved.json)
        aid = assignment_id(proposal['id'])
        self.assertEqual(self.co.read(aid)['input']['labNotePlan'], frozen)
        with self.app.app_context(): self.co.advance(aid)
        self.assertEqual(self.note_calls, [frozen])
        self.assertEqual(self.co.read(aid)['result']['labNoteId'], self.note_id)
        self.assertEqual(self.co.read(aid)['status'], 'complete')
        self.assertNotIn('thesisDraftId', self.co.read(aid)['result'])
        self.assertEqual(self.decision(proposal).status_code, 200)
        self.assertEqual(len(self.submits), 1)

    def test_full_note_missing_integration_preserves_pending_approval(self):
        self.parent();self.body['plan']['labFullNote'] = {'allowPdfText': False}
        self.assertEqual(self.client.post(self.url, json=self.body).status_code, 409)
        self.note_hooks()
        proposal = self.client.post(self.url, json=self.body).json
        self.co.lab_note_dispatch = None
        self.assertEqual(self.decision(proposal).status_code, 409)
        self.assertEqual(self.row(proposal['id'])['status'], 'pending')
        self.assertIsNone(self.row(assignment_id(proposal['id'])))
        self.assertFalse(self.submits)

    def test_thesis_note_scope_and_changed_configuration_need_new_proposal(self):
        self.parent();self.note_hooks()
        self.body['plan'].update(researchPurpose={'kind': 'thesis_review'}, labFullNote={'allowPdfText': True})
        first = self.client.post(self.url, json=self.body).json
        frozen = self.row(first['id'])['input']['labNotePlan']
        self.assertIn('Holistic fixture instructions', frozen['promptTemplate'])
        self.assertIn('longitudinal thesis review', frozen['promptTemplate'])
        self.co.lab_note_defaults = lambda: {'modelKey': 'fixture-v2', 'promptTemplate': 'New instructions'}
        self.body['requestId'] = str(uuid.uuid4())
        second = self.client.post(self.url, json=self.body).json
        self.assertNotEqual(first['fullNote']['configDigest'], second['fullNote']['configDigest'])
        tampered = dict(first['plan'], requestId=assignment_id(first['id']), processingProposal={'id': first['id'], 'revision': 1, 'digest': first['digest']}, labFullNote={'allowPdfText': False})
        self.assertEqual(self.client.post('/api/research/assignments', json=tampered).status_code, 409)
        self.assertEqual(self.row(first['id'])['status'], 'pending')

    def test_meeting_and_full_note_reuse_selected_report_without_collection(self):
        self.parent();self.note_hooks()
        self.body['plan'].update(outputs=['summary_note', 'meeting_prep'], meeting={'date': '2026-10-20', 'type': 'expert'}, labFullNote={'allowPdfText': False})
        original_report = self.report
        def report():
            run = original_report();question = 'What supports the revenue outlook?'
            run['state']['report']['diligence_questions'] = [question]
            run['state']['citations']['/diligence_questions/0'] = {**copy.deepcopy(run['state']['citations']['/summary/one_liner']), 'statement': question}
            return run
        self.report = report
        meetings = [];original_invoke = self.co.invoke
        def invoke(endpoint, method, body, params):
            if endpoint == 'research_meeting_handoff.handoff':
                meetings.append(copy.deepcopy(body))
                return {'meetingId': 45, 'questionSetId': 46}, 200
            return original_invoke(endpoint, method, body, params)
        self.co.invoke = invoke
        response = self.client.post(self.url, json=self.body)
        self.assertEqual(response.status_code, 201, response.json)
        proposal = response.json
        self.assertFalse(meetings);self.assertFalse(self.note_calls)
        approved = self.decision(proposal);self.assertEqual(approved.status_code, 202, approved.json)
        aid = assignment_id(proposal['id'])
        with self.app.app_context(): self.co.advance(aid)
        row = self.co.read(aid)
        self.assertEqual(row['status'], 'complete', row.get('error'))
        self.assertEqual(len(meetings), 1);self.assertEqual(len(self.submits), 1)
        self.assertEqual(len(self.note_calls), 1)
        self.assertEqual(row['result']['meetingId'], 45)
        self.assertIsNone(self.row(child(aid, 'collection')))

    def test_source_readiness_is_frozen_without_text_or_paid_calls(self):
        self.parent();self.note_hooks()
        self.body['plan']['labFullNote'] = {'allowPdfText': False}
        proposal = self.client.post(self.url, json=self.body).json
        readiness = proposal['fullNote']['sourcePreparation']
        self.assertEqual(readiness['batchCount'], 1)
        self.assertEqual(readiness['sourcePack'][0]['sha256'], self.sha)
        self.assertEqual(len(readiness['warnings']), 1)
        self.assertNotIn('batches', readiness)
        self.assertFalse(self.submits);self.assertFalse(self.note_calls)
        from unittest.mock import patch
        with patch('notegen.plan_batches', side_effect=AssertionError('Replay planned sources')):
            self.assertEqual(self.client.post(self.url, json=self.body).status_code, 200)

    def test_changed_extraction_blocks_approval_and_queued_research(self):
        from unittest.mock import patch
        self.parent();self.note_hooks();self.body['plan']['labFullNote'] = {'allowPdfText': False}
        proposal = self.client.post(self.url, json=self.body).json
        def changed(docs):
            return [[dict(d, send_as='file', extracted_text='Changed extraction, same original') for d in docs]]
        with patch('notegen.plan_batches', side_effect=changed):
            self.assertEqual(self.decision(proposal).status_code, 409)
        self.assertEqual(self.row(proposal['id'])['status'], 'pending')
        self.assertEqual(self.decision(proposal).status_code, 202)
        aid = assignment_id(proposal['id'])
        with patch('notegen.plan_batches', side_effect=changed), self.app.app_context():
            self.co.advance(aid)
        self.assertEqual(self.co.read(aid)['status'], 'attention')
        self.assertFalse(self.submits);self.assertFalse(self.note_calls)
        self.assertIn('representation changed', self.co.read(aid)['error'])

    def test_unreadable_and_over_batch_sources_do_not_save_or_dispatch(self):
        from unittest.mock import patch
        self.parent();self.note_hooks();self.body['plan']['labFullNote'] = {'allowPdfText': False}
        extra = self.extra('broken.pdf', b'not a PDF')
        self.body['supplementalIds'] = [extra]
        result = self.client.post(self.url, json=self.body)
        self.assertEqual(result.status_code, 409)
        self.assertIn('broken.pdf', result.json['error'])
        self.assertIsNone(self.row(self.body['requestId']))
        self.body['supplementalIds'] = []
        with patch('notegen.plan_batches', return_value=[]):
            result = self.client.post(self.url, json=self.body)
            self.assertEqual(result.status_code, 409)
            self.assertIn('no readable', result.json['error'])
        with patch('notegen.plan_batches', return_value=[[]]*9):
            result = self.client.post(self.url, json=self.body)
            self.assertEqual(result.status_code, 409)
            self.assertIn('needs 9 batches', result.json['error'])
        self.assertIsNone(self.row(self.body['requestId']))
        self.assertFalse(self.submits);self.assertFalse(self.note_calls)

    def test_old_full_note_proposals_keep_their_frozen_contract(self):
        self.parent();self.note_hooks();self.body['plan']['labFullNote'] = {'allowPdfText': False}
        proposal = self.client.post(self.url, json=self.body).json
        # Synthetic T310 row: representation policy existed, preparation did not.
        from processing_proposals import digest, public
        frozen = self.row(proposal['id'])['input']
        frozen['labNotePlan'].pop('sourcePreparation');frozen.pop('digest')
        frozen['digest'] = digest(frozen)
        with self.db(True) as (_, cur):
            cur.execute('UPDATE mp_jobs SET input=%s::jsonb WHERE id=%s', (json.dumps(frozen), proposal['id']))
        proposal = public(self.row(proposal['id']))
        self.assertEqual(self.decision(proposal).status_code, 202)
        with self.app.app_context(): self.co.advance(assignment_id(proposal['id']))
        self.assertEqual(len(self.submits), 1)
        self.assertNotIn('sourcePreparation', self.note_calls[0])

    def test_coverage_uses_frozen_metadata_without_refreshing_thesis_claims(self):
        self.parent();extra=self.extra()
        self.body['supplementalIds']=[extra]
        proposal=self.client.post(self.url,json=self.body).json
        coverage=proposal['coverage']
        self.assertEqual(coverage['roles'],{'collection':1,'supplemental':1,'unknown':0})
        self.assertEqual(coverage['publication']['beforeWindow'],1)
        self.assertEqual(coverage['publication']['unknown'],1)
        with self.db(True) as (_,cur):
            cur.execute("UPDATE document_files SET metadata=%s::jsonb WHERE ticker=%s",(json.dumps({'published':'2026-10-10','sourceKind':'transcript'}),self.ticker))
        reread=self.client.get(self.url).json['proposals'][0]
        self.assertEqual(reread['coverage'],coverage)
        self.assertEqual(reread['digest'],proposal['digest'])
        self.assertEqual(coverage['incorporation'],'not_assessed')
        self.assertFalse(self.submits)

    def test_collected_metadata_inventory_is_scoped_read_only_and_preserves_receipts(self):
        self.parent();before=self.row(self.ident)
        self.extra(filename='outside.txt')
        self.extra(filename='one.txt',ticker='OTHER',metadata={'publisher':'Wrong company'})
        with self.db(True) as (_,cur):
            cur.execute('UPDATE document_files SET metadata=%s::jsonb WHERE ticker=%s AND filename=%s',(json.dumps({'published':'2026-10-02','publisher':'Company','source':'alphasense_collection','sourceKind':'press-release','sha256':self.sha,'privateBody':'Must not expose'}),self.ticker,'one.txt'))
        url=self.url.replace('processing-proposals','processing-sources')
        r=self.client.get(url);self.assertEqual(r.status_code,200,r.json)
        d=r.json;self.assertEqual((d['ticker'],d['collectionId']),(self.ticker,self.ident))
        self.assertEqual(len(d['documents']),1);doc=d['documents'][0]
        self.assertEqual((doc['filename'],doc['receiptSha256'],doc['publishedDate'],doc['sourceKind']),('one.txt',self.sha,'2026-10-02','press-release'))
        self.assertEqual(doc['availability'],'available');self.assertRegex(doc['id'],r'^thesis_[0-9]+$')
        self.assertNotIn('file_data',doc);self.assertNotIn('privateBody',str(d));self.assertIn('no-store',r.headers['Cache-Control'])
        self.assertEqual(self.row(self.ident),before);self.assertFalse(self.submits)
        self.assertEqual(self.client.get(url+'?after=1').status_code,409)
        self.assertEqual(self.client.get(url.replace(self.ident,str(uuid.uuid4()))).status_code,409)

    def test_collected_inventory_missing_restricted_and_changed_metadata_are_not_verified_bytes(self):
        self.parent();url=self.url.replace('processing-proposals','processing-sources')
        with self.db(True) as (_,cur):
            cur.execute('UPDATE document_files SET metadata=%s::jsonb WHERE ticker=%s',(json.dumps({'usage':'reference_only','published':'not-a-date'}),self.ticker))
        doc=self.client.get(url).json['documents'][0]
        self.assertEqual(doc['availability'],'restricted');self.assertIsNone(doc['publishedDate'])
        with self.db(True) as (_,cur):
            cur.execute('DELETE FROM document_files WHERE ticker=%s',(self.ticker,))
        doc=self.client.get(url).json['documents'][0]
        self.assertEqual(doc['availability'],'missing');self.assertIsNone(doc['id']);self.assertEqual(doc['receiptSha256'],self.sha)
        self.assertFalse(self.submits)

    def test_collected_inventory_rejects_duplicate_receipts_and_unfinished_parent(self):
        self.parent();url=self.url.replace('processing-proposals','processing-sources')
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET result=%s::jsonb WHERE id=%s",(json.dumps({'sources':[{'filename':'one.txt','sha256':self.sha}]*2}),self.ident))
        self.assertEqual(self.client.get(url).status_code,409)
        with self.db(True) as (_,cur):cur.execute("UPDATE mp_jobs SET status='running' WHERE id=%s",(self.ident,))
        self.assertEqual(self.client.get(url).status_code,409)
        self.assertFalse(self.submits)

    def test_proposal_history_pages_and_exact_recovery_are_scoped_readonly(self):
        proposal = self.save()
        frozen = self.row(proposal['id'])['input']
        ids = [proposal['id']] + [str(uuid.uuid4()) for _ in range(55)]
        with self.db(True) as (_, cur):
            cur.execute("UPDATE mp_jobs SET created_at='2026-01-01' WHERE id=%s", (proposal['id'],))
            for ident in ids[1:]:
                cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result,created_at) VALUES(%s,'processing_proposal',%s,'cancelled',%s::jsonb,'{}','2026-01-01')", (ident,self.ticker,json.dumps(frozen)))
        first = self.client.get(self.url)
        self.assertEqual(first.status_code,200,first.json)
        self.assertIn('no-store',first.headers['Cache-Control'])
        ordered = sorted(ids,reverse=True)
        self.assertEqual([r['id'] for r in first.json['proposals']],ordered[:50])
        self.assertEqual(first.json['nextCursor'],ordered[49])
        second = self.client.get(self.url+'?after='+first.json['nextCursor'])
        self.assertEqual([r['id'] for r in second.json['proposals']],ordered[50:])
        self.assertIsNone(second.json['nextCursor'])
        exact = self.client.get(self.url+'?requestId='+ordered[-1])
        self.assertEqual([r['id'] for r in exact.json['proposals']],[ordered[-1]])
        self.assertIsNone(exact.json['nextCursor'])
        self.assertFalse(self.submits)
        self.assertIsNone(self.row(assignment_id(proposal['id'])))

    def test_proposal_history_rejects_bad_or_foreign_cursors_and_hides_foreign_receipts(self):
        proposal = self.save()
        foreign = str(uuid.uuid4())
        frozen = dict(self.row(proposal['id'])['input'],collectionId=str(uuid.uuid4()))
        with self.db(True) as (_,cur):
            cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result) VALUES(%s,'processing_proposal',%s,'pending',%s::jsonb,'{}')",(foreign,self.ticker,json.dumps(frozen)))
        for query in ['after=bad','after=','unknown=1','after='+foreign,'after='+proposal['id']+'&requestId='+proposal['id'],'requestId='+proposal['id']+'&requestId='+proposal['id']]:
            with self.subTest(query=query):self.assertEqual(self.client.get(self.url+'?'+query).status_code,400)
        self.assertEqual(self.client.get(self.url+'?requestId='+foreign).json['proposals'],[])
        self.assertEqual(self.client.get(self.url+'?requestId='+self.ident).json['proposals'],[])
        self.assertEqual(self.row(proposal['id'])['status'],'pending')
        self.assertFalse(self.submits)

    def test_selected_library_metadata_is_exact_bounded_scoped_and_read_only(self):
        self.parent();old=self.extra();foreign=self.extra(ticker='OTHER')
        with self.db(True) as (_,cur):
            cur.execute("SELECT id::text AS id FROM document_files WHERE ticker=%s AND filename='one.txt'",(self.ticker,));collected=cur.fetchone()['id']
            cur.execute("UPDATE document_files SET metadata=%s::jsonb WHERE id=%s",(json.dumps({'usage':'reference_only','private':'SECRET_FIXTURE'}),int(old)))
            cur.execute("INSERT INTO document_files(ticker,filename,metadata) SELECT %s,'new-'||n||'.txt','{}' FROM generate_series(1,220)n",(self.ticker,))
            cur.execute('SELECT id,metadata,file_data FROM document_files ORDER BY id');before=cur.fetchall()
        url=self.url.replace('processing-proposals','processing-library')
        selected=[old,foreign,collected,'9223372036854775807']
        response=self.client.get(url,query_string={'selected':','.join(selected)})
        self.assertEqual(response.status_code,200);d=response.json
        self.assertEqual((d['ticker'],d['collectionId']),(self.ticker,self.ident))
        self.assertEqual(d['selectedIds'],sorted(selected));self.assertEqual(d['missingIds'],sorted([foreign,collected,'9223372036854775807']))
        self.assertEqual([r['id'] for r in d['documents']],['thesis_'+old]);self.assertFalse(d['documents'][0]['eligible'])
        self.assertEqual(response.headers['Cache-Control'],'private, no-store');self.assertNotIn('file_data',response.text);self.assertNotIn('SECRET_FIXTURE',response.text)
        for value in ('',old+','+old,'01','-1','1.0','9223372036854775808',','.join(str(i) for i in range(1,102))):
            self.assertEqual(self.client.get(url,query_string={'selected':value}).status_code,400)
        self.assertEqual(self.client.get(url+'?selected='+old+'&after=1').status_code,400)
        self.assertEqual(self.client.get(url+'?selected='+old+'&selected='+old).status_code,400)
        with self.db() as (_,cur):
            cur.execute('SELECT id,metadata,file_data FROM document_files ORDER BY id');self.assertEqual(cur.fetchall(),before)
        self.assertFalse(self.submits)
