"""Disposable database and synthetic evidence only; inspection never dispatches."""
import base64
import json
import unittest
import uuid
from unittest.mock import patch
from tests.unit import test_processing_proposals as fixture
from tests.unit.test_research_assignments import PG


@unittest.skipUnless((PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class ReadinessTests(unittest.TestCase):
    setUpClass = classmethod(fixture.ProcessingProposalTests.setUpClass.__func__)
    tearDownClass = classmethod(fixture.ProcessingProposalTests.tearDownClass.__func__)
    setUp = fixture.ProcessingProposalTests.setUp
    invoke = fixture.ProcessingProposalTests.invoke
    report = fixture.ProcessingProposalTests.report
    parent = fixture.ProcessingProposalTests.parent
    save = fixture.ProcessingProposalTests.save
    row = fixture.ProcessingProposalTests.row
    decision = fixture.ProcessingProposalTests.decision

    def check(self, proposal):
        return self.client.get(self.url+'/'+proposal['id']+'/readiness')

    def test_readiness_reads_exact_frozen_proposal_without_dispatch_or_mutation(self):
        p=self.save();before=self.row(p['id']);parent=self.row(self.ident)
        with patch.object(self.co,'wake',side_effect=AssertionError('read cannot wake')):
            r=self.check(p)
        self.assertEqual(r.status_code,200,r.json)
        self.assertTrue(r.json['ready']);self.assertEqual(r.json['blockers'],[])
        for k in ('ticker','collectionId','digest','revision'):self.assertEqual(r.json[k],p[k])
        self.assertEqual(r.json['proposalId'],p['id']);self.assertTrue(r.json['readOnly'])
        self.assertIn('no-store',r.headers['Cache-Control'])
        self.assertEqual(self.row(p['id']),before);self.assertEqual(self.row(self.ident),parent)
        self.assertFalse(self.submits);self.assertNotIn(self.raw.decode(),r.get_data(as_text=True))

    def test_changed_sources_block_inspection_and_approval_without_mutating_proposal(self):
        p=self.save();before=self.row(p['id'])
        with self.db(True) as (_,cur):cur.execute('UPDATE document_files SET file_data=%s WHERE ticker=%s',(base64.b64encode(b'changed bytes').decode(),self.ticker))
        r=self.check(p);self.assertEqual(r.status_code,200);self.assertFalse(r.json['ready'])
        self.assertIn('Selected originals changed',r.json['blockers'][0])
        self.assertEqual(self.decision(p).status_code,409)
        self.assertEqual(self.row(p['id']),before);self.assertFalse(self.submits)

    def test_permissions_baseline_credentials_and_active_work_are_visible(self):
        p=self.save();self.co.has_key=lambda:False
        with self.db(True) as (_,cur):
            cur.execute("UPDATE document_files SET metadata='{"+'"usage":"reference_only"'+"}'::jsonb WHERE ticker=%s",(self.ticker,))
            cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,'research_assignment',%s,'attention','{}')",(str(uuid.uuid4()),self.ticker))
        r=self.check(p).json;self.assertFalse(r['ready']);self.assertEqual(len(r['blockers']),3)
        self.assertIn('model key',' '.join(r['blockers']));self.assertIn('unfinished',' '.join(r['blockers']))
        with self.db(True) as (_,cur):cur.execute("INSERT INTO portfolio_analyses(ticker,company,analysis) VALUES(%s,'Synthetic','{}')",(self.ticker,))
        self.assertIn('accepted thesis changed',' '.join(self.check(p).json['blockers']))
        self.assertFalse(self.submits)

    def test_check_does_not_reserve_approval_and_decided_proposals_do_not_recheck_bytes(self):
        p=self.save();self.assertTrue(self.check(p).json['ready'])
        self.co.has_key=lambda:False
        self.assertEqual(self.decision(p).status_code,400)
        self.assertEqual(self.decision(p,'cancel').status_code,200)
        with patch('processing_proposal_readiness.verify_evidence',side_effect=AssertionError('decided')):
            r=self.check(p)
        self.assertEqual(r.status_code,200);self.assertFalse(r.json['ready']);self.assertEqual(r.json['status'],'cancelled')

    def test_scope_invalid_filters_and_dependency_failure_are_safe(self):
        p=self.save();url=self.url+'/'+p['id']+'/readiness'
        self.assertEqual(self.client.get(url.replace(self.ident,str(uuid.uuid4()))).status_code,404)
        self.assertEqual(self.client.get(url+'?ticker=OTHER').status_code,400)
        self.assertEqual(self.client.get(url.replace(p['id'],'bad')).status_code,400)
        with patch('processing_proposal_readiness.verify_evidence',side_effect=RuntimeError('private detail')):
            r=self.check(p)
        self.assertEqual(r.status_code,503);self.assertNotIn('private detail',r.get_data(as_text=True));self.assertFalse(self.submits)

    def test_research_case_revision_change_is_read_only_blocker(self):
        p=self.save();before=self.row(p['id'])
        with self.db(True) as (_,cur):
            cur.execute("INSERT INTO investment_case_versions(ticker,revision,body) VALUES(%s,1,'{}')",(self.ticker,))
        result=self.check(p).json
        self.assertFalse(result['ready']);self.assertIn('research baseline changed',' '.join(result['blockers']))
        self.assertEqual(self.row(p['id']),before);self.assertFalse(self.submits)

    def test_full_note_check_uses_frozen_plan_without_dispatch_or_mutable_defaults(self):
        self.parent();fixture.ProcessingProposalTests.note_hooks(self)
        self.body['plan']['labFullNote']={'allowPdfText':False}
        saved=self.client.post(self.url,json=self.body);self.assertEqual(saved.status_code,201,saved.json)
        p=saved.json
        self.co.lab_note_defaults=lambda: (_ for _ in ()).throw(AssertionError('mutable defaults'))
        self.assertTrue(self.check(p).json['ready'])
        self.co.lab_note_dispatch=None
        self.assertIn('integration is unavailable',' '.join(self.check(p).json['blockers']))
        self.assertFalse(self.note_calls);self.assertFalse(self.submits)
