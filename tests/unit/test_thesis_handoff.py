import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import thesis_handoff as h
from test_thesis_imports import sample


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.folder = root / 'sources'; self.folder.mkdir()
        self.source = self.folder / 'source.txt'; self.source.write_text('Synthetic evidence')
        self.workspace = root / 'work'
        self.baseline = {'hash': 'a'*64, 'mode': 'initial', 'expectedNoExistingThesis': True}
        self.calls = []
        self.identifier = '12345678-1234-1234-1234-123456789abc'
        self.uploaded = None
        self.fail_detail = False
        h.prepare('TEST', self.folder, self.workspace, self.call)
        self.draft = sample(); self.draft['baseline'] = self.baseline
        h.save(self.workspace / 'draft.json', self.draft)

    def call(self, path='', body=None):
        self.calls.append((path, body is not None))
        if path.startswith('/prepare/'):
            return {'package': {'baseline': copy.deepcopy(self.baseline)}}
        if body is not None:
            self.uploaded = body
            return {'id': self.identifier}
        if self.fail_detail:
            raise ValueError('Synthetic connection loss')
        return {'ticker': 'TEST', 'fingerprint': h.digest(self.uploaded), 'status': 'pending', 'stale': False}

    def test_receipt_and_retry_without_approval_or_model_calls(self):
        receipt = h.submit(self.workspace, call=self.call)
        self.assertEqual(receipt['status'], 'pending')
        self.assertIn('?thesisDraft=' + self.identifier, receipt['reviewUrl'])
        self.assertEqual(h.submit(self.workspace, call=self.call), receipt)
        self.assertTrue(all(path in ('/prepare/TEST', '', '/' + self.identifier) for path, _ in self.calls))
        self.assertEqual(h.read(self.workspace / 'receipt.json'), receipt)

    def test_response_loss_preserves_identical_submission(self):
        self.fail_detail = True
        with self.assertRaises(ValueError): h.submit(self.workspace, call=self.call)
        frozen = (self.workspace / 'submission.json').read_bytes()
        self.fail_detail = False
        self.draft['analysis']['conclusion'] = 'Edited after uncertain upload'
        h.save(self.workspace / 'draft.json', self.draft)
        with self.assertRaises(ValueError): h.submit(self.workspace, self.workspace / 'draft.json', self.call)
        h.submit(self.workspace, call=self.call)
        self.assertEqual(frozen, (self.workspace / 'submission.json').read_bytes())

    def test_wrong_baseline_and_changed_source_stop_before_upload(self):
        self.draft['baseline'] = {**self.baseline, 'hash': 'b'*64}
        h.save(self.workspace / 'draft.json', self.draft)
        with self.assertRaises(ValueError): h.submit(self.workspace, call=self.call)
        self.draft['baseline'] = self.baseline
        h.save(self.workspace / 'draft.json', self.draft)
        self.source.write_text('Changed evidence')
        with self.assertRaises(ValueError): h.submit(self.workspace, call=self.call)
        self.source.unlink()
        with self.assertRaises(ValueError): h.submit(self.workspace, call=self.call)
        self.assertFalse(any(post for _, post in self.calls))

    def test_rejects_output_inside_sources_and_existing_workspace(self):
        with self.assertRaises(ValueError): h.prepare('TEST', self.folder, self.folder / 'output', self.call)
        with self.assertRaises(FileExistsError): h.prepare('TEST', self.folder, self.workspace, self.call)

    def test_inventory_is_not_review_attestation(self):
        item = h.read(self.workspace / 'state.json')['files'][0]
        self.assertEqual(item['sha256'], h.file_hash(self.source))
        self.assertIn('not attested', item['reviewStatus'])

    def test_credentials_never_resolved_for_unsupported_routes(self):
        with patch.object(h, 'secret', side_effect=AssertionError('Must not resolve key')):
            for path in ('/x/approve', '//evil.invalid', '/prepare/X?redirect=1'):
                with self.assertRaises(ValueError): h.request(path, {})
        with self.assertRaises(ValueError): h.NoRedirect().redirect_request(None, None, None, None, None, None)

    def test_mismatched_receipt_is_not_saved(self):
        def wrong(path='', body=None):
            result = self.call(path, body)
            if path == '/' + self.identifier: result['fingerprint'] = 'b'*64
            return result
        with self.assertRaises(ValueError): h.submit(self.workspace, call=wrong)
        self.assertFalse((self.workspace / 'receipt.json').exists())
