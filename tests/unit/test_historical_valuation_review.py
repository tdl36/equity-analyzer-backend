import copy
import hashlib
import json
import unittest
from historical_valuation_review import prepare_pending, replay_pending
from tests.unit import test_historical_price_evidence as price_fixtures


class PendingHistoricalTests(unittest.TestCase):
    def setUp(self):
        self.data, self.originals = price_fixtures.MarketBindingTests().fixture()
        self.baseline = {'assignmentId': '00000000-0000-4000-8000-000000000001',
                         'parentEditionId': '00000000-0000-4000-8000-000000000002',
                         'parentEditionDigest': 'c'*64, 'fundamentalsDigest': 'a'*64, 'sourcePackFingerprint': 'b'*64}
        self.context = {'cik': 1, 'symbol': 'TEST', 'price_retrieval_dates': {'price': '2025-03-04'}}

    def prepare(self):
        return prepare_pending(self.data, self.originals, self.baseline, **self.context)

    def replay(self, snapshot):
        return replay_pending(snapshot, self.originals, self.baseline, **self.context)

    def test_deterministic_pending_replay_preserves_unresolved_checks(self):
        snapshot = self.prepare()
        self.assertEqual(snapshot, self.prepare())
        self.assertEqual(snapshot, self.replay(snapshot))
        payload = snapshot['payload']
        self.assertEqual(payload['status'], 'pending')
        self.assertFalse(payload['diagnosis']['valuationReady'])
        self.assertTrue(payload['diagnosis']['investorReviewRequired'])
        self.assertIn('corporate_actions', payload['diagnosis']['unresolvedChecks'])

    def test_snapshot_detaches_mutable_inputs(self):
        snapshot = self.prepare(); before = copy.deepcopy(snapshot)
        self.data['observations'][0]['price']['value'] = '999'
        self.baseline['fundamentalsDigest'] = 'c'*64
        self.context['price_retrieval_dates']['price'] = '2025-03-05'
        self.assertEqual(snapshot, before)

    def test_every_baseline_change_requires_new_version(self):
        for key in self.baseline:
            with self.subTest(key=key):
                baseline = dict(self.baseline); snapshot = self.prepare()
                self.baseline[key] = ('00000000-0000-4000-8000-000000000003'
                                      if key.endswith('Id') else 'd'*64)
                with self.assertRaises(ValueError): self.replay(snapshot)
                self.assertNotEqual(snapshot['reviewDigest'], self.prepare()['reviewDigest'])
                self.baseline = baseline

    def test_receipt_and_trusted_context_changes_require_new_version(self):
        snapshot = self.prepare(); self.originals['price'] += b' '
        with self.assertRaises(ValueError): self.replay(snapshot)
        self.originals['price'] = self.originals['price'][:-1]
        self.context['price_retrieval_dates']['price'] = '2025-03-05'
        with self.assertRaises(ValueError): self.replay(snapshot)

    def test_edited_diagnosis_or_fake_approval_is_rejected_even_when_rehashed(self):
        for key, value in [('status', 'accepted'), ('diagnosis', {'valuationReady': True})]:
            snapshot = self.prepare(); snapshot['payload'][key] = value
            snapshot['reviewDigest'] = hashlib.sha256(json.dumps(snapshot['payload'],
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            with self.assertRaises(ValueError): self.replay(snapshot)

    def test_missing_source_arrival_requires_new_pending_version(self):
        content = self.originals.pop('price'); snapshot = self.prepare()
        self.assertEqual(snapshot, self.replay(snapshot))
        self.originals['price'] = content
        with self.assertRaises(ValueError): self.replay(snapshot)

    def test_invalid_baseline_and_tampered_envelope_rejected(self):
        snapshot = self.prepare(); snapshot['reviewDigest'] = '0'*64
        with self.assertRaises(ValueError): self.replay(snapshot)
        self.baseline['parentEditionId'] = None
        with self.assertRaises(ValueError): self.prepare()


class HistoricalBaselineTests(unittest.TestCase):
    def fixture(self):
        from tests.unit.test_onepager_edition_contract import edition_fixture
        from onepager_edition_contract import prepare_edition
        args = edition_fixture()
        return args[0], args[1], args[5], prepare_edition(*args)

    def baseline(self, args, parent=None):
        from historical_valuation_review import baseline_from_edition
        return baseline_from_edition(*args, current_parent_id=parent)

    def test_target_content_digest_changes_even_when_edition_id_does_not(self):
        args = self.fixture(); original = self.baseline(args)
        args[3]['claims'][0]['proposal']['statement'] = 'Changed target content'
        changed = self.baseline(args)
        self.assertEqual(original['parentEditionId'], changed['parentEditionId'])
        self.assertNotEqual(original['parentEditionDigest'], changed['parentEditionDigest'])

    def test_changed_report_fundamentals_and_lineage_are_rejected(self):
        for field in ['report', 'fundamentals', 'lineage', 'stopped']:
            args = self.fixture(); parent = None
            if field == 'report': args[0]['state']['report']['new'] = 'changed'
            if field == 'fundamentals': args[2]['price'] = 101
            if field == 'lineage': parent = '00000000-0000-4000-8000-000000000009'
            if field == 'stopped': args[1]['status'] = 'stopped'
            with self.subTest(field=field), self.assertRaises(ValueError): self.baseline(args, parent)

    def test_current_accepted_target_preserves_ancestry_without_granting_approval(self):
        args = self.fixture(); baseline = self.baseline(args, args[3]['id'])
        self.assertEqual(baseline['parentEditionId'], args[3]['id'])
        self.assertNotIn('accepted', baseline)
        self.assertEqual(args[3]['status'], 'pending')
