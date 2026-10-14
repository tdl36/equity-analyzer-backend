"""Synthetic coordinator readiness, without database or provider calls."""
import copy
import unittest
from historical_valuation_readiness import readiness, replay
from tests.unit import test_historical_valuation_candidates as fixtures


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.CandidateTests(); f.setUp()
        self.records, self.baseline = f.records, f.baseline
        self.identity = dict(cik=1, symbol='TEST')

    def build(self, dates=None):
        return readiness(self.records, self.baseline, **self.identity,
                         dates=dates or ['2025-03-03'])

    def test_selected_receipts_frozen_without_dispatch_permission(self):
        p = self.build()['payload']
        self.assertEqual(p['status'], 'inputs_selected')
        self.assertEqual(p['observations'][0]['selection']['priceSource']['sha256'], self.records[0]['sha256'])
        self.assertEqual(len(p['observations'][0]['candidateDigest']), 64)
        self.assertFalse(p['dispatchAllowed']); self.assertFalse(p['valuationReady'])
        self.assertTrue(p['investorReviewRequired'])

    def test_partial_coverage_retains_both_dates_and_explicit_gaps(self):
        p = self.build(['2025-03-02', '2025-03-03'])['payload']
        self.assertEqual(p['status'], 'needs_evidence_review')
        self.assertEqual(len(p['observations']), 2)
        self.assertNotIn('selection', p['observations'][0])
        self.assertEqual(p['observations'][0]['gaps'][0]['code'], 'missing_coverage')
        self.assertIn('selection', p['observations'][1])

    def test_replay_rejects_new_ambiguity_and_changed_target(self):
        snapshot = self.build()
        self.assertEqual(replay(snapshot, self.records, self.baseline, **self.identity), snapshot)
        self.records.append({**self.records[0], 'id': 'alias'})
        with self.assertRaises(ValueError): replay(snapshot, self.records, self.baseline, **self.identity)
        self.records.pop(); self.baseline['fundamentalsDigest'] = 'f'*64
        with self.assertRaises(ValueError): replay(snapshot, self.records, self.baseline, **self.identity)

    def test_tampered_permission_digest_and_receipts_rejected(self):
        snapshot = self.build()
        for field in ('dispatchAllowed', 'valuationReady'):
            changed = copy.deepcopy(snapshot); changed['payload'][field] = True
            with self.assertRaises(ValueError): replay(changed, self.records, self.baseline, **self.identity)
        snapshot['readinessDigest'] = '0'*64
        with self.assertRaises(ValueError): replay(snapshot, self.records, self.baseline, **self.identity)

    def test_order_independent_and_detached(self):
        a = self.build(['2025-03-03','2025-03-02'])
        self.records.reverse()
        self.assertEqual(a, self.build(['2025-03-02','2025-03-03']))
        self.baseline['fundamentalsDigest'] = 'e'*64
        self.assertNotEqual(a['payload']['baseline'], self.baseline)

    def test_bounds_and_empty_inventory(self):
        for dates in ([], ['2025-03-03']*2, [True], ['2025-03-03']*31, '2025-03-03'):
            with self.assertRaises(ValueError): readiness(self.records,self.baseline,**self.identity,dates=dates)
        self.records = []
        p = self.build()['payload']
        self.assertEqual(p['status'],'needs_evidence_review')
        self.assertEqual(len(p['observations'][0]['gaps']),2)
