"""Synthetic retained inventory; no network or financial writes."""
import copy
import unittest
from historical_valuation_selection import plan_selection
from tests.unit import test_historical_valuation_candidates as fixtures


class SelectionTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.CandidateTests();f.setUp()
        self.records=f.records;self.baseline=f.baseline

    def plan(self,date='2025-03-03'):
        return plan_selection(self.records,self.baseline,cik=1,symbol='TEST',price_date=date)

    def test_unique_exact_date_receipts_select_pending_inputs(self):
        result=self.plan()['payload']
        self.assertEqual(result['status'],'selected')
        self.assertFalse(result['valuationReady'])
        self.assertFalse(result['candidate']['payload']['valuationReady'])
        self.assertEqual(result['candidate']['payload']['observations'][0]['price']['value'],'100.0000')
        self.assertEqual(result['gaps'],[])

    def test_empty_and_missing_inventory_preserve_coverage_gaps(self):
        self.records=[]
        self.assertEqual(len(self.plan()['payload']['gaps']),2)
        self.assertNotIn('candidate',self.plan()['payload'])

    def test_no_adjacent_date_substitution(self):
        result=self.plan('2025-03-02')['payload']
        self.assertEqual(result['status'],'needs_evidence')
        self.assertNotIn('candidate',result)
        self.assertEqual(result['gaps'][0]['input'],'price')

    def test_alias_receipts_remain_ambiguous_even_with_identical_amounts(self):
        self.records.append({**self.records[0],'id':'second-price'})
        result=self.plan()['payload']
        self.assertEqual(result['status'],'ambiguous')
        self.assertNotIn('candidate',result)
        self.assertEqual(len(result['gaps'][0]['eligibleReceipts']),2)

    def test_inventory_order_does_not_change_plan(self):
        before=self.plan();self.records.reverse()
        self.assertEqual(before,self.plan())

    def test_tampered_foreign_and_duplicate_originals_fail_closed(self):
        before=copy.deepcopy(self.records)
        for change in ({'assignmentId':'other'},{'content':b'changed'}):
            self.records=copy.deepcopy(before);self.records[0].update(change)
            with self.assertRaises(ValueError):self.plan()
        self.records=before+[before[0]]
        with self.assertRaises(ValueError):self.plan()

    def test_late_inventory_or_baseline_changes_require_different_plan_identity(self):
        result=self.plan();before=copy.deepcopy(result)
        self.records[0]['retrievedDate']='2025-03-05'
        self.assertNotEqual(result['planDigest'],self.plan()['planDigest'])
        self.baseline['fundamentalsDigest']='f'*64
        self.assertNotEqual(result['planDigest'],self.plan()['planDigest'])
        self.assertEqual(result,before)

    def test_bounds_and_identity_checked_even_when_empty(self):
        for records,cik in [([],True),(self.records*31,1)]:
            with self.assertRaises(ValueError):plan_selection(records,self.baseline,cik=cik,symbol='TEST',price_date='2025-03-03')

    def test_ineligible_originals_remain_diagnosed_without_entering_candidate(self):
        import hashlib
        content=self.records[0]['content'].replace(b'TEST',b'OTHER')
        self.records.append({**self.records[0],'id':'wrong-symbol','content':content,'sha256':hashlib.sha256(content).hexdigest()})
        payload=self.plan()['payload']
        self.assertEqual(payload['status'],'selected')
        self.assertEqual(len(payload['checks']),3)
        self.assertFalse(next(row for row in payload['checks'] if row['source']['id']=='wrong-symbol')['eligible'])
        self.assertNotIn('wrong-symbol',payload['candidate']['payload']['retainedHashes'])
