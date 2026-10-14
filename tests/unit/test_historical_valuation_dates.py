"""Frozen calendar policy, synthetic financial snapshots only."""
import copy
import unittest
from historical_valuation_dates import frozen_dates
from onepager_corrections import digest
from tests.unit import test_historical_valuation_candidates as fixtures


class FrozenDatesTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.CandidateTests();f.setUp();self.baseline=f.baseline
        self.financials={'asOf':'2025-03-04','annuals':[
            {'start':'2023-01-01','end':'2023-12-31'},
            {'start':'2024-01-01','end':'2024-12-31'}]}

    def plan(self):
        self.baseline['fundamentalsDigest']=digest({'input':{},'fundamentals':self.financials,'vintage':self.financials})
        return frozen_dates(self.financials,self.baseline,assignment_input={})

    def test_exact_year_end_including_weekend_is_not_shifted(self):
        p=self.plan()['payload']
        self.assertEqual(p['dates'],['2023-12-31','2024-12-31'])
        self.assertFalse(p['tradingDatesVerified']);self.assertFalse(p['valuationReady'])
        self.assertEqual(p['anchors'][0]['basis'],'fiscal_period_end')

    def test_changed_financials_cannot_reuse_baseline(self):
        self.plan();self.financials['asOf']='2025-03-05'
        with self.assertRaises(ValueError):frozen_dates(self.financials,self.baseline,assignment_input={})

    def test_invalid_duplicate_or_uncompleted_period_withholds_entire_series(self):
        for row in ({'start':'2024-01-01','end':'2024-12-31'},
                    {'start':'2024-03-05','end':'2025-03-04'},
                    {'start':'2025-01-01','end':'2025-12-31'},
                    {'start':'2024-02-30','end':'2024-12-31'},None):
            old=copy.deepcopy(self.financials)
            self.financials['annuals'].append(row)
            p=self.plan()['payload']
            self.assertEqual(p['dates'],[]);self.assertEqual(p['status'],'needs_date_review')
            self.assertTrue(p['gaps']);self.financials=old

    def test_empty_and_oversize_history(self):
        self.financials['annuals']=[]
        self.assertEqual(self.plan()['payload']['gaps'],[{'code':'no_frozen_annual_periods'}])
        self.financials['annuals']=[{}]*31
        with self.assertRaises(ValueError):self.plan()

    def test_no_earnings_values_escape_and_snapshot_is_detached(self):
        self.financials['annuals'][0]['eps']={'value':999,'filed':'2025-03-03'}
        p=self.plan()
        self.assertNotIn('999',str(p));before=copy.deepcopy(p)
        self.financials['annuals'].clear()
        self.assertEqual(p,before)
