import copy
import unittest
from decimal import Decimal
from financial_history_context import context
from research_infographic import render


class FinancialHistoryContextTests(unittest.TestCase):
    def setUp(self):
        self.snapshot={'ticker':'ABT','cik':1800,'asOf':'2026-10-11','annuals':[
            {'start':'2024-01-01','end':'2024-12-31','year':'2024',
             'eps':{'value':7.64,'unit':'USD/shares'},
             'netIncome':{'value':13402000000,'unit':'USD'}}]}

    def test_issuer_reported_bridge_preserves_snapshot(self):
        before=copy.deepcopy(self.snapshot);note=context(self.snapshot)
        self.assertIn('issuer-adjusted $4.67',note['summary'])
        self.assertIn('not normalized',note['caveat'])
        self.assertEqual(Decimal('7.64')-Decimal('2.97'),Decimal('4.67'))
        self.assertEqual(Decimal('13.402')+Decimal('2.633')-Decimal('7.835'),Decimal('8.200'))
        self.assertEqual(self.snapshot,before)

    def test_wrong_issuer_missing_vintage_and_lookahead_are_withheld(self):
        for change in [{'ticker':'UNH'},{'cik':1},{'asOf':'2025-01-21'},{'asOf':None},{'asOf':'bad'}]:
            self.assertIsNone(context(self.snapshot|change))

    def test_period_units_values_and_duplicates_must_match(self):
        for key,value in [('start','2024-04-01'),('end','2024-09-30'),('eps',{'value':7.65,'unit':'USD/shares'}),('eps',{'value':7.64,'unit':'EUR/shares'}),('netIncome',{'value':13402,'unit':'USD'}),('eps',{'value':float('nan'),'unit':'USD/shares'})]:
            snap=copy.deepcopy(self.snapshot);snap['annuals'][0][key]=value
            self.assertIsNone(context(snap))
        self.assertIsNone(context(self.snapshot|{'annuals':self.snapshot['annuals']*2}))

    def test_renderer_labels_new_context_separately_and_retains_original_values(self):
        run={'ticker':'ABT','sources':[],'state':{'report':{},'citations':{}}}
        before=copy.deepcopy(self.snapshot);h=render(run,{},self.snapshot)
        self.assertIn('Historical earnings context',h)
        self.assertIn('Added public context, not part of the original frozen research pack',h)
        self.assertIn('Published 2025-01-22',h)
        self.assertIn('Verified 2026-10-12',h)
        self.assertEqual(self.snapshot,before)
