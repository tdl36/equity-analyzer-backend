import copy
import unittest
from financial_provenance import register

class RegisterTests(unittest.TestCase):
    def test_exact_period_units_calculation_and_safe_filing_link(self):
        snap={'cik':1,'annuals':[{'start':'2025-01-01','end':'2025-12-31','freeCashFlow':{'value':1234567,'unit':'USD','filed':'2026-02-01','accession':'0000000001-26-000001','calculation':'operatingCashFlow − capex'}}]}
        before=copy.deepcopy(snap);result=register(snap)
        for part in ['2025-01-01 to 2025-12-31','1234567 USD','operatingCashFlow − capex','0000000001-26-000001-index.html','not necessarily industrial']:
            self.assertIn(part,result)
        self.assertEqual(snap,before)
    def test_malformed_values_urls_and_notes_are_not_executable(self):
        snap={'cik':'<script>','financialWarnings':['<script>bad</script>'],'annuals':[{'revenue':{'value':float('inf')},'eps':{'value':2,'accession':'javascript:bad'}}]}
        result=register(snap);self.assertNotIn('<script>',result);self.assertNotIn('href=',result)
        self.assertNotIn('Revenue:',result);self.assertIn('Diluted EPS: 2',result)
