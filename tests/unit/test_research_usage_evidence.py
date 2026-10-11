import unittest
from decimal import Decimal
from research_usage_evidence import read_usage
class UsageEvidenceTests(unittest.TestCase):
    def test_filters_exact_job_and_feature_and_allowlists_data(self):
        class Cursor:
            def execute(self,sql,args): self.sql,self.args=sql,args
            def fetchall(self):return [dict(created_at='2026-10-11',provider='anthropic',model='synthetic',stage='report-0',input_tokens=100,output_tokens=20,cost_usd=Decimal('0.002'),attempt=1,secret='PRIVATE')]
        cur=Cursor();result=read_usage(cur,"job'",'stock-analysis')
        self.assertEqual(cur.args,("job'",'stock-analysis'));self.assertNotIn("job'",cur.sql)
        self.assertEqual(result['records'][0]['estimatedCostUsd'],.002)
        self.assertNotIn('PRIVATE',str(result));self.assertFalse(result['truncated'])
    def test_empty_and_unknown_price_do_not_imply_free(self):
        class Cursor:
            def execute(self,*args):pass
            def fetchall(self):return []
        r=read_usage(Cursor(),'job','company-research')
        self.assertEqual(r['records'],[]);self.assertIn('do not establish zero usage',r['limitation'])
