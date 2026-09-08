import unittest
from research_numbers import reconcile,number
class NumericTests(unittest.TestCase):
    def point(self,value,unit='usd_m',period='Q1 FY2027',basis='reported',prefix='Reported revenue'):
        words={'usd_m':'million USD','usd_bn':'billion USD','percent':'percent','bps':'bps'}[unit]
        return {'value':value,'unit':unit,'period':period,'basis':basis,'sourceId':'s1','page':1,'quote':f'{prefix} for {period} was {value} {words}, as recorded in the company release.'}
    def run_comparison(self,a,b,kind='broker_estimate'):
        sources=[{'id':'s1','filename':'release','pages':[{'page':1,'text':a['quote']+'\n'+b['quote']}]}]
        return reconcile([{'metric':'Revenue','actual':a,'benchmark':b,'benchmarkType':kind}],sources)[0]
    def test_currency_scales_align_before_calculation(self):
        r=self.run_comparison(self.point(540),self.point(.52,'usd_bn'));self.assertEqual(r['status'],'arithmetic_checked');self.assertEqual(float(r['delta']),20000000);self.assertEqual(float(r['relativePercent']),3.8462)
    def test_percentage_points_are_not_percent_growth(self):
        r=self.run_comparison(self.point(12,'percent'),self.point(1150,'bps'));self.assertEqual(r['basisPointDelta'],'50.00');self.assertEqual(float(r['delta']),.5)
    def test_period_mismatch_suppresses_misleading_delta(self):
        r=self.run_comparison(self.point(540),self.point(520,period='FY2027'));self.assertIsNone(r['delta']);self.assertIn('Fiscal periods differ for an expectations comparison',r['issues'])
    def test_measurement_bases_cannot_be_mixed(self):
        r=self.run_comparison(self.point(10,'percent',basis='organic',prefix='Organic growth'),self.point(9,'percent'));self.assertIsNone(r['delta'])
    def test_missing_numeric_token_and_unit_are_rejected(self):
        a=self.point(540);a['value']=54;r=self.run_comparison(a,self.point(520));self.assertIsNone(r['delta'])
        a=self.point(540);a['unit']='usd_bn';r=self.run_comparison(a,self.point(520));self.assertIsNone(r['delta'])
    def test_single_broker_estimate_is_not_consensus(self):
        r=self.run_comparison(self.point(540),self.point(520,prefix='Our estimate, not consensus,'),kind='consensus');self.assertIsNone(r['delta'])
    def test_zero_benchmark_has_no_relative_growth(self):
        r=self.run_comparison(self.point(540),self.point(0));self.assertEqual(r['status'],'arithmetic_checked');self.assertIsNone(r['relativePercent'])
    def test_nonfinite_boolean_and_unreasonable_values_rejected(self):
        for v in (True,float('nan'),float('inf'),'1e30'):
            with self.assertRaises(ValueError):number(v)

    def test_compact_magnitude_tokens_are_readable(self):
        a=self.point(540);a['quote']='For Q1 FY2027, reported revenue was $540m in the company release.'
        r=self.run_comparison(a,self.point(520));self.assertEqual(r['status'],'arithmetic_checked')
    def test_malformed_model_values_do_not_crash_or_leak_objects_into_ui(self):
        a=self.point(540);a.update(value={},unit={},basis=[],period={})
        r=self.run_comparison(a,self.point(520));self.assertEqual(r['status'],'needs_review');self.assertIsNone(r['actual']['value']);self.assertIsNone(r['actual']['unit'])
    def test_non_gaap_is_not_gaap(self):
        a=self.point(540,basis='gaap',prefix='Non-GAAP revenue');b=self.point(520,basis='gaap',prefix='GAAP revenue')
        r=self.run_comparison(a,b);self.assertIsNone(r['delta'])
