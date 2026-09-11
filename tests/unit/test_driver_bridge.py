import unittest
from driver_bridge import calculate
class BridgeTests(unittest.TestCase):
 def data(self):return dict(revenueMillions='1000',sharesMillions='100',beforeMarginPct='10',afterMarginPct='12',taxPct='25',baselineEPS='2',multiple='20',referencePrice='40',currency='USD',period='FY2026',basis='Adjusted',asOf='2026-01-01')
 def test_hand_calculated_bridge(self):
  r=calculate(self.data());self.assertEqual(r['epsDelta'],'0.1500');self.assertEqual(r['afterValue'],'43.0000');self.assertEqual(r['afterPriceReturnPct'],'7.5000')
 def test_loss_scenario_retains_eps_without_fake_valuation(self):
  r=calculate({**self.data(),'baselineEPS':'-2'});self.assertEqual(r['afterEPS'],'-1.8500');self.assertIsNone(r['afterValue'])
 def test_invalid_units_dates_and_numbers(self):
  for k,v in [('sharesMillions','0'),('taxPct','101'),('baselineEPS','NaN'),('currency',''),('period',''),('asOf','2099-01-01'),('revenueMillions',True)]:
   with self.subTest(k=k),self.assertRaises((ValueError,TypeError)):calculate({**self.data(),k:v})
