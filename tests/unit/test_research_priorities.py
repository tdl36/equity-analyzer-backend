import unittest
from datetime import date
from research_priorities import validate
class PriorityTests(unittest.TestCase):
    def test_signed_exposures_and_explicit_provenance(self):
        p=validate({'asOf':'2026-09-07','positions':[{'ticker':'MDT','weightPct':-5}]},date(2026,9,7))
        self.assertEqual(p['source'],'user_reported');self.assertEqual(p['positions'][0]['weightPct'],-5)
    def test_bad_dates_duplicates_and_nonfinite_weights(self):
        for value in [{'asOf':'2026-09-08','positions':[]},{'asOf':'bad','positions':[]}, {'asOf':'2026-09-07','positions':[{'ticker':'MDT','weightPct':float('nan')}]}, {'asOf':'2026-09-07','positions':[{'ticker':'MDT','weightPct':1},{'ticker':'MDT','weightPct':2}]}]:
            with self.assertRaises(ValueError):validate(value,date(2026,9,7))
