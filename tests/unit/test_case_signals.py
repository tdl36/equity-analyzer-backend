import unittest
import uuid
from case_signals import validate, decay_rules
class CaseSignalsTests(unittest.TestCase):
    def setUp(self):
        self.a=str(uuid.uuid4())
        self.row=dict(id=str(uuid.uuid4()),assumptionId=self.a,statement='Original statement',source='Original p2',asOf='2026-01-01',targetType='risk',direction='challenge',kind='structural')
    def test_roundtrip_and_link_guard(self):
        result=validate({'observations':[self.row]},[self.a])
        self.assertEqual(result,validate(result,[self.a]))
        with self.assertRaisesRegex(ValueError,'removed assumption'):validate(result,[])
    def test_duplicate_and_invalid_observation(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate({'observations':[self.row,self.row]},[self.a])
        for changes in ({'asOf':'9999-01-01'},{'asOf':'2026-02-30'},{'source':''},{'kind':'forever'}):
            with self.assertRaises(ValueError):validate({'observations':[{**self.row,**changes}]},[self.a])
    def test_rules(self):
        self.assertIsNone(decay_rules({'event':15,'cyclical':90,'structural':20})['structural'])
        for n in (0,-1,True,1.5,4000):
            with self.assertRaises(ValueError):decay_rules({'event':n})
    def test_position_and_baseline_guards(self):
        for p in ({'weight':'NaN'},{'weight':'101'},{'weight':'2'},{'benchmarkWeight':'2'}):
            with self.assertRaises(ValueError):validate({'position':p},[self.a])
        r=dict(id=str(uuid.uuid4()),assumptionId=self.a,market='2',baselineType='broker')
        with self.assertRaises(ValueError):validate({'variants':[r]},[self.a])
    def test_optional_legacy_case_and_framework(self):
        from investment_case import validate as case
        from investor_framework import validate as framework
        self.assertNotIn('signals',case({}))
        self.assertNotIn('decayRules',framework({}))
        self.assertEqual(framework({'decayRules':{'event':15,'cyclical':90}})['decayRules']['event'],15)
if __name__=='__main__':unittest.main()
