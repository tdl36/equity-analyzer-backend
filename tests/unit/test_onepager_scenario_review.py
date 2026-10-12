import copy
import unittest
from onepager_scenario_review import scenario_review

class ScenarioReviewTests(unittest.TestCase):
    def fixture(self):
        values={'name':'Base','eps_or_fcf':'FY2027 adjusted diluted EPS USD 5/share',
                'valuation_multiple':'20x FY2027 adjusted diluted EPS; rationale: synthetic assumption',
                'implied_value':'USD 100/share','upside_downside':'25% price return vs USD 80/share on 2026-10-01'}
        run={'sources':[{'id':'s1'}],'state':{'report':{'scenarios':[values]},'citations':{}}}
        for k,v in values.items():run['state']['citations']['/scenarios/0/'+k]={'statement':v,'review':'supported','passageMatched':True,'evidence':[{'sourceId':'s1','matched':True}]}
        return run,{'claims':[]}
    def test_original_supported_inputs_and_proposed_override_are_separate(self):
        run,record=self.fixture();before=copy.deepcopy(run)
        self.assertEqual(scenario_review(run,record)[0]['status'],'checked')
        record['claims']=[{'proposal':{'path':'/scenarios/0/implied_value','statement':'USD 110/share'}}]
        result=scenario_review(run,record)
        self.assertEqual(result[0]['status'],'mismatch');self.assertEqual(result[0]['inputs']['implied_value'],'USD 110/share')
        self.assertEqual(run,before)
    def test_unsupported_or_missing_fields_never_supply_arithmetic(self):
        run,record=self.fixture();run['state']['citations']['/scenarios/0/eps_or_fcf']['review']='needs_review'
        result=scenario_review(run,record)[0]
        self.assertEqual(result['status'],'unverified');self.assertIsNone(result['inputs']['eps_or_fcf'])
        self.assertEqual(scenario_review({'state':{},'sources':[]},record),[])
    def test_missing_rows_can_be_proposed_but_incomplete_or_mixed_basis_stays_visible(self):
        run,record=self.fixture();proposals=[{'proposal':{'path':k.replace('/0/','/2/'),'statement':v['statement']}} for k,v in run['state']['citations'].items()]
        empty={'state':{},'sources':[]};record={'claims':proposals}
        self.assertEqual(scenario_review(empty,record)[0]['index'],2)
        self.assertEqual(scenario_review(empty,record)[0]['status'],'checked')
        proposals[2]['proposal']['statement']='20x FY2028 GAAP diluted EPS; rationale: synthetic assumption'
        self.assertEqual(scenario_review(empty,record)[0]['status'],'mismatch')
        record['claims']=proposals[:2]
        self.assertEqual(scenario_review(empty,record)[0]['status'],'unverified')
