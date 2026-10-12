import copy
import unittest
from onepager_readiness import readiness
from research_infographic import render


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.run={'ticker':'TEST','sources':[{'id':'s'}], 'state':{'report':{},'citations':{}}}

    def add(self,path,text='A qualified claim',supported=True):
        parts=path.strip('/').split('/');node=self.run['state']['report']
        for key in parts[:-1]:node=node.setdefault(key,{})
        node[parts[-1]]=text
        self.run['state']['citations'][path]={'statement':text,'review':'supported' if supported else 'needs_review','passageMatched':True,'evidence':[{'sourceId':'s','matched':True}]}

    def statuses(self,rendered=None):
        return {r['section']:r['status'] for r in readiness(self.run,rendered or {})}

    def test_absent_and_unmatched_distinct(self):
        self.assertEqual(self.statuses()['thesis'],'absent')
        self.add('/summary/investment_thesis/0',supported=False)
        self.assertEqual(self.statuses()['thesis'],'unmatched')
        self.run['state']['citations']['/summary/investment_thesis/0']['review']='supported'
        self.run['state']['citations']['/summary/investment_thesis/0']['statement']='stale'
        self.assertEqual(self.statuses()['thesis'],'unmatched')

    def test_no_cross_pair_or_monitor_as_argument(self):
        self.add('/debates/0/resolution_signal')
        self.assertEqual(self.statuses()['debate'],'absent')
        self.add('/debates/0/market_view');self.add('/debates/1/alternative_view')
        self.assertEqual(self.statuses()['debate'],'incomplete_pair')
        self.add('/debates/0/alternative_view',supported=False)
        self.assertEqual(self.statuses()['debate'],'incomplete_pair')

    def test_complete_pair_and_actual_render_selection(self):
        self.add('/debates/0/evidence_for_market','A'*360)
        self.add('/debates/0/evidence_for_alternative')
        self.assertEqual(self.statuses()['debate'],'over_budget')
        self.assertEqual(self.statuses({'debate':True})['debate'],'rendered')

    def test_outside_bounded_selection_not_called_length(self):
        self.add('/debates/20/market_view');self.add('/debates/20/alternative_view')
        self.assertEqual(self.statuses()['debate'],'outside_selection')

    def test_placeholder_and_exact_view_path(self):
        self.add('/summary/earnings_direction','Unavailable')
        self.add('/summary/earnings_direction_extra')
        self.assertEqual(self.statuses()['view'],'absent')

    def test_output_has_paths_no_claim_bodies_or_mutation(self):
        self.add('/summary/investment_thesis/0','PRIVATE CLAIM '+('x'*300))
        before=copy.deepcopy(self.run);rows=readiness(self.run,{})
        self.assertNotIn('PRIVATE CLAIM',str(rows));self.assertEqual(before,self.run)
        self.assertIn('/summary/investment_thesis/0',rows[0]['supportedPaths'])

    def test_render_diagnoses_matching_and_space_separately(self):
        self.add('/summary/investment_thesis/0','x'*300)
        self.add('/debates/0/alternative_view')
        h=render(self.run,{}, {'annuals':[]})
        self.assertIn('data-readiness="over_budget"',h)
        self.assertIn('data-readiness="incomplete_pair"',h)
        self.assertNotIn('Paired evidence unavailable at one-page length',h)
        self.assertIn('Evidence readiness — core investment argument',h)
        self.assertNotIn('x'*300,h)
