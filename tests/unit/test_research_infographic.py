import copy
import unittest
from research_infographic import Editor, render, sensitivity

class EditorialTests(unittest.TestCase):
    def setUp(self):
        self.run={'ticker':'TEST','sources':[{'id':'s','filename':'source.pdf','sourceUrl':'javascript:bad'}],'state':{'citations':{}}}
        self.f={'annuals':[],'quote':None,'sourceUrl':'https://www.sec.gov/','sourceHash':'a','asOf':'2026-10-10','retrievedAt':'2026-10-10'}
    def add(self,path,statement,**kw):
        node=self.run['state'].setdefault('report',{})
        parts=path.strip('/').split('/')
        for key in parts[:-1]:node=node.setdefault(key,{})
        node[parts[-1]]=statement
        self.run['state']['citations'][path]=dict(statement=statement,review='supported',passageMatched=True,evidence=[dict(sourceId='s',matched=True)]) | kw
    def test_rank_distinct_evidence_and_keep_full_qualified_claim(self):
        self.add('/summary/investment_thesis/0','MA margins could recover, but only if costs stabilize')
        self.add('/summary/investment_thesis/1','MA margins improving with pricing')
        self.add('/summary/investment_thesis/2','Medicaid rates could improve from a loss')
        editor=Editor(self.run);picked=editor.select(['/summary/investment_thesis/'],2)
        self.assertEqual(len(picked),2)
        self.assertEqual(sum('MA margins' in c['statement'] for c in picked),1)
        self.assertIn('only if costs stabilize',editor.claim(editor.get('/summary/investment_thesis/0')))
    def test_two_debates_no_crosspair_and_no_unreviewed_claim(self):
        for i in range(2):
            self.add(f'/debates/{i}/evidence_for_market',f'Complete argument {i}')
            self.add(f'/debates/{i}/evidence_for_alternative',f'Counter argument {i}')
        self.add('/debates/2/market_view','ORPHAN')
        self.add('/debates/3/alternative_view','OTHER TOPIC')
        self.add('/summary/one_liner','UNREVIEWED',review='needs_review')
        before=copy.deepcopy(self.run);h=render(self.run,{},self.f)
        for i in range(2):self.assertIn(f'Complete argument {i}',h);self.assertIn(f'Counter argument {i}',h)
        for value in ['ORPHAN','OTHER TOPIC','UNREVIEWED','javascript:bad']:self.assertNotIn(value,h)
        self.assertEqual(self.run,before)
    def test_sensitivity_is_arithmetic_not_target_forecast(self):
        h=sensitivity({'price':100,'forwardPE':10})
        self.assertIn('$81',h);self.assertIn('$121',h);self.assertIn('not forecasts',h)
        self.assertIn('unavailable',sensitivity({'price':100,'forwardPE':0}))
        self.assertIn('unavailable',sensitivity({'price':float('nan'),'forwardPE':10}))
    def test_escaping_preserves_source_content(self):
        self.add('/summary/one_liner','<script>unsafe</script>')
        h=render(self.run,{},self.f);self.assertIn('&lt;script&gt;',h);self.assertNotIn('<script>',h)

    def test_long_claim_is_omitted_whole_with_disclosure(self):
        long='A qualified statement ' * 40 + 'only if demand holds.'
        self.add('/summary/investment_thesis/0',long)
        self.add('/summary/investment_thesis/1','A shorter complete claim, conditional on demand holding.')
        h=render(self.run,{},self.f)
        self.assertNotIn('A qualified statement',h)
        self.assertIn('A shorter complete claim, conditional on demand holding.',h)
        self.assertIn('Some supported text omitted for length',h)

    def test_long_debate_skips_both_sides_and_its_monitor(self):
        self.add('/debates/0/market_view','Long argument ' * 60)
        self.add('/debates/0/alternative_view','Short orphan counterpoint')
        self.add('/debates/0/resolution_signal','Orphan monitor')
        self.add('/debates/1/market_view','Whole market view')
        self.add('/debates/1/alternative_view','Whole alternative view')
        self.add('/debates/1/topic','Demand')
        self.add('/debates/1/resolution_signal','Demand monitor')
        h=render(self.run,{},self.f)
        for text in ['Long argument','Short orphan counterpoint','Orphan monitor']:
            self.assertNotIn(text,h)
        for text in ['Whole market view','Whole alternative view','Demand monitor']:
            self.assertIn(text,h)

    def test_catalyst_timing_retains_evidence_reference(self):
        self.run['sources'].append({'id':'timing','filename':'timing.pdf'})
        self.add('/catalysts/0/catalyst','Upcoming investor meeting')
        self.add('/catalysts/0/timing','Next month',evidence=[{'sourceId':'timing','matched':True}])
        h=render(self.run,{},self.f)
        self.assertIn('Next month<sup><a href="#source-2">2</a></sup>',h)

    def test_thesis_prefers_complete_ownership_argument_to_short_driver(self):
        self.add('/summary/investment_thesis/0','Margins can recover if renewal prices exceed medical cost growth.')
        self.add('/summary/top_drivers/0','2027 margin recovery')
        picked=Editor(self.run).select(['/summary/investment_thesis/','/summary/top_drivers/'],1,prefer_prefix=True)
        self.assertEqual(picked[0]['statement'],'Margins can recover if renewal prices exceed medical cost growth.')
        h=render(self.run,{},self.f)
        self.assertNotIn('What must be true for the investment case to work?',h)

    def test_saved_history_does_not_claim_fresh_retrieval(self):
        h=render(self.run,{},self.f)
        self.assertIn('Saved GAAP history',h)
        self.assertNotIn('Latest-filed GAAP history',h)
