import copy
import unittest
from research_infographic import Editor, render, sensitivity

class EditorialTests(unittest.TestCase):
    def setUp(self):
        self.run={'ticker':'TEST','sources':[{'id':'s','filename':'source.pdf','sourceUrl':'javascript:bad'}],'state':{'citations':{}}}
        self.f={'annuals':[],'quote':None,'sourceUrl':'https://www.sec.gov/','sourceHash':'a','asOf':'2026-10-10','retrievedAt':'2026-10-10'}
    def add(self,path,statement,**kw):
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
