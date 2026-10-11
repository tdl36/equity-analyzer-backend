import copy
import unittest
from research_onepager import render

class OnePagerTests(unittest.TestCase):
    def setUp(self):
        self.run={'id':'synthetic','ticker':'TEST','sources':[{'id':'s1','filename':'original.pdf','sourceUrl':'javascript:alert(1)'}],'state':{'citations':{}}}
    def add(self,path,text,**kw):
        node=self.run['state'].setdefault('report',{})
        parts=path.strip('/').split('/')
        for key in parts[:-1]:node=node.setdefault(key,{})
        node[parts[-1]]=text
        self.run['state']['citations'][path]={'statement':text,'review':'supported','passageMatched':True,'evidence':[{'sourceId':'s1','matched':True}],**kw}
    def test_no_orphan_valuation_or_cross_debate_pair(self):
        self.add('/valuation/metrics/0/historical_range','Sector 0.88x')
        self.add('/scenarios/1/implied_value','410')
        self.add('/debates/0/market_view','FIRST VIEW')
        self.add('/debates/1/alternative_view','DIFFERENT TOPIC')
        h=render(self.run,{})
        for text in ['Sector 0.88x','410','FIRST VIEW','DIFFERENT TOPIC']:self.assertNotIn(text,h)
        self.assertIn('Current company valuation',h)
    def test_supported_full_claims_escaped_and_input_unchanged(self):
        statement='<script>alert(1)</script> '+('Full statement. '*80)
        self.add('/summary/one_liner',statement)
        self.add('/summary/investment_thesis/0','Unreviewed claim',review='needs_review')
        before=copy.deepcopy(self.run);h=render(self.run,{})
        self.assertIn('&lt;script&gt;',h);self.assertNotIn('<script>',h)
        self.assertNotIn('javascript:',h);self.assertNotIn('Unreviewed claim',h)
        self.assertIn('Full statement. '*80,h);self.assertEqual(self.run,before)
        self.assertIn('href="#source-1"',h)
    def test_paired_debate_and_identified_valuation(self):
        self.add('/debates/0/market_view','Market claim');self.add('/debates/0/alternative_view','Counter claim')
        self.add('/valuation/metrics/0/metric','Company PE');self.add('/valuation/metrics/0/current','20x')
        h=render(self.run,{})
        for text in ['Market claim','Counter claim','Company PE','20x']:self.assertIn(text,h)
