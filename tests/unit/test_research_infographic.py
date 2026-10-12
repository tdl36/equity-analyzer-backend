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
    def test_preserve_distinct_same_sector_arguments_and_full_qualifiers(self):
        self.add('/summary/investment_thesis/0','MA margins could recover, but only if costs stabilize')
        self.add('/summary/investment_thesis/1','MA margins face regulatory pressure independent of costs')
        self.add('/summary/investment_thesis/2','Medicaid rates could improve from a loss')
        editor=Editor(self.run);picked=editor.select(['/summary/investment_thesis/'],2)
        self.assertEqual(len(picked),2)
        self.assertEqual(sum('MA margins' in c['statement'] for c in picked),2)
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
        picked=Editor(self.run).select(['/summary/investment_thesis/','/summary/top_drivers/'],1)
        self.assertEqual(picked[0]['statement'],'Margins can recover if renewal prices exceed medical cost growth.')
        h=render(self.run,{},self.f)
        self.assertNotIn('What must be true for the investment case to work?',h)

    def test_saved_history_does_not_claim_fresh_retrieval(self):
        h=render(self.run,{},self.f)
        self.assertIn('Saved GAAP history',h)
        self.assertNotIn('Latest-filed GAAP history',h)

    def test_driver_cannot_fill_missing_or_overlong_thesis(self):
        self.add('/summary/top_drivers/0','Quarterly volume rose 5%.')
        for statement in (None, 'Conditional ownership argument ' * 30):
            if statement:self.add('/summary/investment_thesis/0',statement)
            h=render(self.run,{},self.f)
            self.assertNotIn('Quarterly volume rose 5%.',h)
            self.assertIn('Source-supported thesis evidence unavailable.',h)
            self.assertIn('Incomplete research brief.',h)
            self.assertIn('company description, investment thesis, paired market debate',h)

    def test_coverage_checks_rendered_sections_without_certifying_quality(self):
        self.add('/summary/one_liner','Business description.')
        self.add('/summary/investment_thesis/0','Margins can recover if prices exceed costs.')
        self.add('/debates/0/market_view','The market expects sustained demand.')
        self.add('/debates/0/alternative_view','Demand could slow after replenishment.')
        self.add('/debates/0/resolution_signal','Watch next quarter orders versus sales.')
        self.add('/summary/earnings_direction','Earnings depend on retained pricing.')
        before=copy.deepcopy(self.run)
        h=render(self.run,{},self.f)
        self.assertNotIn('Incomplete research brief.',h)
        self.assertIn('source matching does not establish investment accuracy',h)
        self.assertEqual(self.run,before)
        self.add('/summary/earnings_direction','Long statement ' * 30)
        h=render(self.run,{},self.f)
        self.assertIn('Missing from this page: investment view.',h)

    def test_report_order_beats_numeric_density_source_count_and_citation_insertion(self):
        self.add('/summary/investment_thesis/10','Medical costs: 123%, 456 basis points.',
                 evidence=[{'sourceId':'s','matched':True}]*3)
        self.add('/summary/investment_thesis/2','Pricing may improve margins if demand holds.')
        self.add('/summary/investment_thesis/0','Cash conversion depends on inventory normalizing.')
        editor=Editor(self.run)
        picked=editor.select(['/summary/investment_thesis/'],2,budget=380)
        self.assertEqual([c['statement'] for c in picked],[
            'Cash conversion depends on inventory normalizing.',
            'Pricing may improve margins if demand holds.'])

    def test_same_thesis_as_headline_is_not_lost_and_exact_duplicates_are_suppressed(self):
        statement='Margins can recover if prices outpace costs.'
        self.add('/infographic/headline',statement)
        self.add('/summary/investment_thesis/0',statement)
        self.add('/summary/investment_thesis/1','  '+statement.upper()+'  ')
        self.add('/summary/investment_thesis/2','Cash conversion depends on inventories.')
        h=render(self.run,{},self.f)
        self.assertNotIn('Source-supported thesis evidence unavailable.',h)
        self.assertIn('Cash conversion depends on inventories.',h)
        self.assertNotIn(statement.upper(),h)

    def test_printed_footer_keeps_citation_targets_and_safe_source_links(self):
        self.run['sources'][0]['sourceUrl']='https://example.com/original'
        self.add('/summary/one_liner','Complete supported statement')
        h=render(self.run,{},self.f)
        footer=h.split('<div class="footer">',1)[1].split('</main>',1)[0]
        self.assertIn('id="source-1"',footer)
        self.assertIn('href="https://example.com/original"',footer)
        self.assertEqual(h.count('id="source-1"'),1)
