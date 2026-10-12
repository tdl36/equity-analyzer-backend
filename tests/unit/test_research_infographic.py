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

    def test_historical_strengthener_is_not_a_forward_signpost(self):
        self.add('/monitor/strengtheners/0','A settlement was reached last year.')
        h=render(self.run,{},self.f)
        self.assertNotIn('A settlement was reached last year.',h)
        self.assertIn('Source-supported monitoring evidence unavailable.',h)

    def test_structured_kpis_preserve_row_identity_qualifiers_and_citations(self):
        self.add('/monitor/kpis/0/metric','Order growth')
        self.add('/monitor/kpis/0/current','2% in Q2')
        self.add('/monitor/kpis/0/bull_threshold','Above 5% in Q3, if pricing holds')
        self.add('/monitor/kpis/1/current','ORPHAN BASELINE')
        self.add('/monitor/kpis/1/bear_threshold','ORPHAN THRESHOLD')
        self.add('/monitor/kpis/0/bear_threshold','UNREVIEWED THRESHOLD',review='needs_review')
        before=copy.deepcopy(self.run);h=render(self.run,{},self.f)
        self.assertIn('Above 5% in Q3, if pricing holds<sup>',h)
        self.assertIn('2% in Q2<sup>',h)
        for text in ['ORPHAN BASELINE','ORPHAN THRESHOLD','UNREVIEWED THRESHOLD']:
            self.assertNotIn(text,h)
        self.assertIn('Not established in saved evidence',h)
        self.assertEqual(before,self.run)

    def test_structured_risks_preserve_impact_timing_and_early_warning(self):
        for key,value in {'risk':'Reimbursement cuts','earnings_impact':'Margin pressure if rates fall',
                          'window':'Next contract year','early_warning':'Proposed rate notice'}.items():
            self.add('/risks/0/'+key,value)
        self.add('/risks/1/earnings_impact','ORPHAN RISK IMPACT')
        self.add('/financials/weakened/0','Secondary historical weakness')
        h=render(self.run,{},self.f)
        for value in ['Reimbursement cuts','Margin pressure if rates fall','Next contract year','Proposed rate notice']:
            self.assertIn(value+'<sup>',h)
        self.assertNotIn('ORPHAN RISK IMPACT',h)
        self.assertNotIn('Secondary historical weakness',h)

    def test_overlong_structured_row_is_omitted_whole(self):
        self.add('/monitor/kpis/0/metric','Omitted metric')
        self.add('/monitor/kpis/0/bull_threshold','Conditional threshold ' * 30)
        self.add('/monitor/kpis/1/metric','Whole metric')
        self.add('/monitor/kpis/1/bull_threshold','Above 5% next quarter')
        h=render(self.run,{},self.f)
        self.assertNotIn('Omitted metric',h)
        self.assertNotIn('Conditional threshold',h)
        self.assertIn('Whole metric',h)
        self.assertIn('Some supported text omitted for length',h)

    def test_scenarios_keep_inputs_and_outputs_with_their_own_evidence(self):
        for i,name in enumerate(['Bear','Base','Bull']):
            self.add(f'/scenarios/{i}/name',name)
            self.add(f'/scenarios/{i}/eps_or_fcf',f'FY2027 adjusted EPS ${4+i}, conditional on demand')
            self.add(f'/scenarios/{i}/valuation_multiple',f'{15+i}x adjusted P/E')
            self.add(f'/scenarios/{i}/implied_value',f'${(4+i)*(15+i)} per share')
        self.add('/scenarios/1/upside_downside','10% versus $72.73 on 2026-10-01; excludes dividends')
        before=copy.deepcopy(self.run);h=render(self.run,{},self.f)
        self.assertIn('FY2027 adjusted EPS $5, conditional on demand<sup>',h)
        self.assertIn('10% versus $72.73 on 2026-10-01; excludes dividends<sup>',h)
        self.assertIn('Returns are not rebased',h)
        self.assertIn('Arithmetic unverified:',h)
        self.assertEqual(self.run,before)
        scenario=h.split('<table class="scenario-values">')[1].split('</table>')[0]
        rows=scenario.split('<tbody>')[1].split('</tr>')[:3]
        for i,row in enumerate(rows):
            self.assertIn(f'${(4+i)*(15+i)} per share',row)
            self.assertIn(f'FY2027 adjusted EPS ${4+i}',row)

    def test_scenarios_do_not_infer_values_or_render_orphan_outputs(self):
        self.add('/scenarios/0/name','Bear')
        self.add('/scenarios/0/eps_or_fcf','FY2027 EPS $4')
        self.add('/scenarios/0/valuation_multiple','15x')
        self.add('/scenarios/0/implied_value','UNREVIEWED VALUE',review='needs_review')
        self.add('/scenarios/1/implied_value','ORPHAN VALUE')
        h=render(self.run,{},self.f)
        for value in ['UNREVIEWED VALUE','ORPHAN VALUE','$60']:
            self.assertNotIn(value,h)
        self.assertIn('Not established in saved evidence',h)

    def test_empty_scenario_section_is_not_filled_by_mechanical_sensitivity(self):
        self.f['quote']={'price':100,'forwardPE':10}
        h=render(self.run,{},self.f)
        self.assertIn('Source-supported scenario inputs unavailable.',h)
        self.assertIn('Mechanical sensitivity is not a bull/base/bear valuation.',h)

    def test_explicit_scenario_arithmetic_only_uses_supported_fields(self):
        values={'name':'Base','eps_or_fcf':'FY2027 adjusted diluted EPS USD 6.00/share',
                'valuation_multiple':'18.0x FY2027 adjusted diluted EPS; rationale: peer recovery',
                'implied_value':'USD 108.00/share',
                'upside_downside':'+8.43% price return vs USD 99.60/share on 2026-10-09'}
        for key,value in values.items(): self.add('/scenarios/0/'+key,value)
        self.assertIn('EPS × P/E and price return agree',render(self.run,{},self.f))
        self.add('/scenarios/0/implied_value','USD 109.00/share')
        h=render(self.run,{},self.f)
        self.assertIn('implied value does not equal EPS',h)
        self.assertIn('USD 109.00/share<sup>',h)
        self.add('/scenarios/0/implied_value','USD 108.00/share',review='needs_review')
        h=render(self.run,{},self.f)
        self.assertNotIn('EPS × P/E and price return agree',h)
        self.assertIn('Arithmetic unverified:',h)
