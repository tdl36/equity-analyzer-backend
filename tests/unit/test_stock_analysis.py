import copy
import unittest
from stock_analysis import SCHEMA,GROUPS,options,validate_report,generate,compare
from stock_analysis_visual import document,amount,chart

SOURCE={'id':'source','filename':'Synthetic original.txt','originalHash':'a'*64,'extractionHash':'b'*64,
        'text':'Synthetic Company reported revenue of USD 125 million for fiscal 2025, compared with USD 100 million for fiscal 2024.'}

def blank(schema):
    if schema['type']=='object':return {k:blank(s) for k,s in schema['properties'].items()}
    if schema['type']=='array':return []
    return 'Unavailable'

def result(group):
    report={k:blank(SCHEMA['properties'][k]) for k in group}
    citations=[]
    if 'summary' in group:
        report['summary']['one_liner']='Synthetic company revenue was USD 125 million in fiscal 2025.'
        citations=[{'path':'/summary/one_liner','basis':'reported_fact','evidence':[{'sourceId':'source','excerpt':SOURCE['text']}]}]
    return {'report':report,'citations':citations}

class StudioTests(unittest.TestCase):
    def test_options_reject_invalid_modes_and_types(self):
        self.assertEqual(options({})['mode'],'deep')
        for data in ({'mode':'fake'},{'horizon':23},{'question':'x'*12001},{'priorId':3}):
            with self.assertRaises(ValueError):options(data)

    def test_shape_identity_and_field_citations(self):
        raw=result(GROUPS[0]);report,c=validate_report(raw,GROUPS[0],[SOURCE],'deep')
        self.assertTrue(c['/summary/one_liner']['passageMatched'])
        for change in ('field','source','path','duplicate','basis'):
            raw=result(GROUPS[0])
            if change=='field':del raw['report']['industry']
            if change=='source':raw['citations'][0]['evidence'][0]['sourceId']='invented'
            if change=='path':raw['citations'][0]['path']='/imaginary'
            if change=='duplicate':raw['citations']*=2
            if change=='basis':raw['citations'][0]['basis']='proven_consensus'
            with self.subTest(change=change),self.assertRaises(ValueError):validate_report(raw,GROUPS[0],[SOURCE],'deep')
        raw=result(GROUPS[0]);raw['citations']=[]
        _,c=validate_report(raw,GROUPS[0],[SOURCE],'deep')
        self.assertEqual(c['/summary/one_liner']['review'],'needs_review')

    def test_generation_checkpoints_and_unknown_outcome(self):
        saved=[];calls=[]
        def ask(prompt,tokens,stage):
            calls.append(stage)
            if stage=='report-1':raise TimeoutError()
            if stage.startswith('report'):return result(GROUPS[int(stage[-1])])
            return {'findings':[{'claimId':'/summary/one_liner','status':'supported','reason':'Matches synthetic evidence.'}]}
        with self.assertRaises(TimeoutError):generate({},[SOURCE],{'revision':2},ask,lambda v:saved.append(copy.deepcopy(v)),lambda:None,{'mode':'deep'})
        self.assertEqual(saved[-1]['completed'],['report-0','review-0'])
        self.assertEqual(saved[-1]['inFlight'],'report-1')
        with self.assertRaises(ValueError):generate(saved[-1],[SOURCE],{},ask,lambda v:None,lambda:None,{'mode':'deep'})
        self.assertEqual(calls,['report-0','review-0','report-1'])
        state=copy.deepcopy(saved[-1]);state.pop('inFlight')
        final=generate(state,[SOURCE],{'revision':2},lambda p,t,k:result(GROUPS[int(k[-1])]),lambda v:None,lambda:None,{'mode':'deep'})
        self.assertEqual(len(final['completed']),12)
        self.assertEqual(len(final['report']),16)
        self.assertNotIn('inFlight',final)

    def test_comparison_does_not_invent_fundamental_change(self):
        report,c=validate_report(result(GROUPS[0]),GROUPS[0],[SOURCE],'deep')
        c['/summary/one_liner']['review']='supported'
        old={'id':'old','state':{'report':report,'citations':c},'sources':[SOURCE]}
        new=copy.deepcopy(old['state']);new['report']['summary']['one_liner']='Different wording'
        d=compare(old,new,[SOURCE],{'revision':3})
        self.assertEqual(d['sections'][0]['kind'],'wording_or_interpretation')
        self.assertEqual(d['sourceChanges'],[])
        src={**SOURCE,'originalHash':'c'*64}
        d=compare(old,new,[src],{'revision':3})
        self.assertEqual(d['sections'][0]['kind'],'evidence_selection_changed')
        self.assertEqual(d['sourceChanges'][0]['change'],'changed')

    def test_returned_response_is_retained_before_validation_and_not_rebilled(self):
        saved=[];calls=[]
        def ask(prompt,tokens,stage):calls.append(stage);return {'report':{},'citations':[]}
        with self.assertRaises(ValueError):generate({},[SOURCE],{},ask,lambda v:saved.append(copy.deepcopy(v)),lambda:None,{'mode':'deep'})
        self.assertNotIn('inFlight',saved[-1]);self.assertIn('report-0',saved[-1]['responses'])
        with self.assertRaises(ValueError):generate(saved[-1],[SOURCE],{},ask,lambda v:None,lambda:None,{'mode':'deep'})
        self.assertEqual(calls,['report-0'])

    def test_visual_omits_unreviewed_fields_and_escapes_text(self):
        report,c=validate_report(result(GROUPS[0]),GROUPS[0],[SOURCE],'deep')
        report['summary']['one_liner']='<script>unsafe</script>'
        run={'id':'r','ticker':'SYNTH','created_at':'2026-10-09','status':'complete','input':{'mode':'deep'},'baseline':{'revision':1},'sources':[SOURCE],'state':{'report':report,'citations':c}}
        self.assertNotIn('unsafe',document(run,True))
        self.assertIn('&lt;script&gt;',document(run,False));self.assertNotIn('<script>',document(run,False))
        self.assertIn('Original SHA-256',document(run,True))

    def test_chart_does_not_infer_scale_or_plot_unsupported_values(self):
        self.assertIsNone(amount('roughly 5 to 10 million'))
        self.assertIsNone(amount('123'));self.assertIsNone(amount('USD NaN'))
        self.assertEqual(str(amount('USD 125 million')[0]),'125')
        report={'financials':{'historical':[{'period':'FY2024','revenue':'USD 100 million'},{'period':'FY2025','revenue':'USD 125 million'}]}}
        citations={}
        for i in range(2):
            for k in ('period','revenue'):
                citations[f'/financials/historical/{i}/{k}']={'review':'supported','passageMatched':True,'evidence':[{'excerpt':SOURCE['text'],'matched':True}]}
        self.assertIn('<svg',chart(report,citations))
        report['financials']['historical'][1]['revenue']='USD 999 million'
        self.assertNotIn('<svg',chart(report,citations))
        report['financials']['historical'][1]['revenue']='USD 125 billion'
        self.assertNotIn('<svg',chart(report,citations))

class ThesisRepairTests(unittest.TestCase):
    def test_repair_retains_prior_output_and_other_completed_groups(self):
        from stock_analysis import repair_unsupported_thesis
        state={'completed':[k+str(i) for i in range(6) for k in ('report-','review-')],
               'report':{'summary':{'investment_thesis':['Unsupported old claim']},'financials':{'kept':True}},
               'citations':{'/summary/investment_thesis/0':{'review':'needs_review','passageMatched':True},'/financials/value':{'review':'supported'}},'retries':['report-0']}
        fixed=repair_unsupported_thesis(state)
        self.assertEqual(fixed['report']['financials'],{'kept':True});self.assertNotIn('report-0',fixed['completed'])
        self.assertIn('review-5',fixed['completed']);self.assertEqual(len(fixed['retries']),3)
        self.assertEqual(fixed['qualityRevisions'][0]['report']['summary'],state['report']['summary'])
        self.assertIn('summary',state['report'])
        state['citations']['/summary/investment_thesis/0']['review']='supported'
        with self.assertRaisesRegex(ValueError,'already exist'):repair_unsupported_thesis(state)
    def test_unresolved_call_or_unfinished_review_cannot_be_repaired(self):
        from stock_analysis import repair_unsupported_thesis
        for state in ({'inFlight':'review-0'},{'completed':['report-0']}):
            with self.assertRaises(ValueError):repair_unsupported_thesis(state)
