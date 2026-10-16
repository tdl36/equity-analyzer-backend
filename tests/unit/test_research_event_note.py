"""Event derivative fidelity using synthetic, saved report evidence only."""
import copy
import unittest
from stock_evidence import report_fields
from research_assignment_outputs import note


def fixture():
    report={'summary':{'investment_thesis':['Event implication'], 'what_market_believes':['Supported prior expectation']},
            'earnings':{'reporting_period':'Q3 2026','headline':'Event headline','what_changed':'Guidance changed',
                        'results':[{'metric':'Revenue','actual':'120','prior_guidance':'110','consensus':'Unavailable',
                                    'interpretation':'Comparison interpretation'}]},
            'expectations':{'revision_cycle':'Revision context'},
            'catalysts':[{'catalyst':'Identified catalyst','window':'October 2026','status':'Reported',
                          'impact':'Potential consequence','kpi_to_watch':'Follow-up KPI'}],
            'risks':[{'risk':'Unresolved risk'}],
            'debates':[{'topic':'Unresolved debate'}],
            'diligence_questions':['What explains the guidance change?'],
            'monitor':{'strengtheners':['Monitor next update']}}
    citations={path:{'statement':text,'review':'supported','passageMatched':True,
                      'basis':'interpretation' if 'interpretation' in path or 'investment_thesis' in path else 'reported_fact',
                      'evidence':[{'sourceId':'S1','matched':True,'excerpt':text}]}
               for path,text in report_fields(report).items()}
    run={'id':'fixture-report','ticker':'TEST','sources':[{'id':'S1','filename':'fixture.txt','originalHash':'a'*64}],
         'state':{'report':report,'citations':citations}}
    assignment={'since':'2026-07-01','until':'2026-10-15','horizon':'12–24 months',
                'researchPurpose':{'kind':'event_note','event':{'title':'Guidance update','date':'2026-10-01'}}}
    return run,assignment


class EventNoteTests(unittest.TestCase):
    def test_event_debrief_retains_results_expectations_questions_and_provenance(self):
        run,a=fixture();before=copy.deepcopy((run,a));result=note(run,'summary_note',a)
        for text in ['Event headline','Guidance changed','Revenue','120','110','Comparison interpretation',
                     'Revision context','What explains the guidance change?','Potential consequence','Unresolved debate',
                     'Supported prior expectation','[1]','a'*64]:
            self.assertIn(text,result)
        self.assertNotIn('<h2>Investment view</h2>',result)
        self.assertNotIn('12–24 months',result)
        self.assertNotIn('<b>investment thesis',result)
        self.assertIn('Collection window:',result)
        self.assertIn('separate thesis proposal',result)
        self.assertEqual((run,a),before)

    def test_changed_rejected_or_unmatched_claims_do_not_reach_event_note(self):
        run,a=fixture();run['state']['report']['earnings']['headline']='Uncited replacement'
        run['state']['citations']['/earnings/what_changed']['review']='rejected'
        run['state']['citations']['/diligence_questions/0']['evidence'][0]['matched']=False
        result=note(run,'summary_note',a)
        for text in ['Uncited replacement','Event headline','Guidance changed','What explains the guidance change?']:
            self.assertNotIn(text,result)
        self.assertIn('120',result)
        self.assertNotIn('>Unavailable ',result)

    def test_missing_event_coverage_stays_explicit_without_inventing_consensus(self):
        run,a=fixture();run['state']['citations']={};result=note(run,'summary_note',a)
        self.assertIn('Source-supported coverage unavailable.',result)
        self.assertNotIn('120',result);self.assertNotIn('Supported prior expectation',result)
        self.assertIn('does not establish that every claim occurred on the event date',result)

    def test_legacy_and_longitudinal_notes_keep_existing_sections(self):
        run,a=fixture()
        for purpose in (None,{'kind':'thesis_review'}):
            b=copy.deepcopy(a);b.pop('researchPurpose')
            if purpose:b['researchPurpose']=purpose
            result=note(run,'summary_note',b)
            self.assertIn('<h2>Investment view</h2>',result)
            self.assertNotIn('Event headline',result)

    def test_event_labels_and_claims_are_escaped(self):
        run,a=fixture();a['researchPurpose']['event']['title']='<script>event</script>'
        run['state']['report']['earnings']['headline']='<img src=x onerror=alert(1)>'
        run['state']['citations']['/earnings/headline']['statement']=run['state']['report']['earnings']['headline']
        result=note(run,'summary_note',a)
        self.assertIn('&lt;script&gt;event',result);self.assertIn('&lt;img',result)
        self.assertNotIn('<script>',result);self.assertNotIn('<img',result)

    def test_event_results_follow_report_order_not_citation_arrival_order(self):
        run,a=fixture();run['state']['citations']=dict(reversed(list(run['state']['citations'].items())))
        result=note(run,'summary_note',a)
        self.assertLess(result.index('Revenue'),result.index('120'))
        self.assertLess(result.index('<br>120 '),result.index('<br>110 '))

    def test_long_event_section_discloses_omitted_supported_fields(self):
        run,a=fixture();questions=['Question '+str(i) for i in range(45)]
        run['state']['report']['diligence_questions']=questions
        template=run['state']['citations']['/diligence_questions/0']
        for i,text in enumerate(questions):
            run['state']['citations']['/diligence_questions/'+str(i)]={**copy.deepcopy(template),'statement':text}
        result=note(run,'summary_note',a)
        self.assertIn('6 additional supported fields are not shown',result)
        self.assertIn('Question 39',result);self.assertNotIn('Question 40',result)
