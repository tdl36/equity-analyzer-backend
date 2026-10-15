"""Synthetic frozen metadata only; no files, databases or provider calls."""
import copy
import unittest
from research_source_provenance import frozen_context, render
from research_assignment_outputs import note
from research_onepager import render as onepager
from research_infographic import render as infographic


def fixture():
    source = {'id': 'report-source-1', 'filename': 'historical-original.pdf', 'originalHash': 'a'*64,
              'coverage': {'mode': 'bounded_excerpts', 'omittedCharacters': 100, 'limitation': 'Middle pages omitted.'},
              'textExtraction': {'limitations': ['Charts were not extracted.', 'Charts were not extracted.']}}
    assignment = {'since': '2026-07-01', 'until': '2026-10-15', 'horizon': '12–24 months',
                  'processingProposal': {'id': 'synthetic'},
                  'researchPurpose': {'kind': 'event_note', 'event': {'title': 'Synthetic guidance update', 'date': '2026-10-01'}},
                  'labPack': [{'documentId': '17', 'filename': source['filename'], 'sha256': source['originalHash']}],
                  'processingSourceContext': {'17': {'role': 'supplemental', 'publishedDate': '2024-03-01',
                    'publisher': 'Synthetic publisher', 'sourceKind': 'transcript', 'referencedBySavedThesis': False}}}
    run = {'id': 'synthetic-report', 'ticker': 'TEST', 'sources': [source], 'state': {'citations': {}}}
    fundamentals = {'annuals': [], 'quote': None, 'sourceUrl': 'https://www.sec.gov/', 'sourceHash': 'synthetic',
                    'asOf': '2026-10-15', 'retrievedAt': '2026-10-15'}
    return source, assignment, run, fundamentals


class SourceProvenanceTests(unittest.TestCase):
    def test_frozen_context_is_matched_by_hash_name_and_unique_document_identity(self):
        source, assignment, _, _ = fixture()
        self.assertEqual(frozen_context(source, assignment)['role'], 'supplemental')
        for field, value in [('filename', 'different.pdf'), ('originalHash', 'b'*64), ('originalHash', None)]:
            with self.subTest(field=field,value=value):
                self.assertIsNone(frozen_context({**source,field:value}, assignment))
        ambiguous=copy.deepcopy(assignment);ambiguous['labPack'].append(copy.deepcopy(ambiguous['labPack'][0]))
        self.assertIsNone(frozen_context(source, ambiguous))
        ambiguous['labPack'][1]['sha256']='b'*64
        self.assertIsNone(frozen_context(source, ambiguous))
        assignment['labPack'][0]['documentId']='18'
        self.assertIsNone(frozen_context(source, assignment))

    def test_legacy_missing_or_changed_context_never_infers_from_current_source_metadata(self):
        source, assignment, _, _ = fixture()
        source.update(publishedDate='2099-01-01',publisher='CURRENT METADATA',role='collection')
        for a in ({}, {**assignment,'labPack':[]}, {**assignment,'processingSourceContext':{}}, {**assignment,'labPack':[{'filename':source['filename'],'sha256':'b'*64}]}):
            with self.subTest(a=a):
                result=render(source,a)
                self.assertIn('no unambiguous frozen selection metadata',result)
                self.assertNotIn('2099',result);self.assertNotIn('CURRENT METADATA',result)
                self.assertIn('Charts were not extracted.',result)

    def test_metadata_and_extraction_text_are_escaped_without_private_fields(self):
        source, assignment, _, _ = fixture()
        entry=assignment['processingSourceContext']['17'];entry.update(publisher='<script>publisher</script>',publishedDate='2026-02-30',privateBody='PRIVATE BODY',referencedBySavedThesis=True)
        source['textExtraction']['limitations'].append('<img src=x onerror=alert(1)>')
        source['coverage']['limitation']='<script>context</script>'
        before=copy.deepcopy((source,assignment));result=render(source,assignment)
        self.assertIn('&lt;script&gt;',result);self.assertIn('&lt;img',result)
        self.assertNotIn('<script>',result);self.assertNotIn('<img',result);self.assertNotIn('PRIVATE BODY',result)
        self.assertIn('Published: unknown',result);self.assertIn('incorporation and current validity are not established',result)
        self.assertEqual(result.count('Charts were not extracted.'),1)
        self.assertEqual((source,assignment),before)

    def test_omissions_warn_even_without_freeform_limitation_and_do_not_invent_physical_coverage(self):
        source, assignment, _, _ = fixture();source['coverage'].pop('limitation')
        self.assertIn('some extracted text was omitted',render(source,assignment))
        source['coverage']={'mode':'full_returned_extraction','omittedCharacters':0}
        result=render(source,assignment)
        self.assertNotIn('some extracted text was omitted',result);self.assertIn('Charts were not extracted.',result)

    def test_all_derivative_renderers_include_same_frozen_provenance_without_changing_research(self):
        source, assignment, run, f = fixture();before=copy.deepcopy((run,assignment,f))
        for name, result in [('summary',note(run,'summary_note',assignment)), ('stock_summary',note(run,'stock_summary',assignment)), ('onepager',onepager(run,assignment)), ('infographic',infographic(run,assignment,f))]:
            with self.subTest(renderer=name):
                for expected in ['Supplemental history; not collected', '2024-03-01', 'Synthetic publisher', 'Recorded source type: Transcript', 'does not establish that the evidence is new or unused', 'Middle pages omitted.', 'Charts were not extracted.', 'a'*64]:
                    self.assertIn(expected,result)
        summary=note(run,'summary_note',assignment)
        self.assertIn('Standalone event analysis; this does not update the thesis',summary)
        self.assertIn('linked collection discovery, not the publication dates',summary)
        self.assertIn('Discovery window',onepager(run,assignment))
        self.assertIn('<b>Dates:</b> Discovery ',infographic(run,assignment,f))
        self.assertEqual((run,assignment,f),before)

    def test_infographic_does_not_claim_excerpting_for_full_returned_extraction(self):
        source, assignment, run, f = fixture();source['coverage']={'mode':'full_returned_extraction','omittedCharacters':0}
        result=infographic(run,assignment,f)
        self.assertNotIn('Some originals were excerpted',result)
        self.assertIn('not an exhaustive company review',result)
        self.assertIn('Charts were not extracted.',result)
