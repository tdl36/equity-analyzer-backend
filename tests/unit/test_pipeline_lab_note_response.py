import json
import unittest
from pipeline_lab_note_response import parse


class NoteResponse(unittest.TestCase):
    def setUp(self):
        self.pack=[{'reference':'charlie:thesis_1','filename':'fixture.pdf','sha256':'a'*64,'bytes':10}]
        self.charts={k:{'label':label,'revenue':[{'segment':'A','value':10}],
                        'profit':[{'segment':'A','value':-1}]} for k,label in
                       [('priorYear','FY2025A'),('currentYear','FY2026E')]}
        self.note='===NOTE_START===\n# Fixture company\nInterpretation [1].\n===NOTE_END==='
        self.sources='===SOURCES_START===\n[1] Fixture report p2\n===SOURCES_END==='
    def run_parse(self,text,reason='end_turn'):
        return parse({'text':text,'provider':'synthetic','model':'fixture','stopReason':reason},'Fixture draft',self.pack)
    def test_full_response_retains_note_sources_and_four_chart_series(self):
        result=self.run_parse('===CHART_DATA==='+json.dumps(self.charts)+'===CHART_DATA_END==='+self.note+self.sources)
        self.assertEqual(result['markdown'],'# Fixture company\nInterpretation [1].')
        self.assertEqual(result['generationEvidence']['sourcesMarkdown'],'[1] Fixture report p2')
        self.assertEqual(len(result['generationEvidence']['chartSpecs']),4)
        self.assertEqual(result['generationEvidence']['chartSpecs'][1]['data'][0]['value'],-1)
        self.assertTrue(any('investor verification' in issue for issue in result['qualityIssues']))
    def test_missing_optional_sections_and_truncation_are_explicit(self):
        result=self.run_parse(self.note,'max_tokens')
        self.assertEqual(len(result['qualityIssues']),4)
        self.assertEqual(result['generationEvidence']['chartSpecs'],[])
    def test_no_fallback_for_missing_duplicate_or_overlapping_boundaries(self):
        for text in ('Arbitrary provider text',self.note+self.note,
                     self.note.replace('===NOTE_END===',''),
                     self.note.replace('===NOTE_END===',self.sources.replace('===SOURCES_END===','')+'===NOTE_END======SOURCES_END===')):
            with self.assertRaises(ValueError):self.run_parse(text)
    def test_invalid_chart_json_and_duplicate_keys_fail_without_repair_calls(self):
        for raw in ('not json','{"priorYear":{},"priorYear":{},"currentYear":{}}',
                    json.dumps({**self.charts,'currentYear':self.charts['priorYear']})):
            with self.assertRaises(ValueError):
                self.run_parse('===CHART_DATA==='+raw+'===CHART_DATA_END==='+self.note)
