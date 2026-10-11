import json,unittest
from research_json_recovery import recover_report_envelope
class RepairTests(unittest.TestCase):
    def test_removes_only_premature_root_closure_preserving_all_values(self):
        report={'financials':{'text':'escaped " quote } brace and α'},'earnings':[]}
        refs=[{'path':'/financials/text','excerpt':'Synthetic source text'}]
        text=json.dumps({'report':report})+',"citations":'+json.dumps(refs)+'}'
        self.assertEqual(recover_report_envelope('\n```json\n'+text+'\n```'),{'report':report,'citations':refs})
    def test_rejects_truncation_duplicates_unrelated_objects_and_prose(self):
        for text in ['{"report":{}},"citations":[', '{"report":{}},"citations":[],"citations":[]}',
                     '{"other":{}},"citations":[]}', 'Here is {"report":{}},"citations":[]}',
                     '{"report":{},"citations":[]}','{"report":{}},"citations":[],"extra":true}']:
            with self.subTest(text=text),self.assertRaises(ValueError):recover_report_envelope(text)
