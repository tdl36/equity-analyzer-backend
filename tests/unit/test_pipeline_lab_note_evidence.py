import copy
import unittest
from pipeline_lab_note_evidence import normalize, issues


class Evidence(unittest.TestCase):
    def setUp(self):
        self.value = {'sourcesMarkdown': '[1] Fixture report, page 2', 'stopReason': 'end_turn',
                      'chartSpecs': [{'kind': 'profit', 'period': 'FY2025A',
                                      'data': [{'segment': 'A', 'value': -2}]}]}

    def test_negative_profit_is_preserved_without_quality_certification(self):
        result = normalize(self.value)
        self.assertEqual(result, self.value)
        self.assertIn('require investor verification', issues(result)[0])

    def test_missing_and_truncated_output_stays_visible(self):
        result = normalize({'sourcesMarkdown': '', 'chartSpecs': [], 'stopReason': 'max_tokens'})
        self.assertEqual(len(issues(result)), 4)

    def test_rejects_invalid_numbers_duplicates_and_unbounded_values(self):
        for number in (True, float('nan'), float('inf'), 10**400, '12'):
            value = copy.deepcopy(self.value)
            value['chartSpecs'][0]['data'][0]['value'] = number
            with self.assertRaises(ValueError): normalize(value)
        for field, bad in (('sourcesMarkdown', 'x'*500001), ('chartSpecs', self.value['chartSpecs']*2),
                           ('stopReason', None)):
            with self.assertRaises(ValueError): normalize({**self.value, field: bad})
