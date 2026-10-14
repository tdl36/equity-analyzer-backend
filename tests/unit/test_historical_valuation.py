import copy
import unittest
from historical_valuation import check_observation


def observation():
    common = {'securityId': 'TEST-common', 'currency': 'USD', 'shareBasis': 'one common share',
              'source': {'id': 'synthetic', 'sha256': 'a' * 64}}
    return {'price': dict(common, basis='unadjusted_close', date='2025-03-03', value='100'),
            'earnings': dict(common, basis='GAAP diluted EPS', periodType='annual',
                             accession='0000000001-25-000001', publishedDate='2025-02-15',
                             periodStart='2024-01-01', periodEnd='2024-12-31', value='5')}


class HistoricalValuationTests(unittest.TestCase):
    def test_calculation_preserves_inputs_and_does_not_certify_sources(self):
        row = observation(); before = copy.deepcopy(row)
        result = check_observation(row)
        self.assertEqual(result['multiple'], '20.0000')
        self.assertEqual(result['status'], 'calculated')
        self.assertIn('still require review', result['message'])
        self.assertEqual(row, before)

    def test_later_restatement_and_same_day_publication_are_withheld(self):
        for published in ['2025-03-03', '2026-02-01']:
            row = observation(); row['earnings']['publishedDate'] = published
            self.assertEqual(check_observation(row)['status'], 'unverified')

    def test_incompatible_or_unspecified_security_currency_and_split_basis(self):
        for key in ['securityId', 'currency', 'shareBasis']:
            for value in [None, '', 'different']:
                row = observation(); row['earnings'][key] = value
                self.assertEqual(check_observation(row)['status'], 'unverified')

    def test_no_adjusted_price_forward_eps_or_ttm_substitution(self):
        for part, key, value in [('price', 'basis', 'adjusted_close'),
                                 ('earnings', 'basis', 'adjusted diluted EPS'),
                                 ('earnings', 'periodType', 'TTM')]:
            row = observation(); row[part][key] = value
            self.assertEqual(check_observation(row)['status'], 'unverified')

    def test_missing_receipts_malformed_dates_and_numbers_are_withheld(self):
        for part, key, value in [('price', 'source', {}), ('earnings', 'source', None),
                                 ('price', 'date', '2025-02-30'), ('price', 'value', True),
                                 ('price', 'value', 'NaN'), ('earnings', 'value', 5),
                                 ('earnings', 'accession', ''), ('earnings', 'periodStart', '2024-07-01'),
                                 ('price', 'date', '2027-03-03'), ('price', 'value', '0')]:
            row = observation(); row[part][key] = value
            self.assertEqual(check_observation(row)['status'], 'unverified')

    def test_nonpositive_earnings_not_a_zero_multiple(self):
        for value in ['0', '-5']:
            row = observation(); row['earnings']['value'] = value
            result = check_observation(row)
            self.assertEqual(result['status'], 'not_meaningful')
            self.assertNotIn('multiple', result)


class HistoricalSupplementTests(unittest.TestCase):
    def fixture(self):
        import hashlib
        row = observation()
        for part in ['price', 'earnings']:
            row[part]['source'] = {'id': part, 'sha256': hashlib.sha256(part.encode()).hexdigest()}
        return {'version': 1, 'securityId': 'TEST-common', 'observations': [row]}, {'price': b'price', 'earnings': b'earnings'}

    def check(self, data, originals):
        from historical_valuation import check_supplement
        return check_supplement(data, originals)

    def test_exact_originals_and_stable_digest_without_mutation(self):
        data, originals = self.fixture(); before = copy.deepcopy(data)
        result = self.check(data, originals)
        self.assertEqual(result['observations'][0]['status'], 'calculated')
        self.assertTrue(result['sourceReviewRequired'])
        self.assertTrue(result['investorReviewRequired'])
        self.assertEqual(data, before)
        reordered = dict(reversed(list(data.items())))
        self.assertEqual(result['digest'], self.check(reordered, originals)['digest'])
        data['observations'][0]['price']['value'] = '101'
        self.assertNotEqual(result['digest'], self.check(data, originals)['digest'])

    def test_absent_and_different_originals_withhold_ratio(self):
        for originals in [{}, {'price': b'changed', 'earnings': b'earnings'}, {'price': 'price', 'earnings': b'earnings'}]:
            data, _ = self.fixture(); row = self.check(data, originals)['observations'][0]
            self.assertEqual(row['status'], 'unverified'); self.assertNotIn('multiple', row)

    def test_duplicate_dates_never_silently_deduplicated_or_selected(self):
        data, originals = self.fixture(); data['observations'] *= 2
        rows = self.check(data, originals)['observations']
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['status'] == 'unverified' and 'multiple' not in r for r in rows))

    def test_cross_security_rows_and_malformed_rows_remain_visible(self):
        data, originals = self.fixture(); data['securityId'] = 'OTHER-common'
        self.assertEqual(self.check(data, originals)['observations'][0]['status'], 'unverified')
        data['observations'] = [None, {'price': []}]
        self.assertEqual(len(self.check(data, originals)['observations']), 2)

    def test_bounded_inventory_and_strict_version(self):
        for value in [[], [observation()] * 61, None]:
            data, originals = self.fixture(); data['observations'] = value
            with self.assertRaises(ValueError): self.check(data, originals)
        for value in [True, 1.0, 2]:
            data, originals = self.fixture(); data['version'] = value
            with self.assertRaises(ValueError): self.check(data, originals)
