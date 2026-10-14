import copy
import hashlib
import json
import unittest
from historical_valuation_evidence import daily_close_from_original


class PriceEvidenceTests(unittest.TestCase):
    def fixture(self):
        return {'Meta Data': {'1. Information': 'Daily Prices (open, high, low, close) and Volumes',
                             '2. Symbol': 'TEST', '3. Last Refreshed': '2025-03-03'},
                'Time Series (Daily)': {'2025-03-03': {'1. open': '98', '2. high': '101',
                    '3. low': '97', '4. close': '100.0000', '5. volume': '10000'}}}

    def extract(self, raw, **kw):
        content = json.dumps(raw).encode() if isinstance(raw, dict) else raw
        receipt = {'id': 'price', 'sha256': hashlib.sha256(content).hexdigest()}
        return daily_close_from_original(content, kw.get('receipt', receipt), kw.get('symbol', 'TEST'),
                                         kw.get('date', '2025-03-03'), kw.get('retrieved', '2025-03-04'))

    def test_exact_raw_close_preserves_input_and_does_not_infer_identity(self):
        raw = self.fixture(); before = copy.deepcopy(raw)
        result = self.extract(raw)
        self.assertEqual(result['status'], 'extracted')
        self.assertEqual(result['price']['value'], '100.0000')
        self.assertEqual(raw, before)
        for key in ('currency', 'securityId', 'shareBasis'):
            self.assertNotIn(key, result['price'])

    def test_missing_date_and_same_day_are_not_substituted(self):
        for kw in [{'date': '2025-03-02'}, {'retrieved': '2025-03-03'},
                   {'retrieved': '2025-03-01'}, {'date': '2025-03-05'}, {'date': 'bad'}]:
            self.assertEqual(self.extract(self.fixture(), **kw)['status'], 'unverified')

    def test_adjusted_series_provider_errors_and_wrong_symbol_are_withheld(self):
        raw = self.fixture(); raw['Time Series (Daily)']['2025-03-03']['5. adjusted close'] = '50'
        for data in [raw, {'Information': 'Rate limit'}, {'Error Message': 'bad symbol'}]:
            self.assertEqual(self.extract(data)['status'], 'unverified')
        self.assertEqual(self.extract(self.fixture(), symbol='OTHER')['status'], 'unverified')

    def test_invalid_values_and_ohlc_consistency(self):
        for key, value in [('4. close', True), ('4. close', 'NaN'), ('4. close', '102'),
                           ('3. low', '0'), ('5. volume', '0'), ('5. volume', '-1')]:
            raw = self.fixture(); raw['Time Series (Daily)']['2025-03-03'][key] = value
            self.assertEqual(self.extract(raw)['status'], 'unverified')

    def test_duplicate_dates_and_receipt_mismatch_fail_closed(self):
        self.assertEqual(self.extract(b'{"Meta Data":{},"Meta Data":{}}')['status'], 'unverified')
        self.assertEqual(self.extract(self.fixture(), receipt={'id': 'price', 'sha256': '0'*64})['status'], 'unverified')

    def test_response_vintage_cannot_precede_inventory_or_follow_retrieval(self):
        for refreshed in ['2025-03-02', '2025-03-05', 'bad']:
            raw = self.fixture(); raw['Meta Data']['3. Last Refreshed'] = refreshed
            self.assertEqual(self.extract(raw)['status'], 'unverified')
        raw = self.fixture(); raw['Time Series (Daily)']['2025-03-04'] = {}
        self.assertEqual(self.extract(raw)['status'], 'unverified')


class MarketBindingTests(unittest.TestCase):
    def fixture(self):
        from tests.unit.test_historical_valuation_evidence import SECBindingTests
        data, originals = SECBindingTests().fixture()
        content = json.dumps(PriceEvidenceTests().fixture()).encode()
        originals['price'] = content
        data['observations'][0]['price']['source']['sha256'] = hashlib.sha256(content).hexdigest()
        data['observations'][0]['price']['value'] = '100'
        return data, originals

    def check(self, data, originals, **kw):
        from historical_valuation_evidence import check_market_supplement
        return check_market_supplement(data, originals, kw.get('cik', 1), kw.get('symbol', 'TEST'),
                                       kw.get('dates', {'price': '2025-03-04'}))

    def test_both_amounts_bind_without_claiming_valuation_readiness(self):
        data, originals = self.fixture(); before = copy.deepcopy(data)
        result = self.check(data, originals); row = result['observations'][0]
        self.assertEqual(row['priceExtraction'], 'matched_raw_daily_close')
        self.assertEqual(row['earningsExtraction'], 'matched_sec_annual_fact')
        self.assertEqual(row['multiple'], '20.0000')
        self.assertFalse(result['valuationReady'])
        self.assertIn('corporate_actions', result['unresolvedChecks'])
        self.assertTrue(result['investorReviewRequired']); self.assertEqual(data, before)

    def test_tampered_proposed_price_and_earnings_remove_ratio(self):
        for part, value in [('price', '99'), ('earnings', '6')]:
            data, originals = self.fixture(); data['observations'][0][part]['value'] = value
            row = self.check(data, originals)['observations'][0]
            self.assertEqual(row['status'], 'unverified'); self.assertNotIn('multiple', row)

    def test_receipt_date_is_required_independently_of_proposal(self):
        for dates in [{}, {'price': '2025-03-03'}]:
            data, originals = self.fixture()
            data['observations'][0]['price']['retrievedDate'] = '2025-03-04'
            row = self.check(data, originals, dates=dates)['observations'][0]
            self.assertEqual(row['status'], 'unverified'); self.assertNotIn('multiple', row)

    def test_issuer_and_symbol_must_both_match(self):
        for kw in [{'cik': 2}, {'symbol': 'OTHER'}]:
            data, originals = self.fixture()
            self.assertEqual(self.check(data, originals, **kw)['observations'][0]['status'], 'unverified')

    def test_edited_price_original_cannot_retain_calculation(self):
        data, originals = self.fixture(); originals['price'] += b' '
        row = self.check(data, originals)['observations'][0]
        self.assertEqual(row['status'], 'unverified'); self.assertNotIn('multiple', row)

    def test_negative_earnings_stay_not_meaningful_when_both_sources_match(self):
        data, originals = self.fixture()
        sec = json.loads(originals['sec'])
        sec['facts']['us-gaap']['EarningsPerShareDiluted']['units']['USD/shares'][0]['val'] = -5
        originals['sec'] = json.dumps(sec).encode()
        earnings = data['observations'][0]['earnings']; earnings['value'] = '-5'
        earnings['source']['sha256'] = hashlib.sha256(originals['sec']).hexdigest()
        row = self.check(data, originals)['observations'][0]
        self.assertEqual(row['status'], 'not_meaningful'); self.assertNotIn('multiple', row)
        self.assertEqual(row['priceExtraction'], 'matched_raw_daily_close')

    def test_evidence_identity_binds_receipts_retrieval_and_issuer_context(self):
        data, originals = self.fixture()
        result = self.check(data, originals)
        for kw in [{'dates': {'price': '2025-03-05'}}, {'cik': 2}, {'symbol': 'OTHER'}]:
            changed = self.check(data, originals, **kw)
            self.assertEqual(changed['digest'], result['digest'])
            self.assertNotEqual(changed['evidenceDigest'], result['evidenceDigest'])
        originals['price'] += b' '
        self.assertNotEqual(self.check(data, originals)['evidenceDigest'], result['evidenceDigest'])

    def test_evidence_digest_is_deterministic_and_missing_original_is_explicit(self):
        data, originals = self.fixture()
        result = self.check(data, originals)
        reordered = dict(reversed(list(originals.items())))
        self.assertEqual(self.check(data, reordered)['evidenceDigest'], result['evidenceDigest'])
        originals.pop('price')
        missing = self.check(data, originals)
        self.assertIsNone(missing['evidenceContext']['originalHashes']['price'])
        self.assertNotEqual(missing['evidenceDigest'], result['evidenceDigest'])
        self.assertFalse(missing['valuationReady'])

    def test_invalid_or_unbounded_context_is_rejected(self):
        data, originals = self.fixture()
        for kw in [{'cik': True}, {'cik': -1}, {'symbol': ''}, {'symbol': 'X'*31},
                   {'dates': {'price': 'bad'}}, {'dates': {1: '2025-03-04'}},
                   {'dates': {str(i): '2025-03-04' for i in range(61)}}]:
            with self.assertRaises(ValueError): self.check(data, originals, **kw)

    def test_returned_context_cannot_mutate_receipt_inventory(self):
        data, originals = self.fixture(); dates = {'price': '2025-03-04'}
        result = self.check(data, originals, dates=dates)
        result['evidenceContext']['priceRetrievalDates']['price'] = '2025-03-05'
        self.assertEqual(dates, {'price': '2025-03-04'})
