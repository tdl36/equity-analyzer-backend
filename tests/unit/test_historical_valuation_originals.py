import hashlib
import unittest
from historical_valuation_originals import load_originals


class OriginalBindingTests(unittest.TestCase):
    def setUp(self):
        self.raw = b'{"immutable":"original"}'
        self.ref = {'id': 'price', 'sha256': hashlib.sha256(self.raw).hexdigest()}
        self.record = dict(self.ref, assignmentId='assignment', content=self.raw,
                           kind='daily_prices', retrievedDate='2025-03-04')

    def load(self, records=None, refs=None):
        return load_originals(refs or [self.ref], [self.record] if records is None else records,
                              assignment_id='assignment')

    def test_only_trusted_retrieval_date_reaches_diagnostic(self):
        self.ref['retrievedDate'] = '2099-01-01'
        self.ref['url'] = 'file:///never-open'
        raw, dates = self.load()
        self.assertEqual(raw, {'price': self.raw})
        self.assertEqual(dates, {'price': '2025-03-04'})

    def test_missing_original_is_not_fabricated(self):
        self.assertEqual(self.load([]), ({}, {}))

    def test_changed_bytes_or_receipt_rejected(self):
        for field, value in [('content', b'changed'), ('sha256', 'a'*64),
                             ('assignmentId', 'other'), ('kind', 'generated_summary'),
                             ('retrievedDate', 'yesterday')]:
            with self.subTest(field=field):
                record = dict(self.record); record[field] = value
                with self.assertRaises(ValueError): self.load([record])

    def test_conflicting_and_duplicate_inventory_rejected(self):
        with self.assertRaises(ValueError): self.load([self.record, self.record])
        with self.assertRaises(ValueError): self.load(refs=[self.ref, dict(self.ref, sha256='b'*64)])
        with self.assertRaises(ValueError): self.load([dict(self.record, id='other')])

    def test_sec_receipt_does_not_supply_price_vintage(self):
        self.record['kind'] = 'sec_companyfacts'
        self.assertEqual(self.load()[1], {})

    def test_per_original_and_inventory_limits(self):
        with self.assertRaises(ValueError): self.load([dict(self.record, content=b'x'*15000001)])
        with self.assertRaises(ValueError): self.load([self.record]*61)
        with self.assertRaises(ValueError): self.load(refs=[self.ref]*61)

    def test_duplicate_consistent_references_share_one_original(self):
        self.assertEqual(self.load(refs=[self.ref, dict(self.ref)])[0], {'price': self.raw})
