"""Disposable PostgreSQL only; never use the application database or providers."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from historical_valuation_store import save_pending, read_pending
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit import test_historical_valuation_review as fixtures


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class HistoricalStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        pg.StockPostgresTests.tearDownClass.__func__(cls)

    def setUp(self):
        fixture = fixtures.PendingHistoricalTests(); fixture.setUp()
        self.data, self.originals = fixture.data, fixture.originals
        self.baseline, self.context = fixture.baseline, fixture.context
        self.ident = '00000000-0000-4000-8000-000000000003'
        with self.db(True) as (_, c): c.execute('DROP TABLE IF EXISTS historical_valuation_reviews')

    def save(self):
        with self.db(True) as (_, c):
            return save_pending(c, self.ident, self.data, self.originals, self.baseline, **self.context)

    def read(self):
        with self.db() as (_, c):
            return read_pending(c, self.ident, self.originals, self.baseline, **self.context)

    def test_read_does_not_create_schema(self):
        self.assertIsNone(self.read())
        with self.db() as (_, c):
            c.execute("SELECT to_regclass('historical_valuation_reviews') AS name")
            self.assertIsNone(c.fetchone()['name'])

    def test_saved_pending_reopens_without_acceptance(self):
        first = self.save(); self.assertFalse(first['reused'])
        self.assertTrue(self.save()['reused'])
        result = self.read()
        self.assertEqual(result['reviewDigest'], first['reviewDigest'])
        self.assertEqual(result['payload']['status'], 'pending')
        self.assertFalse(result['payload']['diagnosis']['valuationReady'])

    def test_concurrent_duplicate_submission_has_one_immutable_record(self):
        with ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(lambda _: self.save(), range(2)))
        self.assertEqual(sorted(x['reused'] for x in results), [False, True])
        with self.db() as (_, c):
            c.execute('SELECT COUNT(*) AS n FROM historical_valuation_reviews')
            self.assertEqual(c.fetchone()['n'], 1)

    def test_changed_submission_cannot_overwrite_existing_snapshot(self):
        self.save(); original = self.read()
        self.data['observations'][0]['price']['value'] = '101'
        with self.assertRaises(ValueError): self.save()
        self.assertEqual(self.read(), original)

    def test_evidence_and_parent_changes_block_replay(self):
        self.save(); self.originals['price'] += b' '
        with self.assertRaises(ValueError): self.read()
        self.originals['price'] = self.originals['price'][:-1]
        self.baseline['parentEditionId'] = '00000000-0000-4000-8000-000000000004'
        with self.assertRaises(ValueError): self.read()

    def test_transaction_failure_leaves_no_partial_review(self):
        with self.assertRaises(RuntimeError):
            with self.db(True) as (_, c):
                save_pending(c, self.ident, self.data, self.originals, self.baseline, **self.context)
                raise RuntimeError('Synthetic failure before commit')
        self.assertIsNone(self.read())
        self.assertFalse(self.save()['reused'])
