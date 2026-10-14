"""Uses only disposable PostgreSQL and synthetic original bytes."""
import hashlib
import unittest
from concurrent.futures import ThreadPoolExecutor
from historical_valuation_source_store import save_original, read_originals, read_records
from tests.unit import test_stock_analysis_postgres as pg


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class HistoricalOriginalStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        pg.StockPostgresTests.tearDownClass.__func__(cls)

    def setUp(self):
        self.aid = '00000000-0000-4000-8000-000000000001'
        self.record = dict(assignmentId=self.aid, id='price', content=b'{}',
                           sha256=hashlib.sha256(b'{}').hexdigest(),
                           kind='daily_prices', retrievedDate='2026-10-12')
        self.refs = [{k:self.record[k] for k in ('id','sha256')}]
        with self.db(True) as (_, c):
            c.execute('DROP TABLE IF EXISTS historical_valuation_originals')

    def save(self, record=None):
        with self.db(True) as (_, c):
            return save_original(c, record or self.record)

    def read(self, aid=None):
        with self.db() as (_, c):
            return read_originals(c, self.refs, assignment_id=aid or self.aid)

    def test_absent_schema_read_is_nonmutating(self):
        self.assertEqual(self.read(), ({}, {}))
        with self.db() as (_, c):
            c.execute("SELECT to_regclass('historical_valuation_originals') AS name")
            self.assertIsNone(c.fetchone()['name'])

    def test_scope_exact_bytes_and_trusted_date(self):
        self.save()
        self.assertEqual(self.read(), ({'price':b'{}'}, {'price':'2026-10-12'}))
        self.assertEqual(self.read('00000000-0000-4000-8000-000000000002'), ({}, {}))

    def test_concurrent_replay_is_immutable(self):
        with ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(lambda _:self.save(), range(2)))
        self.assertEqual(sorted(r['reused'] for r in results), [False, True])
        for change in ({'retrievedDate':'2026-10-13'}, {'kind':'sec_companyfacts'},
                       {'content':b'[]', 'sha256':hashlib.sha256(b'[]').hexdigest()}):
            with self.assertRaises(ValueError): self.save({**self.record, **change})
        self.assertEqual(self.read()[0], {'price':b'{}'})

    def test_corrupt_bytes_cannot_be_read(self):
        self.save()
        with self.db(True) as (_, c):
            c.execute('UPDATE historical_valuation_originals SET content=%s', (b'[]',))
        with self.assertRaises(ValueError): self.read()

    def test_oversized_stored_body_rejected_before_loading(self):
        self.save()
        with self.db(True) as (_, c):
            c.execute("UPDATE historical_valuation_originals SET content=convert_to(repeat('x',15000001),'UTF8')")
        with self.assertRaisesRegex(ValueError, 'byte bound'): self.read()

    def test_rollback_retains_no_partial_ingestion(self):
        with self.assertRaises(RuntimeError):
            with self.db(True) as (_, c):
                save_original(c, self.record)
                raise RuntimeError('Synthetic interruption')
        self.assertEqual(self.read(), ({}, {}))

    def test_provenance_is_preserved_unverified_and_immutable(self):
        metadata = dict(provider='synthetic', observedUrl='https://example.test/prices',
                        retrievedAt='2026-10-12T12:00:00Z', collectorVersion='fixture-1')
        record = {**self.record, 'provenance': metadata}
        self.save(record)
        self.assertTrue(self.save(record)['reused'])
        with self.db() as (_, c):
            row = read_records(c, self.refs, assignment_id=self.aid)[0]
        self.assertEqual(row['provenance'], metadata)
        self.assertFalse(row['provenanceVerified'])
        for key, value in [('provider', 'other'), ('observedUrl', 'https://example.test/other'),
                           ('retrievedAt', '2026-10-12T13:00:00Z'), ('collectorVersion', 'fixture-2')]:
            with self.assertRaises(ValueError):
                self.save({**record, 'provenance': {**metadata, key: value}})
        with self.assertRaises(ValueError): self.save(self.record)

    def test_legacy_rows_cannot_be_relabelled(self):
        self.save()
        with self.db(True) as (_, c):
            c.execute('ALTER TABLE historical_valuation_originals DROP COLUMN provenance')
        with self.db() as (_, c):
            row = read_records(c, self.refs, assignment_id=self.aid)[0]
            self.assertIsNone(row['provenance'])
            self.assertFalse(row['provenanceVerified'])
        with self.assertRaises(ValueError):
            self.save({**self.record, 'provenance': dict(provider='synthetic',
                observedUrl='https://example.test/prices', retrievedAt='2026-10-12T12:00:00Z',
                collectorVersion='fixture-1')})
