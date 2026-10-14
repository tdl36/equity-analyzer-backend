"""Batch readiness with disposable PostgreSQL, no provider calls."""
import unittest
from unittest.mock import patch
from historical_valuation_candidate_context import readiness_from_storage, replay_readiness_from_storage
from tests.unit import test_historical_valuation_candidate_context as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit.test_onepager_edition_contract import AID, EID


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class ReadinessContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.CandidateContextTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls): fixtures.CandidateContextTests.tearDownClass.__func__(cls)
    setUp = fixtures.CandidateContextTests.setUp
    retain = fixtures.CandidateContextTests.retain

    def build(self, dates=None, baseline=None):
        with self.db() as (_, cur):
            return readiness_from_storage(cur, AID, EID, dates or ['2025-03-02','2025-03-03'], baseline or self.baseline)

    def replay(self, snapshot):
        with self.db() as (_, cur): return replay_readiness_from_storage(cur,AID,EID,snapshot)

    def test_absent_inventory_never_creates_schema(self):
        snapshot=self.build()
        self.assertEqual(snapshot['payload']['status'],'needs_evidence_review')
        self.assertEqual(snapshot,self.replay(snapshot))
        with self.db() as (_,cur):
            cur.execute("SELECT to_regclass('historical_valuation_originals') AS name")
            self.assertIsNone(cur.fetchone()['name'])

    def test_reads_bodies_once_and_preserves_partial_coverage(self):
        from historical_valuation_source_store import read_records
        self.retain()
        with patch('historical_valuation_candidate_context.read_records', wraps=read_records) as read:
            snapshot=self.build()
            self.assertEqual(read.call_count,1)
        rows=snapshot['payload']['observations']
        self.assertEqual([r['status'] for r in rows],['needs_evidence','selected'])
        self.assertFalse(snapshot['payload']['dispatchAllowed'])
        self.assertEqual(snapshot,self.replay(snapshot))

    def test_arrival_ambiguity_target_change_and_corruption_require_new_checkpoint(self):
        empty=self.build(); self.retain()
        with self.assertRaises(ValueError):self.replay(empty)
        selected=self.build()
        self.retain([{**self.records[0],'id':'alias'}])
        with self.assertRaises(ValueError):self.replay(selected)
        with self.assertRaises(ValueError):self.build(baseline={**self.baseline,'parentEditionDigest':'a'*64})
        with self.db(True) as (_,cur):cur.execute("UPDATE historical_valuation_originals SET content='broken'::bytea")
        with self.assertRaises(ValueError):self.build()

    def test_invalid_dates_fail_before_inventory_body_read(self):
        self.retain()
        with patch('historical_valuation_candidate_context.read_records') as read:
            with self.assertRaises(ValueError):self.build(['2025-03-03','2025-03-03'])
            read.assert_not_called()

    def test_assignment_and_originals_remain_locked_for_batch(self):
        from psycopg2.errors import LockNotAvailable
        from historical_valuation_context import load_context
        self.retain()
        with self.db() as (_,cur):
            readiness_from_storage(cur,AID,EID,['2025-03-02','2025-03-03'],self.baseline)
            # Trusted ingestion must acquire load_context's assignment lock before
            # inserting new receipts; this prevents phantom eligible originals.
            with self.assertRaises(LockNotAvailable):
                with self.db(True) as (_,other):
                    other.execute("SET LOCAL lock_timeout='100ms'")
                    load_context(other,AID,EID)
            with self.assertRaises(LockNotAvailable):
                with self.db(True) as (_,other):
                    other.execute("SET LOCAL lock_timeout='100ms'")
                    other.execute('DELETE FROM historical_valuation_originals WHERE assignment_id=%s',(AID,))
        self.assertEqual(self.build()['payload']['observations'][1]['status'],'selected')

    def test_foreign_and_oversize_inventory_fail_closed(self):
        self.retain([{**r,'assignmentId':'00000000-0000-4000-8000-000000000099'} for r in self.records])
        self.assertTrue(all(r['status']=='needs_evidence' for r in self.build()['payload']['observations']))
        self.retain([{**self.records[0],'id':f'price-{i}'} for i in range(61)])
        with self.assertRaisesRegex(ValueError,'planning bound'):self.build()
