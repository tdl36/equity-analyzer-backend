"""Storage assembly on disposable PostgreSQL and synthetic source bytes only."""
import copy
import unittest
from historical_valuation_candidate_context import assemble_from_storage, replay_from_storage
from historical_valuation_context import load_context
from historical_valuation_source_store import save_original
from tests.unit import test_historical_valuation_context as contexts
from tests.unit import test_historical_valuation_candidates as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit.test_onepager_edition_contract import AID, EID


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class CandidateContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): contexts.HistoricalContextTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls): contexts.HistoricalContextTests.tearDownClass.__func__(cls)

    def setUp(self):
        contexts.HistoricalContextTests.setUp(self)
        f=fixtures.CandidateTests();f.setUp()
        self.selections=f.selections
        self.records=[{**r,'assignmentId':AID} for r in f.records]
        with self.db(True) as (_,cur):
            cur.execute('DROP TABLE IF EXISTS historical_valuation_originals')
            self.baseline,_=load_context(cur,AID,EID)

    def retain(self, records=None):
        with self.db(True) as (_,cur):
            for record in self.records if records is None else records:save_original(cur,record)

    def assemble(self, baseline=None):
        with self.db() as (_,cur):return assemble_from_storage(cur,AID,EID,self.selections,baseline or self.baseline)

    def replay(self, snapshot):
        with self.db() as (_,cur):return replay_from_storage(cur,AID,EID,snapshot)

    def test_no_schema_creation_when_evidence_absent(self):
        result=self.assemble()
        self.assertEqual(len(result['payload']['observations'][0]['gaps']),2)
        with self.db() as (_,cur):
            cur.execute("SELECT to_regclass('historical_valuation_originals') AS name")
            self.assertIsNone(cur.fetchone()['name'])
        self.assertEqual(result,self.replay(result))

    def test_extracts_scoped_records_without_writing_originals_or_target(self):
        self.retain()
        def stored():
            with self.db() as (_,cur):
                cur.execute('SELECT sha256,content FROM historical_valuation_originals ORDER BY source_id')
                originals=cur.fetchall()
                cur.execute('SELECT status,input,result FROM mp_jobs WHERE id=%s',(AID,))
                return originals,cur.fetchone()
        before=stored();result=self.assemble()
        self.assertEqual(result['payload']['observations'][0]['status'],'inputs_extracted')
        self.assertEqual(result,self.replay(result));self.assertEqual(before,stored())
        self.assertNotIn('content',str(result['payload'].keys()))

    def test_cross_assignment_originals_remain_missing(self):
        self.retain([{**r,'assignmentId':'00000000-0000-4000-8000-000000000099'} for r in self.records])
        self.assertEqual(len(self.assemble()['payload']['observations'][0]['gaps']),2)

    def test_source_arrival_requires_a_new_candidate_version(self):
        first=self.assemble();self.retain()
        with self.assertRaises(ValueError):self.replay(first)
        self.assertNotEqual(first['candidateDigest'],self.assemble()['candidateDigest'])

    def test_changed_baseline_stopped_target_and_corrupted_bytes_fail_closed(self):
        wrong={**self.baseline,'sourcePackFingerprint':'e'*64}
        with self.assertRaises(ValueError):self.assemble(wrong)
        self.retain()
        with self.db(True) as (_,cur):cur.execute("UPDATE historical_valuation_originals SET content='corrupt'::bytea")
        with self.assertRaises(ValueError):self.assemble()
        with self.db(True) as (_,cur):cur.execute("UPDATE mp_jobs SET status='stopped' WHERE id=%s",(AID,))
        with self.assertRaises(ValueError):self.assemble()

    def test_snapshot_qualifiers_or_digest_cannot_be_changed(self):
        self.retain();snapshot=self.assemble()
        changed=copy.deepcopy(snapshot);changed['payload']['valuationReady']=True
        with self.assertRaises(ValueError):self.replay(changed)
        changed=copy.deepcopy(snapshot);changed['candidateDigest']='0'*64
        with self.assertRaises(ValueError):self.replay(changed)

    def test_target_and_source_locks_remain_held_through_caller_transaction(self):
        from psycopg2.errors import LockNotAvailable
        self.retain()
        with self.db() as (_,cur):
            assemble_from_storage(cur,AID,EID,self.selections,self.baseline)
            for query,args in [
                ("UPDATE mp_jobs SET status='stopped' WHERE id=%s",(AID,)),
                ("UPDATE historical_valuation_originals SET content='changed'::bytea WHERE assignment_id=%s",(AID,))]:
                with self.assertRaises(LockNotAvailable):
                    with self.db(True) as (_,other):
                        other.execute("SET LOCAL lock_timeout='100ms'")
                        other.execute(query,args)
        self.assertEqual(self.assemble()['payload']['observations'][0]['status'],'inputs_extracted')

    def test_planner_uses_complete_scoped_inventory_without_creating_schema(self):
        from historical_valuation_candidate_context import plan_from_storage
        def plan():
            with self.db() as (_,cur):return plan_from_storage(cur,AID,EID,'2025-03-03',self.baseline)
        self.assertEqual(plan()['payload']['status'],'needs_evidence')
        with self.db() as (_,cur):
            cur.execute("SELECT to_regclass('historical_valuation_originals') AS name")
            self.assertIsNone(cur.fetchone()['name'])
        self.retain()
        self.assertEqual(plan()['payload']['status'],'selected')
        self.retain([{**self.records[0],'id':'price-alias'}])
        self.assertEqual(plan()['payload']['status'],'ambiguous')
        with self.db(True) as (_,cur):
            for i in range(60):save_original(cur,{**self.records[0],'id':f'price-{i}'})
        with self.assertRaisesRegex(ValueError,'planning bound'):plan()
