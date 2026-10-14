"""Context loading on disposable PostgreSQL; no production state or providers."""
import json
import unittest
from historical_valuation_context import load_context
from onepager_edition_contract import baseline, prepare_edition
from tests.unit.test_onepager_edition_contract import edition_fixture, AID, EID
from tests.unit import test_onepager_editions as editions
from tests.unit import test_stock_analysis_postgres as pg


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class HistoricalContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        editions.EditionPersistenceTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        pg.StockPostgresTests.tearDownClass.__func__(cls)

    def setUp(self):
        editions.EditionPersistenceTests.setUp(self)
        run, assignment, body, corrections, docs, financials = edition_fixture()
        financials['cik'] = 1
        assignment['result']['visualFundamentals'] = financials
        body.update({k:v for k,v in baseline(run, assignment, financials).items() if k != 'ticker'})
        edition = prepare_edition(run, assignment, body, corrections, docs, financials)
        with self.db(True) as (_, c):
            c.execute('UPDATE mp_jobs SET result=%s::jsonb WHERE id=%s', (json.dumps(assignment['result']), AID))
            c.execute('CREATE TABLE onepager_editions(id UUID PRIMARY KEY,assignment_id TEXT,run_id TEXT,record JSONB)')
            c.execute('INSERT INTO onepager_editions VALUES(%s,%s,%s,%s::jsonb)', (EID,AID,run['id'],json.dumps(edition)))

    def read(self, aid=AID, eid=EID):
        with self.db() as (_, c): return load_context(c, aid, eid)

    def test_loads_saved_identity_and_pending_target(self):
        frozen, identity = self.read()
        self.assertEqual(frozen['assignmentId'], AID)
        self.assertEqual(frozen['parentEditionId'], EID)
        self.assertEqual(identity, {'cik':1, 'symbol':'TEST'})
        self.assertNotIn('accepted', frozen)

    def test_stopped_and_foreign_targets_rejected(self):
        with self.assertRaises(ValueError): self.read(eid='00000000-0000-4000-8000-000000000099')
        with self.db(True) as (_, c): c.execute("UPDATE mp_jobs SET status='stopped'")
        with self.assertRaisesRegex(ValueError, 'stopped'): self.read()

    def test_changed_report_binding_rejected(self):
        with self.db(True) as (_, c): c.execute("UPDATE mp_jobs SET result=jsonb_set(result,'{reportId}','\"different\"')")
        with self.assertRaises(ValueError): self.read()

    def test_changed_fundamentals_rejected(self):
        with self.db(True) as (_, c): c.execute("UPDATE mp_jobs SET result=jsonb_set(result,'{visualFundamentals,cik}','2')")
        with self.assertRaises(ValueError): self.read()
