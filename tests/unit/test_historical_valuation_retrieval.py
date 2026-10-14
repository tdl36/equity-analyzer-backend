"""Disposable database retrieval concurrency and interruption; no providers."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from historical_valuation_retrieval import retrieve_once
from tests.unit import test_historical_valuation_context as contexts
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit.test_onepager_edition_contract import AID,EID

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): contexts.HistoricalContextTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls): contexts.HistoricalContextTests.tearDownClass.__func__(cls)
    def setUp(self):
        contexts.HistoricalContextTests.setUp(self)
        with self.db(True) as (_,cur): cur.execute('DROP TABLE IF EXISTS historical_valuation_retrievals')
        self.rid='00000000-0000-4000-8000-000000000088'
    def run_fetch(self,fetch,selection=None):
        return retrieve_once(self.db,AID,EID,self.rid,selection or {'kind':'sec_companyfacts'},fetch)
    def test_replay_reuses_receipt_and_rejects_selection_change(self):
        fetch=Mock(return_value={'id':'source','sha256':'a'*64})
        self.assertEqual(self.run_fetch(fetch),self.run_fetch(fetch))
        fetch.assert_called_once()
        with self.assertRaises(ValueError): self.run_fetch(fetch,{'kind':'daily_prices'})
        fetch.assert_called_once()
    def test_interrupted_reservation_never_retries_provider(self):
        fetch=Mock(side_effect=RuntimeError('synthetic interruption'))
        with self.assertRaises(RuntimeError): self.run_fetch(fetch)
        with self.assertRaisesRegex(ValueError,'unresolved'): self.run_fetch(fetch)
        fetch.assert_called_once()
    def test_concurrent_request_calls_provider_once(self):
        fetch=Mock(return_value={'id':'source'})
        def attempt(_):
            try:return self.run_fetch(fetch)
            except ValueError:return None
        with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(attempt,range(2)))
        self.assertIn({'id':'source'},results)
        fetch.assert_called_once()
        self.assertEqual(self.run_fetch(fetch),{'id':'source'})
    def test_authenticated_read_only_inspection_without_provider_configuration(self):
        from flask import Flask
        from historical_valuation_api import create_blueprint
        def client(auth=True):
            app=Flask(__name__);app.config['TESTING']=True
            app.register_blueprint(create_blueprint(self.db,lambda:auth))
            return app.test_client()
        url=f'/api/research/assignments/{AID}/onepager-editions/{EID}/historical-retrievals'
        self.assertEqual(client(False).get(url).status_code,403)
        c=client();self.assertEqual(c.get(url).json['retrievals'],[])
        with self.db() as (_,cur):
            cur.execute("SELECT to_regclass('historical_valuation_retrievals') AS name")
            self.assertIsNone(cur.fetchone()['name'])
        fetch=Mock(side_effect=RuntimeError('sensitive provider failure'))
        with self.assertRaises(RuntimeError):self.run_fetch(fetch)
        r=c.get(url);self.assertEqual(r.status_code,200)
        self.assertEqual(r.headers['Cache-Control'],'private, no-store')
        row=r.json['retrievals'][0]
        self.assertEqual(row['status'],'unresolved');self.assertFalse(row['retryAllowed']);self.assertFalse(row['baselineChanged'])
        self.assertNotIn('sensitive',r.text);self.assertNotIn('result',row)
        self.assertFalse(r.json['valuationReady']);fetch.assert_called_once()
        with self.db() as (_,cur):
            cur.execute('SELECT status FROM historical_valuation_retrievals WHERE id=%s',(self.rid,))
            self.assertEqual(cur.fetchone()['status'],'reserved')
        with self.db(True) as (_,cur):cur.execute("UPDATE mp_jobs SET status='stopped'")
        self.assertEqual(c.get(url).status_code,409)

    def test_status_inventory_scoped_bounded_and_receipt_body_private(self):
        import json
        from historical_valuation_retrieval import inspect_retrievals
        self.run_fetch(Mock(return_value={'private':'source body'}))
        with self.db(True) as (_,cur):
            cur.execute('SELECT frozen FROM historical_valuation_retrievals WHERE id=%s',(self.rid,));frozen=cur.fetchone()['frozen']
            other=json.loads(json.dumps(frozen));other['baseline']['assignmentId']='00000000-0000-4000-8000-000000000099'
            cur.execute("INSERT INTO historical_valuation_retrievals(id,frozen,status) VALUES('00000000-0000-4000-8000-000000000099',%s::jsonb,'reserved')",(json.dumps(other),))
            for i in range(100,162):
                cur.execute("INSERT INTO historical_valuation_retrievals(id,frozen,status) VALUES(%s,%s::jsonb,'reserved')",(f'00000000-0000-4000-8000-{i:012d}',json.dumps(frozen)))
        with self.db() as (_,cur):result=inspect_retrievals(cur,AID,EID)
        self.assertTrue(result['truncated']);self.assertEqual(len(result['retrievals']),60)
        self.assertFalse(any(x['id'].endswith('000000000099') for x in result['retrievals']))
        self.assertNotIn('private',str(result));self.assertNotIn('source body',str(result))
    def test_new_identity_cannot_bypass_interrupted_selection(self):
        fetch=Mock(side_effect=RuntimeError('unknown provider outcome'))
        with self.assertRaises(RuntimeError):self.run_fetch(fetch)
        another='00000000-0000-4000-8000-000000000087'
        with self.assertRaisesRegex(ValueError,'earlier retrieval'):
            retrieve_once(self.db,AID,EID,another,{'kind':'sec_companyfacts'},fetch)
        fetch.assert_called_once()
        with self.db() as (_,cur):
            cur.execute('SELECT COUNT(*) AS n FROM historical_valuation_retrievals')
            self.assertEqual(cur.fetchone()['n'],1)

    def test_changed_price_date_cannot_bypass_unknown_compact_request(self):
        fetch=Mock(side_effect=RuntimeError('unknown compact-price outcome'))
        with self.assertRaises(RuntimeError):
            self.run_fetch(fetch,{'kind':'daily_prices','priceDate':'2025-03-03'})
        with self.assertRaisesRegex(ValueError,'earlier retrieval'):
            retrieve_once(self.db,AID,EID,'00000000-0000-4000-8000-000000000087',
                          {'kind':'daily_prices','priceDate':'2025-03-04'},fetch)
        fetch.assert_called_once()
        with self.db() as (_,cur):
            cur.execute('SELECT COUNT(*) AS n FROM historical_valuation_retrievals')
            self.assertEqual(cur.fetchone()['n'],1)
        # A separate source provider is not the uncertain compact-price request.
        separate=Mock(return_value={'id':'sec-original'})
        self.assertEqual(retrieve_once(self.db,AID,EID,
            '00000000-0000-4000-8000-000000000086',{'kind':'sec_companyfacts'},separate),
            {'id':'sec-original'})
        separate.assert_called_once()

    def test_inspection_exposes_older_edition_blocker_ahead_of_retained_receipts(self):
        import json
        from historical_valuation_retrieval import inspect_retrievals
        with self.assertRaises(RuntimeError):
            self.run_fetch(Mock(side_effect=RuntimeError('unknown outcome')))
        older='00000000-0000-4000-8000-000000000086'
        with self.db(True) as (_,cur):
            cur.execute('SELECT frozen FROM historical_valuation_retrievals WHERE id=%s',(self.rid,))
            current=cur.fetchone()['frozen']
            stale=json.loads(json.dumps(current));stale['baseline']['parentEditionId']=older
            cur.execute('UPDATE historical_valuation_retrievals SET frozen=%s::jsonb WHERE id=%s',(json.dumps(stale),self.rid))
            for i in range(100,162):
                cur.execute("INSERT INTO historical_valuation_retrievals(id,frozen,status) VALUES(%s,%s::jsonb,'retained')",(f'00000000-0000-4000-8000-{i:012d}',json.dumps(current)))
            cur.execute("INSERT INTO historical_valuation_retrievals(id,frozen,status) VALUES(%s,%s::jsonb,'retained')",(older,json.dumps(stale)))
        with self.db() as (_,cur):result=inspect_retrievals(cur,AID,EID)
        self.assertTrue(result['truncated'])
        blocker=result['retrievals'][0]
        self.assertEqual(blocker['id'],self.rid)
        self.assertEqual(blocker['parentEditionId'],older)
        self.assertEqual(blocker['status'],'unresolved')
        self.assertTrue(blocker['baselineChanged'])
        self.assertFalse(blocker['retryAllowed'])
        self.assertNotIn(older,[r['id'] for r in result['retrievals']])
        fetch=Mock(side_effect=AssertionError('provider must not run'))
        with self.assertRaisesRegex(ValueError,'earlier retrieval'):
            retrieve_once(self.db,AID,EID,'00000000-0000-4000-8000-000000000087',{'kind':'sec_companyfacts'},fetch)
        fetch.assert_not_called()

    def test_concurrent_distinct_identities_cannot_duplicate_unresolved_fetch(self):
        from threading import Event
        entered,release=Event(),Event()
        def fetch():
            entered.set();self.assertTrue(release.wait(5));return {'id':'retained'}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(self.run_fetch,fetch)
            self.assertTrue(entered.wait(5))
            try:
                with self.assertRaisesRegex(ValueError,'earlier retrieval'):
                    retrieve_once(self.db,AID,EID,'00000000-0000-4000-8000-000000000087',{'kind':'sec_companyfacts'},Mock(side_effect=AssertionError('duplicate provider call')))
            finally:release.set()
            self.assertEqual(first.result(),{'id':'retained'})
        self.assertEqual(self.run_fetch(Mock(side_effect=AssertionError('replay provider call'))),{'id':'retained'})
    def test_retained_receipt_replay_survives_a_later_unresolved_request(self):
        receipt={'id':'retained'};self.run_fetch(Mock(return_value=receipt))
        with self.assertRaises(RuntimeError):
            retrieve_once(self.db,AID,EID,'00000000-0000-4000-8000-000000000087',{'kind':'sec_companyfacts'},Mock(side_effect=RuntimeError('later request interrupted')))
        self.assertEqual(self.run_fetch(Mock(side_effect=AssertionError('provider replay'))),receipt)
    def test_ingestion_commit_survives_response_loss_without_provider_replay(self):
        import json
        from historical_valuation_ingest import ingest_sec
        from historical_valuation_context import load_context
        from tests.unit.test_historical_valuation_ingest import SecIngestTests
        with self.db(True) as (_,cur):
            cur.execute('DROP TABLE IF EXISTS historical_valuation_originals')
            _,identity=load_context(cur,AID,EID)
        raw=json.dumps({'cik':identity['cik'],'facts':{'us-gaap':{}}}).encode()
        session=Mock();session.get.return_value=SecIngestTests().response(raw)
        saved=[]
        def fetch_then_crash():
            saved.append(ingest_sec(self.db,AID,EID,user_agent='Synthetic contact@example.test',session=session,retrieval_id=self.rid))
            raise RuntimeError('process died after committed original and receipt')
        with self.assertRaises(RuntimeError):self.run_fetch(fetch_then_crash)
        self.assertEqual(self.run_fetch(Mock(side_effect=AssertionError('duplicate provider access'))),saved[0])
        session.get.assert_called_once()
        with self.db() as (_,cur):
            cur.execute('SELECT content,provenance FROM historical_valuation_originals WHERE assignment_id=%s',(AID,))
            row=cur.fetchone()
            self.assertEqual(bytes(row['content']),raw)
            self.assertEqual(row['provenance'],saved[0]['provenance'])
            self.assertFalse(saved[0]['provenanceVerified'])
            self.assertNotIn('synthetic-key',json.dumps(row['provenance']))

    def test_price_receipt_survives_response_loss_and_direct_replay_is_blocked(self):
        import json
        from historical_valuation_ingest import ingest_prices
        from tests.unit.test_historical_valuation_ingest import SecIngestTests
        from tests.unit.test_historical_price_evidence import PriceEvidenceTests
        raw=json.dumps(PriceEvidenceTests().fixture()).encode()
        session=Mock();session.get.return_value=SecIngestTests().response(raw)
        with self.db(True) as (_,cur):cur.execute('DROP TABLE IF EXISTS historical_valuation_originals')
        saved=[]
        def ingest():
            return ingest_prices(self.db,AID,EID,'2025-03-03',api_key='synthetic-key',session=session,retrieval_id=self.rid)
        def interrupted():
            saved.append(ingest())
            raise RuntimeError('response lost after commit')
        selection={'kind':'daily_prices','priceDate':'2025-03-03'}
        with self.assertRaises(RuntimeError):self.run_fetch(interrupted,selection)
        self.assertEqual(self.run_fetch(Mock(side_effect=AssertionError('duplicate call')),selection),saved[0])
        with self.assertRaisesRegex(ValueError,'already has an outcome'):ingest()
        session.get.assert_called_once()
        with self.db() as (_,cur):
            cur.execute('SELECT content,provenance FROM historical_valuation_originals WHERE assignment_id=%s',(AID,))
            row=cur.fetchone()
            self.assertEqual(bytes(row['content']),raw)
            self.assertEqual(row['provenance'],saved[0]['provenance'])
            self.assertFalse(saved[0]['provenanceVerified'])
            self.assertNotIn('synthetic-key',json.dumps(row['provenance']))

    def test_wrong_price_date_rejected_before_provider_access(self):
        from historical_valuation_ingest import ingest_prices
        with self.assertRaises(RuntimeError):
            self.run_fetch(Mock(side_effect=RuntimeError()),{'kind':'daily_prices','priceDate':'2025-03-03'})
        session=Mock()
        with self.assertRaisesRegex(ValueError,'source selection'):
            ingest_prices(self.db,AID,EID,'2025-03-04',api_key='synthetic-key',session=session,retrieval_id=self.rid)
        session.get.assert_not_called()

    def test_wrong_reservation_rejected_before_new_original(self):
        import json
        from historical_valuation_ingest import ingest_sec
        from historical_valuation_context import load_context
        from tests.unit.test_historical_valuation_ingest import SecIngestTests
        with self.db(True) as (_,cur):
            cur.execute('DROP TABLE IF EXISTS historical_valuation_originals')
            _,identity=load_context(cur,AID,EID)
        # A real reservation exists, but its source selection cannot be relabeled.
        with self.assertRaises(RuntimeError):self.run_fetch(Mock(side_effect=RuntimeError()),{'kind':'daily_prices','priceDate':'2025-03-03'})
        session=Mock();session.get.return_value=SecIngestTests().response(json.dumps({'cik':identity['cik'],'facts':{'us-gaap':{}}}).encode())
        with self.assertRaisesRegex(ValueError,'source selection'):
            ingest_sec(self.db,AID,EID,user_agent='Synthetic',session=session,retrieval_id=self.rid)
        session.get.assert_not_called()
        with self.db() as (_,cur):
            cur.execute("SELECT to_regclass('historical_valuation_originals') AS name")
            self.assertIsNone(cur.fetchone()['name'])
            cur.execute('SELECT status FROM historical_valuation_retrievals WHERE id=%s',(self.rid,))
            self.assertEqual(cur.fetchone()['status'],'reserved')
