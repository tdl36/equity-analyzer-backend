"""Synthetic HTTP plus disposable PostgreSQL only; no provider calls."""
import unittest
from flask import Flask
from historical_valuation_api import create_blueprint
from historical_valuation_source_store import save_original
from tests.unit import test_historical_valuation_context as contexts
from tests.unit import test_historical_valuation_review as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit.test_onepager_edition_contract import AID, EID


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class HistoricalApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): contexts.HistoricalContextTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls): contexts.HistoricalContextTests.tearDownClass.__func__(cls)

    def setUp(self):
        contexts.HistoricalContextTests.setUp(self)
        f = fixtures.PendingHistoricalTests(); f.setUp()
        self.proposal, self.originals = f.data, f.originals
        self.url = f'/api/research/assignments/{AID}/onepager-editions/{EID}/historical-reviews/00000000-0000-4000-8000-000000000077'
        with self.db(True) as (_, c):
            c.execute('DROP TABLE IF EXISTS historical_valuation_reviews; DROP TABLE IF EXISTS historical_valuation_originals')

    def client(self, auth=True, read_only=False):
        app=Flask(__name__); app.config['TESTING']=True
        app.register_blueprint(create_blueprint(self.db, lambda:auth, read_only=read_only))
        return app.test_client()

    def test_auth_blocks_reads_and_writes_before_database_access(self):
        for method in ('get','post'):
            r=getattr(self.client(False),method)(self.url)
            self.assertEqual(r.status_code,403)
            self.assertEqual(r.headers['Cache-Control'],'private, no-store')

    def test_missing_sources_remain_pending_and_replay_is_idempotent(self):
        c=self.client(); r=c.post(self.url,json=self.proposal)
        self.assertEqual(r.status_code,201,r.json)
        self.assertEqual(c.post(self.url,json=self.proposal).json['reused'],True)
        r=c.get(self.url); self.assertEqual(r.status_code,200,r.json)
        self.assertFalse(r.json['snapshot']['payload']['diagnosis']['valuationReady'])
        self.assertEqual(r.json['status'],'pending')

    def test_all_read_inventories_bind_to_the_same_frozen_baseline(self):
        c = self.client(read_only=True)
        base = self.url.rsplit('/historical-reviews/', 1)[0]
        inventories = [c.get(base + '/' + path) for path in
                       ('historical-originals', 'historical-retrievals', 'historical-reviews')]
        for response in inventories:
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
        from historical_valuation_context import load_context
        with self.db() as (_, cur):
            expected, _ = load_context(cur, AID, EID)
        self.assertTrue(all(response.json['baseline'] == expected for response in inventories))
        self.assertEqual(c.post(self.url, json=self.proposal).status_code, 405)
        self.assertEqual(self.client().post(self.url, json=self.proposal).status_code, 201)
        self.assertEqual(c.get(base + '/historical-reviews').json['baseline'], expected)

    def test_review_inventory_is_read_only_private_and_recheck_required(self):
        c=self.client();url=self.url.rsplit('/',1)[0]
        self.assertEqual(self.client(False).get(url).status_code,403)
        self.assertEqual(c.get(url).json['reviews'],[])
        with self.db() as (_,cur):
            cur.execute("SELECT to_regclass('historical_valuation_reviews') AS name")
            self.assertIsNone(cur.fetchone()['name'])
        self.assertEqual(c.post(self.url,json=self.proposal).status_code,201)
        r=c.get(url);self.assertEqual(r.status_code,200)
        self.assertEqual(r.headers['Cache-Control'],'private, no-store')
        row=r.json['reviews'][0]
        self.assertEqual(row['id'],self.url.rsplit('/',1)[1])
        self.assertEqual(row['status'],'pending')
        self.assertTrue(row['requiresEvidenceRecheck'])
        self.assertFalse(row['baselineChanged'])
        self.assertNotIn('proposal',r.text);self.assertNotIn('observations',r.text)
        self.assertFalse(r.json['valuationReady'])

    def test_review_inventory_is_scoped_bounded_and_does_not_validate_changed_evidence(self):
        c=self.client();url=self.url.rsplit('/',1)[0]
        self.assertEqual(c.post(self.url,json=self.proposal).status_code,201)
        with self.db(True) as (_,cur):
            cur.execute("UPDATE historical_valuation_reviews SET snapshot=jsonb_set(snapshot,'{payload,baseline,parentEditionDigest}',to_jsonb(repeat('0',64)))")
            for i in range(100,162):
                cur.execute('''INSERT INTO historical_valuation_reviews
                    SELECT %s,assignment_id,parent_edition_id,review_digest,snapshot,NOW()
                    FROM historical_valuation_reviews WHERE id=%s''',
                    (f'00000000-0000-4000-8000-{i:012d}',self.url.rsplit('/',1)[1]))
            cur.execute("UPDATE historical_valuation_reviews SET assignment_id='00000000-0000-4000-8000-000000000099' WHERE id=%s",(self.url.rsplit('/',1)[1],))
        result=c.get(url).json
        self.assertTrue(result['truncated']);self.assertEqual(len(result['reviews']),60)
        self.assertTrue(all(r['baselineChanged'] for r in result['reviews']))
        self.assertNotIn(self.url.rsplit('/',1)[1],[r['id'] for r in result['reviews']])
        with self.db(True) as (_,cur):cur.execute("UPDATE mp_jobs SET status='stopped'")
        self.assertEqual(c.get(url).status_code,409)

    def test_new_source_arrival_requires_new_version(self):
        c=self.client(); self.assertEqual(c.post(self.url,json=self.proposal).status_code,201)
        import hashlib
        raw=self.originals['price']
        with self.db(True) as (_,cur):
            save_original(cur,dict(assignmentId=AID,id='price',sha256=hashlib.sha256(raw).hexdigest(),
                content=raw,kind='daily_prices',retrievedDate='2025-03-04'))
        self.assertEqual(c.get(self.url).status_code,409)
        self.assertEqual(c.post(self.url,json=self.proposal).status_code,409)

    def test_stopped_assignment_cannot_save(self):
        with self.db(True) as (_,c): c.execute("UPDATE mp_jobs SET status='stopped'")
        self.assertEqual(self.client().post(self.url,json=self.proposal).status_code,409)

    def test_missing_review_and_conflicting_submission(self):
        c=self.client(); self.assertEqual(c.get(self.url).status_code,404)
        self.assertEqual(c.post(self.url,json=self.proposal).status_code,201)
        self.proposal['observations'][0]['price']['value']='101'
        self.assertEqual(c.post(self.url,json=self.proposal).status_code,409)

    def test_inventory_is_target_scoped_and_readable_without_provider_configuration(self):
        import hashlib
        url=f'/api/research/assignments/{AID}/onepager-editions/{EID}/historical-originals'
        c=self.client()
        self.assertEqual(c.get(url).json['originals'],[])
        raw=self.originals['price']
        with self.db(True) as (_,cur):
            for aid, sid in [(AID,'price'),('00000000-0000-4000-8000-000000000099','foreign')]:
                save_original(cur,dict(assignmentId=aid,id=sid,sha256=hashlib.sha256(raw).hexdigest(),
                    content=raw,kind='daily_prices',retrievedDate='2025-03-04'))
        r=c.get(url)
        self.assertEqual(r.status_code,200)
        self.assertEqual([x['id'] for x in r.json['originals']],['price'])
        self.assertNotIn('content',r.json['originals'][0])
        self.assertFalse(r.json['valuationReady'])
        self.assertEqual(self.client(False).get(url).status_code,403)
        with self.db(True) as (_,cur): cur.execute("UPDATE mp_jobs SET status='stopped'")
        self.assertEqual(c.get(url).status_code,409)
