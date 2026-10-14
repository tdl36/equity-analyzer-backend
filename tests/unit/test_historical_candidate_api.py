"""Read-only synthetic API and disposable PostgreSQL candidate preview."""
import json
import unittest
from unittest.mock import Mock
from flask import Flask
from historical_valuation_api import create_blueprint
from tests.unit import test_historical_valuation_candidate_context as context
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit.test_onepager_edition_contract import AID,EID


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class CandidateApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): context.CandidateContextTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls): context.CandidateContextTests.tearDownClass.__func__(cls)
    setUp = context.CandidateContextTests.setUp
    retain = context.CandidateContextTests.retain
    def client(self, auth=True, db=None):
        app=Flask(__name__);app.config['TESTING']=True
        app.register_blueprint(create_blueprint(db or self.db,lambda:auth,read_only=True))
        return app.test_client()

    @property
    def preview_url(self):return f'/api/research/assignments/{AID}/onepager-editions/{EID}/historical-candidate'

    def query(self):return {'selection':json.dumps({'baseline':self.baseline,'observation':self.selections[0]})}

    def test_readonly_preview_extracts_without_saving_and_does_not_return_original_bodies(self):
        self.retain();r=self.client().get(self.preview_url,query_string=self.query())
        self.assertEqual(r.status_code,200,r.json)
        self.assertEqual(r.headers['Cache-Control'],'private, no-store')
        self.assertEqual(r.json['payload']['observations'][0]['status'],'inputs_extracted')
        self.assertFalse(r.json['payload']['valuationReady'])
        self.assertNotIn('Time Series (Daily)',r.text);self.assertNotIn('us-gaap',r.text)
        self.assertEqual(self.client().post(self.preview_url,json={}).status_code,405)

    def test_missing_evidence_is_partial_and_stale_target_is_not_substituted(self):
        r=self.client().get(self.preview_url,query_string=self.query())
        self.assertEqual(r.status_code,200)
        self.assertEqual(len(r.json['payload']['observations'][0]['gaps']),2)
        self.baseline['fundamentalsDigest']='e'*64
        self.assertEqual(self.client().get(self.preview_url,query_string=self.query()).status_code,409)

    def test_auth_and_invalid_envelopes_never_read_database(self):
        db=Mock(side_effect=AssertionError('Unexpected database access'))
        self.assertEqual(self.client(False,db).get(self.preview_url,query_string=self.query()).status_code,403)
        for query in ({}, {'selection':'x'*4097}, [('selection','{}'),('selection','{}')],
                      {'selection':'{"baseline":{},"baseline":{},"observation":{}}'},
                      {'selection':'{"baseline":NaN,"observation":{}}'}, {'selection':'[]'}):
            self.assertEqual(self.client(True,db).get(self.preview_url,query_string=query).status_code,409)
        db.assert_not_called()

    def test_automatic_plan_reports_missing_unique_and_ambiguous_coverage(self):
        url=self.preview_url.replace('historical-candidate','historical-plan')
        query={'selection':json.dumps({'baseline':self.baseline,'date':'2025-03-03'})}
        client=self.client()
        missing=client.get(url,query_string=query)
        self.assertEqual(missing.status_code,200,missing.json)
        self.assertEqual(missing.json['payload']['status'],'needs_evidence')
        self.assertEqual(missing.headers['Cache-Control'],'private, no-store')
        self.retain()
        selected=client.get(url,query_string=query)
        self.assertEqual(selected.json['payload']['status'],'selected')
        self.assertFalse(selected.json['payload']['candidate']['payload']['valuationReady'])
        self.retain([{**self.records[0],'id':'alias'}])
        ambiguous=client.get(url,query_string=query)
        self.assertEqual(ambiguous.json['payload']['status'],'ambiguous')
        self.assertNotIn('candidate',ambiguous.json['payload'])
        self.assertEqual(client.post(url,json={}).status_code,405)
        self.baseline['parentEditionDigest']='f'*64
        self.assertEqual(client.get(url,query_string={'selection':json.dumps({'baseline':self.baseline,'date':'2025-03-03'})}).status_code,409)

    def test_automatic_plan_auth_and_envelope_fail_before_database(self):
        url=self.preview_url.replace('historical-candidate','historical-plan')
        db=Mock(side_effect=AssertionError('Unexpected database access'))
        self.assertEqual(self.client(False,db).get(url).status_code,403)
        for selection in ('[]','x'*4097,'{"date":"2025-03-03","date":"2025-03-04","baseline":{}}',json.dumps({'baseline':self.baseline,'date':'bad'})):
            self.assertEqual(self.client(True,db).get(url,query_string={'selection':selection}).status_code,409)
        db.assert_not_called()

    def test_batch_readiness_is_private_readonly_and_bound_to_target(self):
        url=self.preview_url.replace('historical-candidate','historical-readiness')
        query={'selection':json.dumps({'baseline':self.baseline,'dates':['2025-03-02','2025-03-03']})}
        self.retain()
        response=self.client().get(url,query_string=query)
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.headers['Cache-Control'],'private, no-store')
        self.assertEqual([r['status'] for r in response.json['payload']['observations']],['needs_evidence','selected'])
        self.assertFalse(response.json['payload']['dispatchAllowed'])
        self.assertFalse(response.json['payload']['valuationReady'])
        self.assertNotIn('Time Series (Daily)',response.text)
        self.assertEqual(self.client().post(url,json={}).status_code,405)
        self.baseline['fundamentalsDigest']='e'*64
        query['selection']=json.dumps({'baseline':self.baseline,'dates':['2025-03-03']})
        self.assertEqual(self.client().get(url,query_string=query).status_code,409)

    def test_batch_readiness_rejects_auth_and_invalid_dates_before_database(self):
        url=self.preview_url.replace('historical-candidate','historical-readiness')
        db=Mock(side_effect=AssertionError('Unexpected database access'))
        self.assertEqual(self.client(False,db).get(url).status_code,403)
        for dates in ([],['bad'],['2025-03-03']*2,[True],['2025-03-03']*31):
            query={'selection':json.dumps({'baseline':self.baseline,'dates':dates})}
            self.assertEqual(self.client(True,db).get(url,query_string=query).status_code,409)
        db.assert_not_called()

    def test_edition_readiness_rejects_dates_and_unauthenticated_reads(self):
        url=self.preview_url.replace('historical-candidate','historical-edition-readiness')
        db=Mock(side_effect=AssertionError('Unexpected database access'))
        self.assertEqual(self.client(False,db).get(url).status_code,403)
        for selection in ({'baseline':self.baseline,'dates':['2025-03-03']},{'baseline':{}},{}):
            self.assertEqual(self.client(True,db).get(url,query_string={'selection':json.dumps(selection)}).status_code,409)
        db.assert_not_called()

    def test_edition_readiness_uses_saved_dates_and_never_writes(self):
        from tests.unit.test_historical_valuation_readiness_context import ReadinessContextTests
        ReadinessContextTests.install_annuals(self,[{'start':'2024-03-04','end':'2025-03-03'}])
        url=self.preview_url.replace('historical-candidate','historical-edition-readiness')
        query={'selection':json.dumps({'baseline':self.baseline})}
        r=self.client().get(url,query_string=query)
        self.assertEqual(r.status_code,200,r.json)
        self.assertEqual(r.headers['Cache-Control'],'private, no-store')
        self.assertEqual(r.json['datePolicy']['payload']['dates'],['2025-03-03'])
        self.assertFalse(r.json['dispatchAllowed'])
        self.assertEqual(self.client().post(url,json={}).status_code,405)
