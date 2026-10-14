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
