"""Synthetic request boundaries; no database, provider or model calls."""
import unittest
from unittest.mock import patch
from flask import Flask
from historical_valuation_api import create_blueprint
from tests.unit.test_onepager_edition_contract import AID, EID

class IngestionBoundaryTests(unittest.TestCase):
    def setUp(self):
        patcher=patch('historical_valuation_api.retrieve_once',side_effect=lambda db,aid,eid,rid,selection,fetch:fetch())
        patcher.start();self.addCleanup(patcher.stop)

    def client(self, auth=True, config=None):
        app=Flask(__name__); app.config['TESTING']=True
        app.register_blueprint(create_blueprint(None,lambda:auth,ingestion_config=lambda:config or {}))
        return app.test_client()

    @property
    def url(self):
        return f'/api/research/assignments/{AID}/onepager-editions/{EID}/historical-originals'

    @patch('historical_valuation_ingest.ingest_sec')
    def test_auth_and_default_disable_prevent_retrieval(self, fetch):
        self.assertEqual(self.client(False).post(self.url,json={'kind':'sec_companyfacts'}).status_code,403)
        self.assertEqual(self.client().post(self.url,json={'kind':'sec_companyfacts'}).status_code,503)
        fetch.assert_not_called()

    @patch('historical_valuation_ingest.ingest_sec',return_value={'id':'retained','sha256':'a'*64})
    def test_server_configuration_only_and_pending_result(self, fetch):
        c=self.client(config={'enabled':True,'sec_user_agent':'configured-contact'})
        for extra in ['url','api_key','user_agent','outputsize']:
            self.assertEqual(c.post(self.url,json={'kind':'sec_companyfacts',extra:'untrusted'}).status_code,409)
        fetch.assert_not_called()
        r=c.post(self.url,json={'kind':'sec_companyfacts'})
        self.assertEqual(r.status_code,200)
        self.assertFalse(r.json['valuationReady'])
        self.assertEqual(r.headers['Cache-Control'],'private, no-store')
        fetch.assert_called_once_with(None,AID,EID,user_agent='configured-contact',retrieval_id=None)

    @patch('historical_valuation_ingest.ingest_prices',side_effect=ValueError('private provider credential'))
    def test_price_entitlement_and_sanitized_no_retry(self, fetch):
        payload={'kind':'daily_prices','priceDate':'2026-10-09'}
        self.assertEqual(self.client(config={'enabled':True}).post(self.url,json=payload).status_code,503)
        fetch.assert_not_called()
        r=self.client(config={'enabled':True,'compact_prices_enabled':True,'price_api_key':'private'}).post(self.url,json=payload)
        self.assertEqual(r.status_code,409)
        self.assertNotIn('private',r.json['error'])
        fetch.assert_called_once_with(None,AID,EID,'2026-10-09',api_key='private',retrieval_id=None)

    def test_invalid_price_dates_cannot_create_unresolved_reservations(self):
        from datetime import datetime, timezone
        c=self.client(config={'enabled':True,'compact_prices_enabled':True,'price_api_key':'synthetic'})
        with patch('historical_valuation_api.retrieve_once') as reserve:
            for value in [None,True,'not-a-date','2025-02-30','20250303',datetime.now(timezone.utc).date().isoformat(),'2099-12-31']:
                with self.subTest(value=value):
                    response=c.post(self.url,json={'kind':'daily_prices','priceDate':value})
                    self.assertEqual(response.status_code,409)
            self.assertEqual(c.post(self.url,json={'kind':'daily_prices'}).status_code,409)
            reserve.assert_not_called()

if __name__=='__main__': unittest.main()
