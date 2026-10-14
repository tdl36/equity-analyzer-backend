"""Synthetic transport only: never retrieve live SEC or paid sources for QA."""
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch
import requests
from historical_valuation_ingest import ingest_sec


class SecIngestTests(unittest.TestCase):
    @contextmanager
    def db(self, commit=False): yield None, object()

    def response(self, raw=b'{"cik":1,"facts":{"us-gaap":{}}}', status=200):
        response=Mock(status_code=status)
        response.iter_content.return_value=[raw]
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        return response

    def run_ingest(self, response=None, contexts=None):
        session=Mock();session.get.return_value=response or self.response()
        with patch('historical_valuation_ingest.load_context', side_effect=contexts or [({'a':1},{'cik':1}),({'a':1},{'cik':1})]), patch('historical_valuation_ingest.save_original',return_value={'reused':False}) as save:
            result=ingest_sec(self.db,'aid','eid',user_agent='Charlie contact@example.test',session=session)
            return result,session,save.call_args.args[1]

    def test_fixed_official_url_and_exact_original_bytes(self):
        result,session,record=self.run_ingest()
        args,kwargs=session.get.call_args
        self.assertEqual(args[0],'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json')
        self.assertFalse(kwargs['allow_redirects'])
        self.assertEqual(record['content'],b'{"cik":1,"facts":{"us-gaap":{}}}')
        self.assertEqual(result['retrievedDate'],record['retrievedDate'])
        self.assertEqual(record['provenance']['observedUrl'],args[0])
        self.assertEqual(record['provenance']['provider'],'sec')
        self.assertEqual(result['provenance'],record['provenance'])
        self.assertFalse(result['provenanceVerified'])

    def test_identical_bytes_in_separate_acquisitions_keep_distinct_identity(self):
        first,_,a=self.run_ingest()
        second,_,b=self.run_ingest()
        self.assertEqual(a['sha256'],b['sha256'])
        self.assertNotEqual(a['id'],b['id'])

    def test_reserved_acquisition_uses_identity_and_atomic_receipt(self):
        session=Mock();session.get.return_value=self.response()
        request_id='00000000-0000-4000-8000-000000000001'
        with patch('historical_valuation_ingest.load_context',return_value=({}, {'cik':1})), patch('historical_valuation_ingest.save_original',return_value={'reused':False}) as save, patch('historical_valuation_retrieval.preflight_receipt') as preflight, patch('historical_valuation_retrieval.retain_receipt') as retain:
            result=ingest_sec(self.db,'aid','eid',user_agent='Charlie',session=session,retrieval_id=request_id)
        self.assertEqual(save.call_args.args[1]['id'],'sec-companyfacts:'+request_id)
        self.assertEqual(retain.call_args.args[3],result)
        preflight.assert_called_once()
        self.assertEqual(session.get.call_count,1)

    def test_wrong_issuer_duplicate_json_and_oversize_rejected(self):
        for raw in (b'{"cik":2,"facts":{"x":{}}}',b'{"cik":1,"cik":1,"facts":{}}',b'x'*15000001):
            with self.subTest(size=len(raw)), self.assertRaises(ValueError): self.run_ingest(self.response(raw))

    def test_redirect_and_target_drift_do_not_save(self):
        with self.assertRaises(ValueError): self.run_ingest(self.response(status=302))
        with self.assertRaisesRegex(ValueError,'target changed'):
            self.run_ingest(contexts=[({'a':1},{'cik':1}),({'a':2},{'cik':1})])

    def test_network_failure_is_sanitized(self):
        session=Mock();session.get.side_effect=requests.Timeout('private transport details')
        with patch('historical_valuation_ingest.load_context',return_value=({}, {'cik':1})):
            with self.assertRaisesRegex(ValueError,'connectivity') as exc:
                ingest_sec(self.db,'aid','eid',user_agent='Charlie',session=session)
        self.assertNotIn('private',str(exc.exception))


class PriceIngestTests(unittest.TestCase):
    def test_compact_only_exact_bytes_and_no_credential_in_receipt(self):
        import json
        from historical_valuation_ingest import ingest_prices
        from tests.unit.test_historical_price_evidence import PriceEvidenceTests
        raw=json.dumps(PriceEvidenceTests().fixture()).encode()
        session=Mock();session.get.return_value=SecIngestTests().response(raw)
        with patch('historical_valuation_ingest.load_context',return_value=({}, {'symbol':'TEST'})), patch('historical_valuation_ingest.save_original',return_value={'reused':False}) as save:
            result=ingest_prices(SecIngestTests().db,'aid','eid','2025-03-03',api_key='synthetic-key',session=session)
        self.assertEqual(session.get.call_args.kwargs['params']['outputsize'],'compact')
        self.assertEqual(save.call_args.args[1]['content'],raw)
        self.assertNotIn('synthetic-key',str(result))
        self.assertNotIn('synthetic-key',str(save.call_args.args[1]))
        self.assertEqual(result['provenance']['observedUrl'],'https://www.alphavantage.co/query')
        self.assertFalse(result['provenanceVerified'])

    def test_provider_error_no_save_or_escalation(self):
        from historical_valuation_ingest import ingest_prices
        session=Mock();session.get.return_value=SecIngestTests().response(b'{"Information":"limit"}')
        with patch('historical_valuation_ingest.load_context',return_value=({}, {'symbol':'TEST'})), patch('historical_valuation_ingest.save_original') as save:
            with self.assertRaises(ValueError):
                ingest_prices(SecIngestTests().db,'aid','eid','2025-03-03',api_key='synthetic-key',session=session)
            save.assert_not_called();self.assertEqual(session.get.call_count,1)

    def test_missing_credential_does_not_contact_provider(self):
        from historical_valuation_ingest import ingest_prices
        session=Mock()
        with self.assertRaises(ValueError):
            ingest_prices(SecIngestTests().db,'aid','eid','2025-03-03',api_key=None,session=session)
        session.get.assert_not_called()
