import unittest
from historical_valuation_provenance import provenance

class ProvenanceTests(unittest.TestCase):
    def test_missing_stays_missing_and_valid_metadata_is_copied(self):
        self.assertIsNone(provenance(None))
        value=dict(provider='fixture', observedUrl='https://example.test/prices',
                   retrievedAt='2026-10-12T12:00:00Z', collectorVersion='v1')
        self.assertEqual(provenance(value), value)
        self.assertIsNot(provenance(value), value)

    def test_credentials_queries_invalid_time_and_certification_rejected(self):
        value=dict(provider='fixture', observedUrl='https://example.test/prices',
                   retrievedAt='2026-10-12T12:00:00Z', collectorVersion='v1')
        for url in ['http://example.test', 'https://user:password@example.test',
                    'https://example.test/?key=secret', 'https://example.test/#secret',
                    'https://example.test:80', 'https://example.test/\npath']:
            with self.assertRaises(ValueError): provenance({**value,'observedUrl':url})
        for stamp in ['2026-10-12', '2026-10-12T12:00:00', 'bad']:
            with self.assertRaises(ValueError): provenance({**value,'retrievedAt':stamp})
        with self.assertRaises(ValueError): provenance({**value, 'verified':True})
