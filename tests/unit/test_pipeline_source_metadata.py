import unittest
from pipeline_source_metadata import fields, observed_url

class Metadata(unittest.TestCase):
    def test_allowlist_keeps_recorded_receipt_without_private_fields(self):
        data=fields(dict(source='alphasense_collection',sha256='a'*64,collectionRun='b'*12,sourceUrl='https://research.alpha-sense.com/doc/123',privateBody='secret',apiKey='secret'))
        self.assertEqual(data['recordedSha256'],'a'*64)
        self.assertEqual(data['collectionRun'],'b'*12)
        self.assertTrue(data['originalReceiptRecorded'])
        self.assertEqual(data['sourceUrl'],'https://research.alpha-sense.com/doc/123')
        self.assertNotIn('secret',str(data))
    def test_unproven_receipt_does_not_expose_link_or_hash(self):
        for source,sha in [('other','a'*64),('alphasense_collection','wrong')]:
            data=fields(dict(source=source,sha256=sha,sourceUrl='https://research.alpha-sense.com/doc/123',collectionRun='b'*12))
            self.assertFalse(data['originalReceiptRecorded'])
            self.assertEqual(data['sourceUrl'],'');self.assertEqual(data['recordedSha256'],'')
    def test_rejects_unsafe_or_secret_urls(self):
        for url in ['javascript:alert(1)','https://example.com/','https://research.alpha-sense.com@evil.com/doc','https://user:pass@research.alpha-sense.com/doc','https://research.alpha-sense.com:bad/doc','https://research.alpha-sense.com/login','https://research.alpha-sense.com/doc?access_token=x','https://research.alpha-sense.com/doc?api_key=x','https://research.alpha-sense.com/doc#token=x','https://research.alpha-sense.com/doc?%73ecret=x','https://research.alpha-sense.com/\nsecret']:
            with self.subTest(url=url):self.assertEqual(observed_url(url),'')
        self.assertEqual(observed_url('https://research.alpha-sense.com/doc?id=123'),'https://research.alpha-sense.com/doc?id=123')

    def test_recorded_type_and_evidenced_author_only(self):
        value=dict(source='alphasense_collection',sha256='a'*64,sourceKind='broker-report',author='Analyst',authorEvidence='Observed byline')
        self.assertEqual(fields(value)['sourceKind'],'broker-report')
        self.assertEqual(fields(value)['author'],'Analyst')
        self.assertEqual(fields({**value,'authorEvidence':''})['author'],'')
        self.assertEqual(fields({**value,'sourceKind':'invented'})['sourceKind'],'')
