import json
import unittest
import test_research_commands as command_tests
from research_task_sources import collect, verify_public_sources
from trusted_research_sources import endpoint, register


class TrustedSourceTests(unittest.TestCase):
    setUp = command_tests.PublicSourceTests.setUp
    fetch = command_tests.PublicSourceTests.fetch

    def prepare(self):
        collect(self.m, self.claim['id'], self.claim['owner'], self.fetch)
        return dict(url='https://clinicaltrials.gov/study/NCT01234567', sourceDate='2026-09-07',
                    title='Verified study record', relevance='Verified sponsor relation to this company',
                    dateEvidence='Last update posted September 7, 2026')

    def body(self, day='2026-09-07'):
        return json.dumps({'protocolSection': {'identificationModule': {'nctId': 'NCT01234567'},
            'statusModule': {'lastUpdatePostDateStruct': {'date': day}}}}).encode()

    def test_canonical_endpoints_and_rejected_urls(self):
        self.assertEqual(endpoint('https://clinicaltrials.gov/study/NCT01234567')[1], 'registry')
        for url in ('http://www.fda.gov/drugs/test', 'https://www.fda.gov.evil.com/drugs/test',
                    'https://user@www.fda.gov/drugs/test', 'https://www.fda.gov:8443/drugs/test',
                    'https://www.fda.gov/drugs/../test', 'https://www.fda.gov/drugs/%2e/test',
                    'https://www.fda.gov/drugs/test?url=http://localhost', 'https://localhost/drugs/test',
                    'https://clinicaltrials.gov/study/NCT123', 'https://www.fda.gov/drugs/test#x'):
            with self.subTest(url=url), self.assertRaises(ValueError):endpoint(url)

    def test_original_receipt_idempotence_and_production_manifest_gate(self):
        record=self.prepare()
        result=register(self.m,self.claim['id'],self.claim['owner'],record,lambda url:self.body())
        self.assertEqual((self.root/result['filename']).read_bytes(),self.body())
        self.assertEqual(register(self.m,self.claim['id'],self.claim['owner'],record,lambda url:self.body()),result)
        row=self.m.db.execute('SELECT * FROM refresh_requests').fetchone()
        files=[dict(filename=p.name,folder='Catalysts/'+self.result['topic']) for p in self.root.iterdir() if not p.name.startswith('.')]
        self.assertEqual(verify_public_sources(self.m,row,lambda t:{'files':files}),3)
        with self.assertRaises(ValueError):verify_public_sources(self.m,row,lambda t:{'files':files[:-1]})

    def test_date_identity_and_relevance_rejected_before_write(self):
        record=self.prepare()
        for override in ({'sourceDate':'2026-09-06'},{'relevance':''}):
            with self.assertRaises(ValueError):register(self.m,self.claim['id'],self.claim['owner'],{**record,**override},lambda u:self.body())
        for body in (self.body('2026-09-06'),self.body().replace(b'NCT01234567',b'NCT99999999')):
            with self.assertRaises(ValueError):register(self.m,self.claim['id'],self.claim['owner'],record,lambda u:body)
        self.assertFalse(list(self.root.glob('REGISTRY*')))

    def test_cancel_during_fetch_prevents_write(self):
        record=self.prepare()
        def cancel(url):
            self.m.db.execute("UPDATE refresh_requests SET status='cancelled'");self.m.db.commit()
            return self.body()
        with self.assertRaises(ValueError):register(self.m,self.claim['id'],self.claim['owner'],record,cancel)
        self.assertFalse(list(self.root.glob('REGISTRY*')))

    def test_changed_original_retained(self):
        record=self.prepare()
        result=register(self.m,self.claim['id'],self.claim['owner'],record,lambda u:self.body())
        with self.assertRaises(ValueError):register(self.m,self.claim['id'],self.claim['owner'],record,lambda u:self.body()+b' ')
        self.assertEqual((self.root/result['filename']).read_bytes(),self.body())

    def test_fda_date_evidence_must_exist_in_original(self):
        record={**self.prepare(),'url':'https://www.fda.gov/drugs/news-events-human-drugs/test',
                'dateEvidence':'September 7, 2026'}
        with self.assertRaises(ValueError):register(self.m,self.claim['id'],self.claim['owner'],record,lambda u:b'<html>Missing date</html>')
        result=register(self.m,self.claim['id'],self.claim['owner'],record,lambda u:b'<html>September 7, 2026</html>')
        self.assertEqual(result['sourceKind'],'fda')
