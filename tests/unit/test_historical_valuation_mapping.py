import copy
import hashlib
import unittest
from historical_valuation_mapping import prepare_mapping, replay_mapping


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.baseline = dict(assignmentId='00000000-0000-4000-8000-000000000001', parentEditionId='00000000-0000-4000-8000-000000000002', fundamentalsDigest='a'*64, sourcePackFingerprint='b'*64, parentEditionDigest='c'*64)
        raw = b'{"issuer":1,"ticker":"TEST","currency":"USD"}'
        self.record = dict(assignmentId=self.baseline['assignmentId'], id='reference', sha256=hashlib.sha256(raw).hexdigest(), content=raw, kind='security_reference', retrievedDate='2025-03-04')
        self.proposal = dict(source={k:self.record[k] for k in ('id','sha256')}, fields=dict(cik='issuer',symbol='ticker',priceCurrency='currency'), priceCurrency='USD',asOf='2025-03-04')

    def prepare(self):
        return prepare_mapping(self.proposal,[self.record],self.baseline,candidate_digest='d'*64,cik=1,symbol='TEST')

    def test_matching_fields_are_pending_not_verified_or_usable(self):
        s=self.prepare()
        self.assertEqual(s['payload']['status'],'pending')
        self.assertFalse(s['payload']['valuationReady'])
        self.assertIn('price_currency_semantics',s['payload']['unresolvedChecks'])
        self.assertEqual(s,replay_mapping(s,[self.record],self.baseline,candidate_digest='d'*64,cik=1,symbol='TEST'))
        self.proposal['priceCurrency']='EUR'
        self.assertEqual(s['payload']['proposal']['priceCurrency'],'USD')

    def test_wrong_scope_kind_or_hash_rejected(self):
        for field,value in [('assignmentId','other'),('kind','daily_prices'),('content',b'changed'),('sha256','e'*64)]:
            with self.subTest(field=field):
                old=self.record.copy();self.record[field]=value
                with self.assertRaises(ValueError):self.prepare()
                self.record=old

    def test_claims_require_exact_distinct_fields(self):
        for change in [dict(priceCurrency='EUR'),dict(fields=dict(cik='issuer',symbol='ticker',priceCurrency='ticker')),dict(asOf='2025-03-05')]:
            old=copy.deepcopy(self.proposal);self.proposal.update(change)
            with self.assertRaises(ValueError):self.prepare()
            self.proposal=old

    def test_ambiguous_or_malformed_original_rejected_even_with_matching_hash(self):
        for raw in [b'{"issuer":1,"issuer":1,"ticker":"TEST","currency":"USD"}',b'{"issuer":true,"ticker":"TEST","currency":"USD"}', b'{"issuer":1,"ticker":"TEST","currency":"USD","x":NaN}', b'[]',b'\xff']:
            self.record['content']=raw;self.record['sha256']=hashlib.sha256(raw).hexdigest();self.proposal['source']['sha256']=self.record['sha256']
            with self.assertRaises(ValueError):self.prepare()

    def test_replay_binds_candidate_baseline_and_retrieval_date(self):
        s=self.prepare()
        for field in ('candidate','baseline','retrieval','acquisition'):
            baseline=self.baseline.copy();record=self.record.copy();digest='d'*64
            if field=='candidate':digest='e'*64
            elif field=='baseline':baseline['parentEditionDigest']='e'*64
            elif field=='retrieval':record['retrievedDate']='2025-03-05'
            else:record['provenance']={'provider':'fixture','observedUrl':'https://example.test/security','retrievedAt':'2025-03-04T12:00:00Z','collectorVersion':'v1'}
            with self.assertRaises(ValueError):replay_mapping(s,[record],baseline,candidate_digest=digest,cik=1,symbol='TEST')

    def test_rehashed_status_forgery_still_rejected(self):
        import json
        s=self.prepare();s['payload']['valuationReady']=True
        s['mappingDigest']=hashlib.sha256(json.dumps(s['payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with self.assertRaises(ValueError):replay_mapping(s,[self.record],self.baseline,candidate_digest='d'*64,cik=1,symbol='TEST')
