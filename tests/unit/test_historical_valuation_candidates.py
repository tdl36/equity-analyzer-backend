"""Synthetic retained originals only; no database, provider, model or approvals."""
import copy
import unittest
from historical_valuation_candidates import assemble_candidates, replay_candidates
from tests.unit import test_historical_valuation_review as fixtures


class CandidateTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.PendingHistoricalTests();f.setUp()
        self.baseline=f.baseline
        row=f.data['observations'][0]
        self.selections=[dict(date=row['price']['date'],priceSource=row['price']['source'],earningsSource=row['earnings']['source'])]
        self.records=[dict(assignmentId=f.baseline['assignmentId'],id=receipt['id'],sha256=receipt['sha256'],
                           kind=kind,retrievedDate='2025-03-04',content=f.originals[receipt['id']])
                      for receipt,kind in [(row['price']['source'],'daily_prices'),(row['earnings']['source'],'sec_companyfacts')]]

    def assemble(self):
        return assemble_candidates(self.selections,self.records,self.baseline,cik=1,symbol='TEST')

    def test_exact_extraction_does_not_invent_mapping_or_valuation(self):
        result=self.assemble();row=result['payload']['observations'][0]
        self.assertEqual(row['status'],'inputs_extracted')
        self.assertEqual(row['price']['value'],'100.0000')
        self.assertEqual(row['earnings']['basis'],'GAAP diluted EPS')
        self.assertFalse(result['payload']['valuationReady'])
        for field in ('securityId','shareBasis','currency','multiple'):
            self.assertNotIn(field,row['price'])
        self.assertNotIn('multiple',row)
        self.assertIn('corporate_actions',result['payload']['unresolvedChecks'])
        self.assertEqual(result,self.assemble())

    def test_missing_evidence_preserves_partial_progress(self):
        self.records=self.records[:1]
        row=self.assemble()['payload']['observations'][0]
        self.assertEqual(row['status'],'incomplete')
        self.assertIn('price',row);self.assertNotIn('earnings',row)
        self.assertEqual(row['gaps'],[{'input':'earnings','code':'missing_original'}])
        self.records=[]
        self.assertEqual(len(self.assemble()['payload']['observations'][0]['gaps']),2)

    def test_wrong_assignment_and_tampered_bytes_are_rejected(self):
        for field,value in [('assignmentId','other'),('content',b'changed')]:
            before=copy.deepcopy(self.records)
            self.records[0][field]=value
            with self.assertRaises(ValueError):self.assemble()
            self.records=before

    def test_date_selection_is_exact_and_duplicate_dates_rejected(self):
        self.selections[0]['date']='2025-03-02'
        row=self.assemble()['payload']['observations'][0]
        self.assertNotIn('price',row)
        self.assertEqual(row['gaps'][0]['code'],'unverified_original')
        self.selections*=2
        with self.assertRaises(ValueError):self.assemble()

    def test_wrong_source_kind_does_not_extract(self):
        self.records[0]['kind']='sec_companyfacts'
        row=self.assemble()['payload']['observations'][0]
        self.assertEqual(row['gaps'][0]['code'],'wrong_original_kind')
        self.assertNotIn('price',row)

    def test_no_caller_financial_overrides_or_extra_receipt_metadata(self):
        self.selections[0]['price']='999'
        with self.assertRaises(ValueError):self.assemble()
        del self.selections[0]['price']
        self.selections[0]['priceSource']['currency']='USD'
        with self.assertRaises(ValueError):self.assemble()

    def test_digest_binds_baseline_and_receipt_vintage_and_is_detached(self):
        first=self.assemble();before=copy.deepcopy(first)
        self.records[0]['retrievedDate']='2025-03-05'
        self.assertNotEqual(first['candidateDigest'],self.assemble()['candidateDigest'])
        self.baseline['fundamentalsDigest']='e'*64
        self.assertNotEqual(first['candidateDigest'],self.assemble()['candidateDigest'])
        self.selections[0]['date']='2025-03-02'
        self.assertEqual(first,before)

    def test_bounded_selection_and_invalid_identity(self):
        for selections in ([],self.selections*31,None):
            with self.assertRaises(ValueError):assemble_candidates(selections,self.records,self.baseline,cik=1,symbol='TEST')
        for cik,symbol in [(True,'TEST'),(1,'invalid symbol')]:
            with self.assertRaises(ValueError):assemble_candidates(self.selections,self.records,self.baseline,cik=cik,symbol=symbol)

    def test_replay_rechecks_snapshot_and_new_evidence(self):
        snapshot=self.assemble()
        self.assertEqual(replay_candidates(snapshot,self.records,self.baseline,cik=1,symbol='TEST'),snapshot)
        changed=copy.deepcopy(snapshot);changed['payload']['valuationReady']=True
        with self.assertRaises(ValueError):replay_candidates(changed,self.records,self.baseline,cik=1,symbol='TEST')
        original_records=self.records;self.records=[]
        missing=self.assemble()
        with self.assertRaises(ValueError):replay_candidates(missing,original_records,self.baseline,cik=1,symbol='TEST')

    def test_replay_rejects_context_changes_without_updating_baseline(self):
        snapshot=self.assemble()
        for cik,symbol in [(2,'TEST'),(1,'OTHER')]:
            with self.assertRaises(ValueError):replay_candidates(snapshot,self.records,self.baseline,cik=cik,symbol=symbol)

    def test_acquisition_changes_invalidate_replay_without_approving_provenance(self):
        missing=self.assemble()
        self.assertTrue(all(v['status']=='missing' for v in missing['payload']['acquisitionChecks'].values()))
        self.records[0]['provenance']=dict(provider='alphavantage',observedUrl='https://www.alphavantage.co/query',
            retrievedAt='2025-03-04T12:00:00Z',collectorVersion='historical-prices-v1')
        current=self.assemble()
        check=current['payload']['acquisitionChecks'][self.records[0]['id']]
        self.assertEqual(check['status'],'consistent_unverified')
        self.assertFalse(check['verified'])
        self.assertFalse(current['payload']['valuationReady'])
        self.assertIn('source_provenance',current['payload']['unresolvedChecks'])
        with self.assertRaisesRegex(ValueError,'new candidate version'):
            replay_candidates(missing,self.records,self.baseline,cik=1,symbol='TEST')
        self.records[0]['provenance']['retrievedAt']='2025-03-04T13:00:00Z'
        with self.assertRaisesRegex(ValueError,'new candidate version'):
            replay_candidates(current,self.records,self.baseline,cik=1,symbol='TEST')

    def test_wrong_sec_issuer_endpoint_and_acquisition_date_remain_explicit_gaps(self):
        self.records[1]['provenance']=dict(provider='sec',observedUrl='https://data.sec.gov/api/xbrl/companyfacts/CIK0000000002.json',
            retrievedAt='2025-03-05T00:00:00Z',collectorVersion='historical-sec-v1')
        check=self.assemble()['payload']['acquisitionChecks'][self.records[1]['id']]
        self.assertEqual(check['status'],'mismatch')
        self.assertEqual(check['gaps'],['endpoint_or_issuer_mismatch','retrieval_date_mismatch'])
        self.assertFalse(check['verified'])

    def test_currency_and_share_basis_are_not_inferred_from_symbol_or_eps(self):
        row=self.assemble()['payload']['observations'][0]
        compatibility=row['compatibility']
        self.assertIsNone(compatibility['priceCurrency'])
        self.assertEqual(compatibility['earningsCurrency'],'USD')
        self.assertEqual(compatibility['priceBasis'],'unadjusted_close')
        self.assertEqual(compatibility['earningsBasis'],'GAAP diluted EPS')
        self.assertFalse(compatibility['valuationReady'])
        self.assertEqual(len(compatibility['gaps']),3)
        self.records=[]
        empty=self.assemble()['payload']['observations'][0]['compatibility']
        self.assertIsNone(empty['earningsCurrency'])
        self.assertIsNone(empty['priceBasis'])
        self.assertIsNone(empty['earningsBasis'])
        self.assertEqual(empty['gaps'],compatibility['gaps'])
