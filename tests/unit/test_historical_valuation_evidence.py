import copy
import hashlib
import json
import unittest
from historical_valuation_evidence import annual_eps_before


def fact(value=5, **kw):
    return dict(start='2024-01-01',end='2024-12-31',filed='2025-02-15',
                form='10-K',accn='0000000001-25-000001',val=value) | kw


class EarningsEvidenceTests(unittest.TestCase):
    def extract(self, facts, **kwargs):
        data={'cik':1,'facts':{'us-gaap':{'EarningsPerShareDiluted':{'units':{'USD/shares':facts}}}}}
        content=json.dumps(data).encode();receipt={'id':'sec','sha256':hashlib.sha256(content).hexdigest()}
        return annual_eps_before(content,receipt,kwargs.get('cik',1),kwargs.get('date','2025-03-03'))

    def test_later_restatement_and_same_day_filing_are_excluded(self):
        rows=[fact(),fact(7,filed='2026-02-01'),fact(9,filed='2025-03-03')];before=copy.deepcopy(rows)
        result=self.extract(rows)
        self.assertEqual(result['status'],'extracted');self.assertEqual(result['earnings']['value'],'5')
        self.assertEqual(rows,before)
        self.assertNotIn('shareBasis',result['earnings'])

    def test_latest_period_then_latest_filing(self):
        rows=[fact(1,start='2023-01-01',end='2023-12-31'),fact(),fact(6,filed='2025-02-20',accn='0000000001-25-000002')]
        self.assertEqual(self.extract(rows)['earnings']['value'],'6')

    def test_conflicts_do_not_fall_back(self):
        for change in [{'val':6},{'start':'2024-01-02'},{'accn':'0000000001-25-000002'}, {'val':'5'}, {'val':True}]:
            result=self.extract([fact(3,filed='2025-02-01'),fact(),fact()|change])
            self.assertEqual(result['status'],'unverified');self.assertNotIn('earnings',result)

    def test_identity_and_period_failures(self):
        for kwargs in [{'cik':2},{'cik':True},{'date':'2025-02-15'},{'date':'2027-03-03'}]:
            self.assertEqual(self.extract([fact()],**kwargs)['status'],'unverified')
        for change in [{'filed':'bad'},{'start':None},{'accn':'bad'}]:
            self.assertEqual(self.extract([fact()|change])['status'],'unverified')

    def test_receipts_and_duplicate_json_keys(self):
        content=b'{"cik":1,"cik":2}'
        receipt={'id':'sec','sha256':hashlib.sha256(content).hexdigest()}
        for data in [content,b'changed']:
            self.assertEqual(annual_eps_before(data,receipt,1,'2025-03-03')['status'],'unverified')

    def test_extreme_decimal_exponent_is_bounded_before_formatting(self):
        content=json.dumps({'cik':1,'facts':{'us-gaap':{'EarningsPerShareDiluted':{'units':{'USD/shares':[fact()]}}}}}).encode().replace(b'"val": 5',b'"val": 1e999999')
        receipt={'id':'sec','sha256':hashlib.sha256(content).hexdigest()}
        self.assertEqual(annual_eps_before(content,receipt,1,'2025-03-03')['status'],'unverified')


class SECBindingTests(unittest.TestCase):
    def fixture(self):
        from tests.unit.test_historical_valuation import observation
        row=observation()
        content=json.dumps({'cik':1,'facts':{'us-gaap':{'EarningsPerShareDiluted':{'units':{'USD/shares':[fact()]}}}}}).encode()
        row['earnings']['source']={'id':'sec','sha256':hashlib.sha256(content).hexdigest()}
        row['price']['source']={'id':'price','sha256':hashlib.sha256(b'price').hexdigest()}
        return {'version':1,'securityId':'TEST-common','observations':[row]}, {'sec':content,'price':b'price'}

    def test_matching_earnings_bind_without_claiming_price_or_split_review(self):
        from historical_valuation_evidence import check_sec_supplement
        data,originals=self.fixture();result=check_sec_supplement(data,originals,1)
        row=result['observations'][0]
        self.assertEqual(row['earningsExtraction'],'matched_sec_annual_fact')
        self.assertIn('remain unverified',row['message'])
        self.assertTrue(result['sourceReviewRequired'])

    def test_proposed_value_date_and_accession_cannot_override_original(self):
        from historical_valuation_evidence import check_sec_supplement
        for key,value in [('value','6'),('publishedDate','2025-02-14'),('accession','0000000001-25-000002')]:
            data,originals=self.fixture();data['observations'][0]['earnings'][key]=value
            row=check_sec_supplement(data,originals,1)['observations'][0]
            self.assertEqual(row['status'],'unverified');self.assertNotIn('multiple',row)

    def test_false_security_and_invalid_original_never_bind(self):
        from historical_valuation_evidence import check_sec_supplement
        data,originals=self.fixture()
        self.assertEqual(check_sec_supplement(data,originals,2)['observations'][0]['status'],'unverified')
        originals['sec']=b'changed'
        self.assertEqual(check_sec_supplement(data,originals,1)['observations'][0]['status'],'unverified')
