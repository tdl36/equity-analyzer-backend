import copy
import unittest
from operating_model import evaluate, VERSION
from investment_case import validate, case_context_hash, create_blueprint


def fixture():
    scenario=dict(growthPct='10',marginPct='20',multiple='10',netDebt='200',otherClaims='50',nonOperatingAssets='25',shares='100',rationale='Synthetic assumptions only')
    return dict(version=VERSION,method='ev_ebitda',units='millions',suitable=True,currency='USD',baseYear=2025,targetYear=2027,
        asOf='2026-10-01',baseRevenue='1000',referencePrice='20',revenueReference='Synthetic FY25 revenue',priceReference='Synthetic price',ebitdaBasis='Synthetic adjusted EBITDA',
        scenarios={n:copy.deepcopy(scenario) for n in ('bear','base','bull')})


class OperatingModelTests(unittest.TestCase):
    def test_decimal_operating_bridge_and_reverse_reconcile(self):
        model=evaluate(fixture());r=model['results']['base']
        self.assertEqual(r['revenue'],'1210.00');self.assertEqual(r['ebitda'],'242.00')
        self.assertEqual(r['enterpriseValue'],'2420.00');self.assertEqual(r['equityValue'],'2195.00')
        self.assertEqual(r['impliedPrice'],'21.95');self.assertEqual(r['priceReturnPct'],'9.75')
        self.assertEqual(r['impliedEbitdaAtReferencePrice'],'222.50')
        self.assertEqual(len(model['sensitivity']),9)
        centre=next(s for s in model['sensitivity'] if s['ebitdaChangePct']=='0' and s['multiple']=='10')
        self.assertEqual(centre['impliedPrice'],r['impliedPrice'])

    def test_saves_recompute_and_restore_reproducible_inputs(self):
        data=fixture();data['results']={'base':{'impliedPrice':'invented'}}
        body=validate({'operatingModel':data});model=body['operatingModel']
        self.assertEqual(model['results']['base']['impliedPrice'],'21.95')
        self.assertEqual(evaluate(model),model)
        old=copy.deepcopy(body);body['operatingModel']['scenarios']['base']['growthPct']='11'
        self.assertNotEqual(case_context_hash(body),case_context_hash(old))
        self.assertNotIn('operatingModel',validate({}))
        self.assertNotIn('operatingModel',validate({'operatingModel':None}))

    def test_rejects_mixed_units_periods_unsupported_methods_and_nonfinite_values(self):
        for change in ({'units':'billions'},{'version':'future'},{'method':'bank'},{'suitable':False},{'baseRevenue':'NaN'},
                       {'referencePrice':True},{'targetYear':2025},{'targetYear':2031},{'baseYear':'2025.5'},
                       {'currency':'US dollars'},{'asOf':'yesterday'},{'revenueReference':''}):
            with self.subTest(change=change),self.assertRaises(ValueError):evaluate({**fixture(),**change})
        for field,value in [('currency','EUR'),('units','billions'),('targetYear',2028),('growthPct','-100'),('marginPct','0'),('multiple','Infinity'),('shares','0'),('otherClaims','-1'),('nonOperatingAssets','-1'),('shares','0.1234567')]:
            data=fixture();data['scenarios']['base'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):evaluate(data)

    def test_negative_equity_is_explicitly_floored_and_cash_is_added(self):
        data=fixture();data['scenarios']['bear']['netDebt']='10000';data['scenarios']['bull']['netDebt']='-200'
        model=evaluate(data)
        self.assertTrue(model['results']['bear']['equityFloored'])
        self.assertEqual(model['results']['bear']['impliedPrice'],'0.00')
        self.assertEqual(model['results']['bear']['priceReturnPct'],'-100.00')
        self.assertEqual(model['results']['bull']['equityValue'],'2595.00')
        self.assertTrue(any('floored' in w for w in model['warnings']))

    def test_preview_is_pure_and_validates_without_database_or_paid_calls(self):
        from flask import Flask
        def db(*a,**k):raise AssertionError('Preview must not access the database')
        app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client()
        response=client.post('/api/research/operating-model/preview',json=fixture())
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['model']['results']['base']['impliedPrice'],'21.95')
        self.assertEqual(client.post('/api/research/operating-model/preview',json={}).status_code,400)
        self.assertEqual(response.headers['Cache-Control'],'no-store')
