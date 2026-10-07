import unittest
import uuid
import test_financial_observations as obs
from test_financial_observations import chosen
from test_operating_model import fixture
from operating_model import evaluate

class BalanceTests(unittest.TestCase):
    setUp=obs.ObservationTests.setUp
    def model(self):
        return {**fixture(),'baseCash':'100','baseDebt':'300','baseCashBasis':'Cash equivalents',
                'baseDebtBasis':'Total debt including leases','baseCashReference':'Synthetic year end',
                'baseDebtReference':'Synthetic year end','balanceSheetComparable':True}
    def test_history_is_separate_from_target_debt(self):
        m=evaluate(self.model())
        self.assertEqual(m['historicalNetDebt'],'200.00')
        self.assertEqual(m['results'],evaluate(fixture())['results'])
        data=self.model();data['baseCash']='0'
        self.assertEqual(evaluate(data)['historicalNetDebt'],'300.00')
        data['baseDebt']='-1'
        with self.assertRaises(ValueError):evaluate(data)
        with self.assertRaises(ValueError):evaluate({**self.model(),'balanceSheetComparable':False})
    def test_cash_and_debt_receipts_rechecked_before_preview_and_save(self):
        model=self.model()
        for metric in ('cash','debt'):
            payload={**chosen(),'basis':model['base'+metric.title()+'Basis']}
            r=self.client.post('/api/research/operating-model/SYNTH/'+metric+'-observation',json=payload)
            self.assertEqual(r.status_code,200)
            self.assertEqual(r.json['observation']['metric'],'consolidated_year_end_'+metric)
            model['base'+metric.title()]='1000'
            model['base'+metric.title()+'Observation']=r.json['observation']
        for patch in ({'baseCash':'999'},{'baseDebtBasis':'Changed'},{'baseYear':2024},{'baseDebt':''}):
            with self.subTest(patch=patch),self.assertRaises(ValueError):evaluate({**model,**patch})
        payload=dict(requestId=str(uuid.uuid4()),revision=0,body={'operatingModel':model})
        self.assertEqual(self.client.post('/api/research/investment-case/SYNTH',json=payload).status_code,200)
        self.doc=None
        self.assertEqual(self.client.post('/api/research/operating-model/preview',json={**model,'ticker':'SYNTH'}).status_code,400)
        self.assertTrue(self.client.post('/api/research/investment-case/SYNTH',json=payload).json['replayed'])
        self.assertEqual(self.client.post('/api/research/investment-case/SYNTH',json={**payload,'requestId':str(uuid.uuid4()),'revision':1}).status_code,409)
