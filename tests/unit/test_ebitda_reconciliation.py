"""Synthetic arithmetic and saved-case regressions; no live database/providers."""
import copy
import unittest
import uuid
import test_financial_observations as obs
from test_operating_model import fixture
from operating_model import evaluate


def model():
    return {**fixture(), 'baseEbitda':'200', 'baseEbitdaReference':'Synthetic FY25',
            'baseEbitdaComparable':True, 'ebitdaReconciliation':{
                'startingEbitda':'180', 'startingBasis':'Reported EBITDA',
                'sourceReference':'Synthetic FY25 reconciliation p.2', 'confirmed':True,
                'adjustments':[
                    {'label':'Restructuring','amount':'30','recurrence':'uncertain','reference':'Synthetic p.2; exclude charge'},
                    {'label':'Disposal gain','amount':'-10','recurrence':'nonrecurring','reference':'Synthetic p.2; exclude gain'}]}}


class ReconciliationTests(unittest.TestCase):
    def test_signed_bridge_recomputes_and_does_not_change_valuation(self):
        data=model();data['ebitdaReconciliation']['reconciledEbitda']='999'
        result=evaluate(data)
        self.assertEqual(result['ebitdaReconciliation']['totalAdjustments'],'20')
        self.assertEqual(result['ebitdaReconciliation']['reconciledEbitda'],'200')
        self.assertEqual(result['results']['base']['impliedPrice'],evaluate(fixture())['results']['base']['impliedPrice'])
        self.assertEqual(evaluate(result),result)
        self.assertTrue(any('uncertain adjustments' in w for w in result['warnings']))

    def test_invalid_or_incomplete_bridges_rejected(self):
        for patch in ({'confirmed':False},{'startingEbitda':'NaN'},{'startingEbitda':'180.000001'},
                      {'startingBasis':''},{'sourceReference':''},{'adjustments':[]},{'adjustments':[{}]*21}):
            data=model();data['ebitdaReconciliation'].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):evaluate(data)
        for patch in ({'amount':0},{'amount':True},{'amount':'Infinity'},{'recurrence':'guaranteed'},{'reference':''},{'label':'Disposal gain'}):
            data=model();data['ebitdaReconciliation']['adjustments'][0].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):evaluate(data)
        data=model();del data['baseEbitda']
        with self.assertRaises(ValueError):evaluate(data)

    def test_exact_decimal_math_and_signed_start(self):
        data=model();bridge=data['ebitdaReconciliation'];bridge['startingEbitda']='-0.1'
        bridge['adjustments'][0]['amount']='200.3';bridge['adjustments'][1]['amount']='-0.2'
        self.assertEqual(evaluate(data)['ebitdaReconciliation']['reconciledEbitda'],'200.0')
        data['baseEbitda']='199.999999'
        with self.assertRaises(ValueError):evaluate(data)


class SavedReconciliationTests(unittest.TestCase):
    setUp=obs.ObservationTests.setUp
    def test_preview_save_replay_and_reject_mismatch(self):
        data=model()
        self.assertEqual(self.client.post('/api/research/operating-model/preview',json=data).status_code,200)
        payload=dict(requestId=str(uuid.uuid4()),revision=0,body={'operatingModel':data})
        saved=self.client.post('/api/research/investment-case/SYNTH',json=payload)
        self.assertEqual(saved.status_code,200)
        self.assertEqual(saved.json['body']['operatingModel']['ebitdaReconciliation']['reconciledEbitda'],'200')
        self.assertTrue(self.client.post('/api/research/investment-case/SYNTH',json=payload).json['replayed'])
        bad=copy.deepcopy(payload);bad['requestId']=str(uuid.uuid4());bad['revision']=1
        bad['body']['operatingModel']['ebitdaReconciliation']['startingEbitda']='1'
        self.assertEqual(self.client.post('/api/research/investment-case/SYNTH',json=bad).status_code,400)
        self.assertEqual(len(self.rows),1)
