"""Synthetic industrial baseline: no real sources, database or provider calls."""
import uuid
import unittest
import test_financial_observations as obs
from test_financial_observations import chosen
from test_operating_model import fixture
from operating_model import evaluate


class EbitdaObservationTests(unittest.TestCase):
    setUp = obs.ObservationTests.setUp
    def baseline(self):
        return {**fixture(), 'baseEbitda':'200', 'baseEbitdaReference':'Synthetic FY2025 EBITDA',
                'baseEbitdaComparable':True}

    def test_historical_margin_and_growth_do_not_change_valuation(self):
        old=evaluate(fixture()); new=evaluate(self.baseline())
        self.assertEqual(new['baseMarginPct'],'20.00')
        for name in ('bear','base','bull'):
            self.assertEqual(old['results'][name]['impliedPrice'],new['results'][name]['impliedPrice'])
        self.assertEqual(new['results']['base']['marginChangePp'],'0.00')
        self.assertEqual(new['results']['base']['ebitdaGrowthPct'],'21.00')
        # Client-supplied calculations cannot override server arithmetic.
        self.assertEqual(evaluate({**self.baseline(),'baseMarginPct':'999'})['baseMarginPct'],'20.00')

    def test_parenthesized_losses_are_not_offered_as_positive_numbers(self):
        self.assertEqual(obs.numbers('EBITDA (200) versus 250 and -300'), ['250'])

    def test_requires_positive_baseline_and_comparability(self):
        for patch in ({'baseEbitda':0},{'baseEbitda':-1},{'baseEbitdaComparable':False},{'baseEbitdaReference':''}):
            with self.subTest(patch=patch),self.assertRaises(ValueError): evaluate({**self.baseline(),**patch})
        self.assertNotIn('baseEbitda',evaluate(fixture()))

    def test_ebitda_link_is_rechecked_saved_and_frozen(self):
        observation={**chosen(),'basis':'Synthetic adjusted EBITDA'}
        url='/api/research/operating-model/SYNTH/ebitda-observation'
        linked=self.client.post(url,json=observation)
        self.assertEqual(linked.status_code,200)
        self.assertEqual(linked.json['observation']['metric'],'consolidated_annual_ebitda')
        model={**self.baseline(),'baseEbitda':'1000','ebitdaBasis':observation['basis'],
               'baseEbitdaObservation':linked.json['observation']}
        for patch in ({'baseEbitda':'999'},{'baseYear':2024},{'currency':'EUR'},{'ebitdaBasis':'Different definition'},{'baseEbitda':''}):
            with self.subTest(patch=patch),self.assertRaises(ValueError): evaluate({**model,**patch})
        preview=self.client.post('/api/research/operating-model/preview',json={**model,'ticker':'SYNTH'})
        self.assertEqual(preview.status_code,200)
        payload=dict(requestId=str(uuid.uuid4()),revision=0,body={'operatingModel':model})
        saved=self.client.post('/api/research/investment-case/SYNTH',json=payload)
        self.assertEqual(saved.status_code,200)
        self.assertEqual(saved.json['body']['operatingModel']['baseEbitdaObservation']['metric'],'consolidated_annual_ebitda')
        self.doc=None
        self.assertEqual(self.client.post('/api/research/operating-model/preview',json={**model,'ticker':'SYNTH'}).status_code,400)
        self.assertTrue(self.client.post('/api/research/investment-case/SYNTH',json=payload).json['replayed'])
        self.assertEqual(self.client.post('/api/research/investment-case/SYNTH',json={**payload,'requestId':str(uuid.uuid4()),'revision':1}).status_code,409)
        self.assertEqual(len(self.rows),1)
