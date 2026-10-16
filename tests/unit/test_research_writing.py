"""Offline generation/restart contract; no provider or database access."""
import copy
import unittest
from stock_analysis import generate, GROUPS
from research_writing import POLICY
from tests.unit.test_stock_analysis import SOURCE, result
from research_package_rewrite import specification, preview, prompt

class WritingTests(unittest.TestCase):
    def test_synthesis_runs_last_uses_reviewed_context_and_keeps_stage_ids(self):
        calls=[]; saved=[]
        def ask(text,tokens,key):
            calls.append((key,text,tokens))
            if key.startswith('report'):
                return result(GROUPS[int(key[-1])])
            return {'findings':[{'claimId':'/summary/one_liner','status':'supported','reason':'Synthetic evidence'}]}
        final=generate({},[SOURCE],{'revision':1},ask,lambda s:saved.append(copy.deepcopy(s)),lambda:None,
                       {'mode':'deep','writingPolicy':POLICY})
        self.assertEqual([k for k,_,_ in calls if k.startswith('report')],['report-1','report-2','report-3','report-4','report-5','report-0'])
        summary=next(p for k,p,_ in calls if k=='report-0')
        self.assertIn('"financials":', summary)
        self.assertIn('"valuation":', summary)
        self.assertIn('multiple premises',summary)
        self.assertNotIn('180 characters',summary)
        self.assertNotIn('one narrow proposition',summary)
        self.assertEqual(len(final['completed']),12)
        resumed=generate(final,[SOURCE],{'revision':1},lambda *a:self.fail('Repeated paid call'),lambda s:None,lambda:None,
                         {'mode':'deep','writingPolicy':POLICY})
        self.assertEqual(final,resumed)

    def test_interruption_does_not_restart_or_repurchase_completed_sections(self):
        saved=[]
        def ask(p,t,k):
            if k=='report-2': raise TimeoutError()
            return result(GROUPS[1])
        with self.assertRaises(TimeoutError):
            generate({},[SOURCE],{'revision':1},ask,lambda s:saved.append(copy.deepcopy(s)),lambda:None,{'mode':'deep','writingPolicy':POLICY})
        self.assertEqual(saved[-1]['completed'],['report-1','review-1'])
        self.assertEqual(saved[-1]['inFlight'],'report-2')
        with self.assertRaises(ValueError):
            generate(saved[-1],[SOURCE],{},lambda *a:self.fail('Ambiguous retry'),lambda s:None,lambda:None,{'mode':'deep','writingPolicy':POLICY})

    def test_rewrite_policy_is_frozen_and_longer_brief_is_valid(self):
        data={'instructions':'Improve argument','fields':[{'id':'brief','label':'Brief','text':'x'*600,'limit':1600}]}
        legacy=specification(data)
        current=specification({**data,'writingPolicy':POLICY})
        self.assertNotIn('writingPolicy',legacy)
        self.assertIn('economic mechanism',prompt(current,[SOURCE],{}))
        self.assertNotEqual(legacy,current)
        with self.assertRaises(ValueError): specification({**data,'writingPolicy':'unknown'})
