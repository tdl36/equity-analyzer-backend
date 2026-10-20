"""Offline integration checks; no external services, database or paid generation."""
import hashlib
import json
from pathlib import Path
import unittest
from research_writing import guidance, POLICY
from research_package_rewrite import specification, prompt, preview, wire_spec, validate
from model_registry import MODELS


class FinancialServicesIntegrationTests(unittest.TestCase):
    def spec(self, policy):
        return specification({'writingPolicy':policy,'instructions':'Improve the argument',
                              'fields':[{'id':'summary-id','label':'Summary','text':'Old draft','limit':1600}]})

    def test_originals_match_pinned_manifest(self):
        root=Path(__file__).resolve().parents[2]/'vendor/anthropics-financial-services'
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(manifest['revision'],'574ed3624aebd0418c7e96cd101262f30210ab26')
        for name,expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),expected,name)

    def test_new_prompt_uses_method_but_legacy_does_not(self):
        self.assertEqual(POLICY,'synthesis-v3')
        new=prompt(self.spec(POLICY),[],{})
        self.assertIn('FINANCIAL SERVICES RESEARCH WORKFLOW v1',new)
        self.assertIn('old/new columns within the SAME',new)
        self.assertIn('No automatic trading or thesis acceptance',new)
        self.assertNotIn('FINANCIAL SERVICES RESEARCH WORKFLOW',prompt(self.spec('synthesis-v2'),[],{}))
        self.assertIn('INVESTMENT JUDGMENT STANDARD',guidance('synthesis-v2'))

    def test_approval_identity_and_dispatch_contract(self):
        model=next(k for k,v in MODELS.items() if 'input' in v and 'output' in v)
        v2=preview(self.spec('synthesis-v2'),[],{},model,'parent')
        spec=self.spec(POLICY)
        v3=preview(spec,[],{},model,'parent')
        self.assertNotEqual(v2['fingerprint'],v3['fingerprint'])
        self.assertEqual(v3['outputTokens'],16000)
        self.assertGreater(v3['estimatedUsd'],v2['estimatedUsd'])
        self.assertEqual(wire_spec(spec)['fields'][0]['id'],'field-1')
        self.assertEqual(validate({'fields':[{'id':'field-1','text':'New draft'}]},spec),{'summary-id':'New draft'})

    def test_stock_analysis_dispatch_uses_new_method_without_extra_calls(self):
        from stock_analysis import generate, GROUPS
        from tests.unit.test_stock_analysis import SOURCE, result
        calls=[]
        def ask(text,tokens,stage):
            calls.append((stage,text))
            if stage.startswith('report-'):
                return result(GROUPS[int(stage[-1])])
            return {'findings':[{'claimId':'/summary/one_liner','status':'supported','reason':'Matches synthetic evidence.'}]}
        generate({},[SOURCE],{'revision':1},ask,lambda s:None,lambda:None,
                 {'mode':'deep','writingPolicy':POLICY})
        self.assertEqual(len(calls),7)
        self.assertEqual(calls[0][0],'report-1')
        self.assertEqual(calls[-2][0],'report-0')
        for stage,text in calls:
            if stage.startswith('report-'):
                self.assertIn('FINANCIAL SERVICES RESEARCH WORKFLOW v1',text)
            else:
                self.assertIn('FINANCIAL SERVICES INDEPENDENT REVIEW v1',text)
