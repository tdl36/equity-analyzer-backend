import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality',ROOT/'scripts/evaluate-research-quality.py')
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)

class SourcePackTests(unittest.TestCase):
    def test_all_source_pack_controls(self):
        for path in (ROOT/'evals/research-quality').glob('*.json'):
            pack=json.loads(path.read_text())
            with self.subTest(pack=path.name):
                self.assertTrue(quality.evaluate(pack['good'],pack)['passed'])
                self.assertFalse(quality.evaluate(pack['bad'],pack)['passed'])
                for case in pack.get('badCases',[]):self.assertFalse(quality.evaluate(case,pack)['passed'])
    def test_changed_real_source_invalidates_annotated_pack(self):
        pack=json.loads((ROOT/'evals/research-quality/fda-rinvoq-crohns.json').read_text())
        altered=copy.deepcopy(pack);altered['sourceParts'][0]['content']+=' altered'
        result=quality.evaluate(pack['good'],altered)
        self.assertFalse(result['passed']);self.assertIn('Annotated public-source excerpt hash changed',result['extractionIssues'])

    def financial_pack(self):
        return json.loads((ROOT/'evals/research-quality/msft-fy2025-financials.json').read_text())
    def test_financial_unit_scaling_and_rounding(self):
        pack=self.financial_pack();candidate=copy.deepcopy(pack['good'])
        candidate['financialAssertions'][0].update(value='281724',unit='USD_million')
        candidate['financialAssertions'][1]['value']='14.93'
        self.assertTrue(quality.evaluate(candidate,pack)['passed'])
    def test_missing_duplicate_nonfinite_and_unknown_assertions_fail(self):
        pack=self.financial_pack()
        variants=[]
        missing=copy.deepcopy(pack['good']);missing.pop('financialAssertions');variants.append(missing)
        duplicate=copy.deepcopy(pack['good']);duplicate['financialAssertions'].append(duplicate['financialAssertions'][0]);variants.append(duplicate)
        unknown=copy.deepcopy(pack['good']);unknown['financialAssertions'].append({'id':'unsupported'});variants.append(unknown)
        for value in (True,'Infinity','NaN','1e100','1e999999999'):
            invalid=copy.deepcopy(pack['good']);invalid['financialAssertions'][0]['value']=value;variants.append(invalid)
        for candidate in variants:
            self.assertFalse(quality.evaluate(candidate,pack)['passed'])
    def test_frozen_multisource_pack_rejects_content_or_identity_drift(self):
        pack=self.financial_pack()
        for field in ('content','name'):
            changed=copy.deepcopy(pack);changed['sourceParts'][0][field]+='changed'
            result=quality.evaluate(pack['good'],changed)
            self.assertFalse(result['passed'])
            self.assertIn('Frozen source pack content or identity changed',result['extractionIssues'])
