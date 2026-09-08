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
    def test_changed_real_source_invalidates_annotated_pack(self):
        pack=json.loads((ROOT/'evals/research-quality/fda-rinvoq-crohns.json').read_text())
        altered=copy.deepcopy(pack);altered['sourceParts'][0]['content']+=' altered'
        result=quality.evaluate(pack['good'],altered)
        self.assertFalse(result['passed']);self.assertIn('Annotated public-source excerpt hash changed',result['extractionIssues'])
