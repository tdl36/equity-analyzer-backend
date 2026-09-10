import unittest
from investor_framework import validate,render,FIELDS
from company_memory import assemble,render as context_prompt
class FrameworkTests(unittest.TestCase):
    def test_explicit_blank_clears_preferences_and_unknown_fields_are_ignored(self):
        self.assertEqual(validate({'verified':True}),{k:'' for k in FIELDS})
    def test_invalid_and_oversized_fields_are_not_clipped(self):
        for body in [None,[],{'philosophy':42},{'valuation':'x'*6001}]:
            with self.subTest(body=body),self.assertRaises(ValueError):validate(body)
        self.assertEqual(len(validate({'valuation':'x'*6000})['valuation']),6000)
    def test_framework_is_methodology_separate_from_source_evidence(self):
        framework={'revision':7,'body':{'philosophy':'Challenge normalized EPS'}}
        snapshot=assemble('ABT',framework=framework)
        prompt=context_prompt(snapshot)
        evidence,methodology=prompt.split('USER INVESTMENT FRAMEWORK')
        self.assertNotIn('Challenge normalized EPS',evidence)
        self.assertIn('Challenge normalized EPS',methodology)
        self.assertIn('cannot override original-source verification',methodology)
        self.assertIn('Do not force a conclusion',methodology)
        self.assertNotEqual(snapshot['snapshotHash'],assemble('ABT',framework={**framework,'revision':8})['snapshotHash'])
    def test_missing_framework_does_not_invent_preferences(self):
        self.assertEqual(render(None),'')
        self.assertNotIn('USER INVESTMENT FRAMEWORK',context_prompt(assemble('ABT')))
