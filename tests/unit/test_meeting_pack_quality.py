import unittest
from meeting_pack_quality import inspect
class QualityTests(unittest.TestCase):
    def test_does_not_confuse_filenames_with_verified_evidence(self):
        q={'question':'What changed?','source_filenames':['a.pdf'],'source':'a.pdf','context':'Margin','follow_up_angle':'Why?','priority':'high'}
        report=inspect([{'questions':[q,q]}],['a.pdf','b.pdf'])
        self.assertEqual(report['withVerifiedPassages'],0);self.assertEqual(report['duplicateQuestions'],1);self.assertEqual(report['uncitedSources'],['b.pdf'])
        self.assertIn('Mechanical checks only',report['scope'])
    def test_numeric_warning_is_not_a_factual_verdict(self):
        report=inspect([{'questions':[{'question':'Why was growth 80%?','supporting_quotes':[{'quote':'The company expects growth of 20%.'}]}]}],[])
        self.assertEqual(report['numericPremisesToReview'],[{'question':1,'values':['80%']}])
