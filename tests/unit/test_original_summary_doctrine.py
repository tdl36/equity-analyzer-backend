"""The original Summary's epistemic doctrine, ported from the Improved pipeline.

The original prompt already carried the stronger fidelity apparatus. What it
lacked was restraint in the assessment step, which asked for an "UNFILTERED"
take, told the model not to hedge, and explicitly requested a credibility
rating — producing an invented "8.5/10" and a psychology read drawn from a
deflecting joke.
"""
import ast
import re
import unittest
from pathlib import Path

SOURCE = Path('app_v3.py').read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    """Evaluate a module-level constant, plus any earlier constant it builds on."""
    namespace = {}
    for node in TREE.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [getattr(t, 'id', '') for t in node.targets]
        if not any(t.isupper() for t in targets if t):
            continue
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'app_v3.py', 'exec'), namespace)
        except Exception:
            continue
        if name in targets:
            return namespace[name]
    raise AssertionError(f'{name} is not a module-level constant')


class DoctrineTests(unittest.TestCase):
    def setUp(self):
        self.doctrine = constant('RESEARCH_DOCTRINE')

    def test_it_outranks_the_instruction_to_be_unfiltered(self):
        self.assertIn('override any instruction above', self.doctrine)

    def test_numerical_credibility_scores_are_forbidden(self):
        self.assertIn('never rate anything out of ten', self.doctrine)
        self.assertIn('Never assign a numerical credibility', self.doctrine)

    def test_inferred_psychology_is_forbidden(self):
        self.assertIn('Never infer psychology, motive', self.doctrine)
        # The specific failure: a joke read as evidence of concern.
        self.assertIn('A joke, a deflection, a hedge or a disfluency is not evidence of feeling',
                      self.doctrine)

    def test_no_baseline_means_no_novelty_or_consensus_claims(self):
        self.assertIn('no baseline', self.doctrine)
        self.assertIn('consensus difference', self.doctrine)

    def test_judgment_must_name_its_support_and_its_limit(self):
        self.assertIn('name the statement it rests on, and state its limit', self.doctrine)

    def test_an_unanswered_question_is_not_treated_as_evasion(self):
        self.assertIn('Missing quantification is not evasion', self.doctrine)


class AssessmentPromptTests(unittest.TestCase):
    def setUp(self):
        self.instruction = constant('ASSESSMENT_INSTRUCTION')

    def test_the_assessment_carries_the_doctrine(self):
        self.assertIn(constant('RESEARCH_DOCTRINE'), self.instruction)

    def test_it_no_longer_asks_for_a_credibility_rating(self):
        self.assertNotIn('Rate overall credibility', self.instruction)
        self.assertNotIn('Rate the overall credibility', self.instruction)
        self.assertIn('Evidence quality', self.instruction)

    def test_it_no_longer_asks_whether_anyone_seemed_disingenuous(self):
        # That invited mind-reading; evasion is now evidenced by wording.
        self.assertNotIn('disingenuous', self.instruction)
        self.assertIn('Quote the wording that shows it', self.instruction)

    def test_it_stays_candid_and_specific(self):
        self.assertIn('candid assessment', self.instruction)
        self.assertIn('name the weak answers and the strong ones', self.instruction)

    def test_it_does_not_tell_the_model_to_stop_hedging(self):
        self.assertNotIn("Don't hedge", self.instruction)
        self.assertIn('do not manufacture confidence the source does not support', self.instruction)


class PromptCoverageTests(unittest.TestCase):
    """Five assessment prompts had drifted into four near-identical copies."""

    def test_every_assessment_prompt_uses_the_shared_instruction(self):
        assignments = re.findall(r'assessment_instruction = (.+)', SOURCE)
        self.assertGreaterEqual(len(assignments), 5)
        for value in assignments:
            self.assertTrue(value.startswith('ASSESSMENT_INSTRUCTION'), value[:80])

    def test_no_prompt_still_requests_a_credibility_rating(self):
        for phrase in ('Rate overall credibility', 'Rate the overall credibility',
                       'UNFILTERED assessment'):
            self.assertNotIn(phrase, SOURCE, phrase)

    def test_both_summary_prompts_carry_the_doctrine(self):
        # The audio and document paths have separate prompts; both must have it.
        self.assertEqual(SOURCE.count('{RESEARCH_DOCTRINE}'), 2)

    def test_the_summary_prompts_keep_their_fidelity_apparatus(self):
        for rule in ('NO QUANTITATIVE TIGHTENING', 'HALLUCINATION GUARD RAIL',
                     'PRESERVE CLARIFYING FOLLOW-UPS', 'Transcript Corrections Log',
                     'STEP 0 — AUTO-CLASSIFICATION'):
            self.assertIn(rule, SOURCE, rule)
