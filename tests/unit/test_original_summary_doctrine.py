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

    def test_the_model_must_still_commit_to_a_view(self):
        # "Don't hedge" existed to stop mushy notes, not to invite invention.
        self.assertIn('If an answer was weak, say it was weak', self.doctrine)
        self.assertIn('Your own vagueness is not caution', self.doctrine)
        self.assertIn('may not do is invent a fact', self.doctrine)

    def test_calling_out_evasion_is_required_not_forbidden(self):
        for behaviour in ('Non-answers', 'evasions', 'redirections',
                          'rehearsed talking points', 'internal contradictions'):
            self.assertIn(behaviour, self.doctrine, behaviour)
        self.assertIn('quote the wording that shows it', self.doctrine)

    def test_only_claims_about_interior_state_are_forbidden(self):
        self.assertIn("anyone's feelings, anxiety, morale, private belief or motive", self.doctrine)
        # The observed failure: a deflecting joke read as evidence of concern.
        self.assertIn('a joke, a hedge or a disfluency is not evidence of an interior state',
                      self.doctrine)

    def test_credibility_is_rated_on_a_defined_scale(self):
        self.assertIn('Rate the credibility of the key claims on this scale', self.doctrine)
        for anchor in ('5 —', '4 —', '3 —', '2 —', '1 —'):
            self.assertIn(anchor, self.doctrine, anchor)
        self.assertIn('name the two or three things driving the rating', self.doctrine)

    def test_an_arbitrary_finer_scale_is_refused(self):
        self.assertIn('Do not invent decimals, a ten-point range', self.doctrine)
        self.assertIn('cannot be compared across notes', self.doctrine)

    def test_a_supplied_baseline_must_be_compared_against(self):
        # thesis_addendum injects the user's registered thesis and asks for
        # per-pillar CONFIRMED / WEAKENED verdicts; the doctrine must not
        # contradict it.
        self.assertIn('Where a thesis, prior statement or estimate appears in this prompt',
                      self.doctrine)
        self.assertIn('compare against it explicitly', self.doctrine)

    def test_an_absent_baseline_may_not_be_invented(self):
        self.assertIn('Where none is supplied', self.doctrine)
        self.assertIn('inventing the comparison rather than making it', self.doctrine)


class AssessmentPromptTests(unittest.TestCase):
    def setUp(self):
        self.instruction = constant('ASSESSMENT_INSTRUCTION')

    def test_the_assessment_carries_the_doctrine(self):
        self.assertIn(constant('RESEARCH_DOCTRINE'), self.instruction)

    def test_bs_detection_survives(self):
        self.assertIn('Red flags / BS detection', self.instruction)
        self.assertIn('rehearsed non-answer', self.instruction)

    def test_it_still_asks_for_a_credibility_rating(self):
        self.assertIn('Credibility of key claims', self.instruction)
        self.assertIn('1-5 scale', self.instruction)

    def test_it_stays_direct_and_opinionated(self):
        self.assertIn("don't sugarcoat", self.instruction)
        self.assertIn('Be direct and opinionated', self.instruction)


class PromptCoverageTests(unittest.TestCase):
    """Five assessment prompts had drifted into four near-identical copies."""

    def test_every_assessment_prompt_uses_the_shared_instruction(self):
        assignments = re.findall(r'assessment_instruction = (.+)', SOURCE)
        self.assertGreaterEqual(len(assignments), 5)
        for value in assignments:
            self.assertTrue(value.startswith('ASSESSMENT_INSTRUCTION'), value[:80])

    def test_no_prompt_requests_an_unanchored_credibility_rating(self):
        # Rating is wanted; an arbitrary decimal out of ten is not.
        for phrase in ('Rate overall credibility of key claims',
                       'Rate the overall credibility of the key claims',
                       'UNFILTERED assessment'):
            self.assertNotIn(phrase, SOURCE, phrase)

    def test_the_doctrine_does_not_contradict_the_thesis_check(self):
        # thesis_addendum is appended straight after RESEARCH_DOCTRINE in the
        # audio prompt and demands per-pillar CONFIRMED / WEAKENED verdicts.
        self.assertIn('{RESEARCH_DOCTRINE}{thesis_addendum}', SOURCE)
        self.assertNotIn('No prior thesis, model or consensus estimate is supplied', SOURCE)
        self.assertIn('CONFIRMED / WEAKENED / NO MENTION', SOURCE)

    def test_both_summary_prompts_carry_the_doctrine(self):
        # The audio and document paths have separate prompts; both must have it.
        self.assertEqual(SOURCE.count('{RESEARCH_DOCTRINE}'), 2)

    def test_the_summary_prompts_keep_their_fidelity_apparatus(self):
        for rule in ('NO QUANTITATIVE TIGHTENING', 'HALLUCINATION GUARD RAIL',
                     'PRESERVE CLARIFYING FOLLOW-UPS', 'Transcript Corrections Log',
                     'STEP 0 — AUTO-CLASSIFICATION'):
            self.assertIn(rule, SOURCE, rule)
