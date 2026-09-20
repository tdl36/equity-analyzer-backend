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

import research_doctrine

SOURCE = Path('app_v3.py').read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    """Evaluate a module-level constant, plus any earlier constant it builds on.

    Doctrine shared with Summary Lab now lives in research_doctrine, so the
    prompts here are composed from imported names. Seed those, then let a local
    definition win if app_v3 ever grows one again.
    """
    namespace = {n: getattr(research_doctrine, n) for n in dir(research_doctrine) if n.isupper()}
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
    if name in namespace:
        return namespace[name]
    raise AssertionError(f'{name} is not a module-level constant')


def constant_function(name):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'app_v3.py', 'exec'), namespace)
    return namespace[name]


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


class TranscriptDateRuleTests(unittest.TestCase):
    """A real note reproduced "71 states" (there are 50) and left "101
    implementation" uncorrected although the questioner said "October 1st"
    aloud in the same exchange."""

    def setUp(self):
        self.rule = constant('TRANSCRIPT_DATE_RULE')

    def test_the_specific_shorthands_are_named(self):
        for shorthand, meaning in (('"11" = 1/1', 'January 1'),
                                   ('"71" = 7/1', None),
                                   ('"101" = 10/1', 'October 1')):
            self.assertIn(shorthand, self.rule, shorthand)
            if meaning:
                self.assertIn(meaning, self.rule, meaning)

    def test_the_two_observed_failures_are_called_out_by_name(self):
        self.assertIn('"71 states" means states whose rate cycle begins 7/1', self.rule)
        self.assertIn('"101 implementation" means an October 1 implementation', self.rule)

    def test_a_digit_run_about_timing_may_not_become_a_count(self):
        self.assertIn('Never reproduce a bare digit run as a count', self.rule)
        self.assertIn('there are only 50 states', self.rule)

    def test_a_year_may_follow_the_date(self):
        self.assertIn('"11 27" = 1/1/27', self.rule)

    def test_the_rule_reaches_the_audio_summary_prompt(self):
        self.assertIn('{TRANSCRIPT_DATE_RULE}', SOURCE)


class CorrectionsLogTests(unittest.TestCase):
    """The log filled with glossary entries such as
    '"ICHRA" -> ICHRA ... Transcript rendered correctly', including terms that
    were not in the transcript at all."""

    def test_only_changed_terms_may_be_logged(self):
        self.assertIn('Log ONLY terms whose wording you changed', SOURCE)
        self.assertIn('never log a term that does not appear in the source', SOURCE)

    def test_an_identity_entry_is_named_as_the_defect(self):
        self.assertIn('never write an entry whose left and right sides are the same word', SOURCE)
        self.assertIn('is a glossary entry, not a correction', SOURCE)

    def test_it_is_not_a_place_to_define_acronyms(self):
        self.assertIn('not a place to define or expand acronyms', SOURCE)

    def test_the_empty_case_still_has_a_defined_output(self):
        self.assertIn('If zero corrections: <p>No corrections required.</p>', SOURCE)


class SourceTypeConsistencyTests(unittest.TestCase):
    """The Brief said INVESTOR/PUBLIC while Key Takeaways said MGMT 1:1 for
    the same meeting, because each tier classified independently."""

    def test_the_brief_is_given_the_classification_already_made(self):
        self.assertIn('_classified_source_type(summary_html)', SOURCE)
        self.assertIn('{source_type_directive}', SOURCE)
        self.assertIn('Reuse that classification verbatim; do not reclassify it.', SOURCE)

    def test_the_helper_reads_either_rendering_of_the_line(self):
        extract = constant_function('_classified_source_type')
        self.assertEqual(extract('<p>Source type: MGMT 1:1 — private</p>'), 'MGMT 1:1')
        self.assertEqual(extract('> Source type: INVESTOR/PUBLIC — conference'), 'INVESTOR/PUBLIC')

    def test_an_unclassified_summary_leaves_the_brief_to_decide(self):
        self.assertEqual(constant_function('_classified_source_type')('<p>nothing</p>'), '')
