"""The Original and Summary Lab must answer to one doctrine, not two copies."""
import ast
import unittest
from pathlib import Path

import research_doctrine
import summary_lab

APP = Path('app_v3.py').read_text()
LAB_SOURCE = Path('summary_lab.py').read_text()


class OneDefinitionTests(unittest.TestCase):
    """Summary Lab kept its own older copy because it cannot import app_v3."""

    def test_neither_pipeline_defines_its_own_doctrine(self):
        for source, name in ((APP, 'app_v3.py'), (LAB_SOURCE, 'summary_lab.py')):
            redefined = [n.targets[0].id for n in ast.parse(source).body
                         if isinstance(n, ast.Assign)
                         and getattr(n.targets[0], 'id', '') in ('RESEARCH_DOCTRINE', 'TRANSCRIPT_DATE_RULE')]
            self.assertEqual(redefined, [], f'{name} redefines shared doctrine')

    def test_the_original_still_imports_the_names_its_prompts_use(self):
        self.assertIn('from research_doctrine import TRANSCRIPT_DATE_RULE, RESEARCH_DOCTRINE', APP)

    def test_the_shared_module_depends_on_nothing(self):
        # It has to stay importable from both sides, so it may not import either.
        imports = [n for n in ast.parse(Path('research_doctrine.py').read_text()).body
                   if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports, [])


class CredibilityScaleTests(unittest.TestCase):
    """T94/T95 decided the Lab was wrong to forbid a rating; it kept forbidding one."""

    def test_the_lab_assessment_carries_the_same_scale_as_the_original(self):
        self.assertIn('5 — specific', summary_lab.SECTIONS['assessment'])
        self.assertIn('1 — internally contradictory', summary_lab.SECTIONS['assessment'])

    def test_no_prompt_still_forbids_a_numerical_rating(self):
        for text in [summary_lab.RULES, *summary_lab.SECTIONS.values()]:
            self.assertNotIn('numerical credibility scores', text)

    def test_the_scale_is_fixed_so_notes_stay_comparable(self):
        self.assertIn('Do not invent decimals', research_doctrine.RESEARCH_DOCTRINE)

    def test_interior_state_is_still_out_of_bounds(self):
        self.assertIn('never anyone’s interior state'.replace('’', "'"), summary_lab.RULES)
        self.assertIn('not evidence of an interior state', research_doctrine.RESEARCH_DOCTRINE)


class DateShorthandTests(unittest.TestCase):
    """T97 taught the Original that "101" is a date; the Lab never heard."""

    def test_the_lab_receives_the_shorthand(self):
        for fragment in ('"101" = 10/1', '"71 states"', 'a rate cycle is a date'):
            self.assertIn(fragment, summary_lab.RULES, fragment)

    def test_the_lab_is_not_told_about_buckets_it_does_not_have(self):
        self.assertNotIn('Bucket A', summary_lab.RULES)
        self.assertIn('Bucket A', research_doctrine.TRANSCRIPT_DATE_RULE)


class AttributionTests(unittest.TestCase):
    """Follow-up Questions printed "Speaker 1/2" 13 times while the note itself
    warned the labels appear to swap partway through the source."""

    def test_raw_speaker_labels_are_forbidden(self):
        self.assertIn('Never print them', summary_lab.RULES)
        self.assertIn('transcription artefact', summary_lab.RULES)

    def test_the_fallback_is_a_role_or_an_admission_not_a_guess(self):
        self.assertIn('the role the source makes clear', summary_lab.RULES)
        self.assertIn('instead of inventing one', summary_lab.RULES)


class SectionHeadingTests(unittest.TestCase):
    """The Executive Brief titled itself "Key Takeaways"."""

    def test_sections_may_not_borrow_another_section_name(self):
        self.assertIn('Never title one section after', summary_lab.RULES)
        self.assertIn('collides with the section of that name', summary_lab.RULES)
