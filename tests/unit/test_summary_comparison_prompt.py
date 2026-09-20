"""The Improved-notes prompt contract: readable prose, and sections that reach the UI."""
import re
import unittest
from pathlib import Path

import summary_comparison


class SectionContractTests(unittest.TestCase):
    """A section the workspace does not know about is generated and never shown."""

    def ui_section_keys(self):
        source = Path('src/summary-comparison.jsx').read_text()
        line = next(l for l in source.splitlines() if l.startswith('const sections='))
        return [m.group(1) for m in re.finditer(r"\['([a-z]+)',", line)]

    def test_every_generated_section_is_rendered_by_the_workspace(self):
        ui = self.ui_section_keys()
        for key in summary_comparison.SECTIONS:
            self.assertIn(key, ui, f'{key} is generated but never displayed')

    def test_the_workspace_shows_only_sections_that_exist(self):
        # 'record' is assembled from the per-part records, not from SECTIONS.
        known = set(summary_comparison.SECTIONS) | {'record'}
        for key in self.ui_section_keys():
            self.assertIn(key, known, f'{key} is displayed but never generated')

    def test_the_qa_log_is_one_of_them(self):
        self.assertIn('qa', summary_comparison.SECTIONS)
        self.assertIn('qa', self.ui_section_keys())


class PromptDisciplineTests(unittest.TestCase):
    """Guard the instructions added because the earlier prompt produced
    quote-stuffed, repetitive notes with broken numbered lists."""

    def test_quoting_is_reserved_rather_than_default(self):
        rules = summary_comparison.RULES.lower()
        self.assertIn('reported speech is the default', rules)
        self.assertIn('one short quoted phrase per point', rules)

    def test_numbered_lists_are_refused(self):
        self.assertIn('Never use numbered lists', summary_comparison.RULES)

    def test_facts_are_stated_once(self):
        self.assertIn('State each fact once', summary_comparison.RULES)

    def test_takeaways_carry_scannable_topic_tags(self):
        instruction = summary_comparison.SECTIONS['takeaways']
        self.assertIn('[GUIDANCE]', instruction)
        self.assertIn('[M&A]', instruction)

    def test_the_qa_log_may_not_invent_an_exchange(self):
        instruction = summary_comparison.SECTIONS['qa']
        self.assertIn('do not invent questions', instruction)
        self.assertIn('no genuine question-and-answer structure', instruction)

    def test_a_prompt_change_carries_a_new_version(self):
        # summary_comparisons is unique on (summary_id, source_hash, version),
        # so reusing a version would mix notes from two different prompts.
        self.assertNotEqual(summary_comparison.VERSION, 'conservative-v1')


class GenerationCoverageTests(unittest.TestCase):
    def test_generate_drafts_every_section_including_the_qa_log(self):
        asked = []

        def ask(rules, prompt, tokens):
            asked.append(prompt)
            return 'drafted'

        state = {}
        summary_comparison.generate('A short transcript body.', state, ask, lambda _s: None)
        self.assertEqual(set(state['sections']), set(summary_comparison.SECTIONS))
        self.assertEqual(state['sections']['qa'], 'drafted')
        # Each section is drafted from the evidence records, not from scratch.
        self.assertTrue(all('evidence records' in p for p in asked[1:]))
