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
        # "at most one quoted phrase per point" was ignored in 8 of 9 takeaways
        # on a real note, so the rule is now a stated quota backed by a check.
        self.assertIn('that quota is a hard limit', rules)

    def test_every_analysis_section_states_its_quota(self):
        for key in summary_comparison.QUOTE_LIMITS:
            self.assertIn('QUOTA', summary_comparison.SECTIONS[key], key)

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


class VersionVisibilityTests(unittest.TestCase):
    """Without the current version the workspace cannot tell an older note
    apart from a current one, and cannot offer to produce the current one."""

    def test_the_listing_route_reports_the_pipeline_version(self):
        source = Path('summary_comparison.py').read_text()
        listing = source[source.index("def get(sid):"):source.index("def enqueue(")]
        self.assertIn('currentVersion=VERSION', listing)

    def test_the_workspace_offers_a_fresh_run_when_every_note_is_older(self):
        ui = Path('src/summary-comparison.jsx').read_text()
        self.assertIn("!rows.some(r=>r.version===currentVersion)", ui)
        # A fresh run must not pass resumeId, which would resume the old row
        # under its own version instead of producing the current one.
        self.assertIn("resumeId:fresh?undefined:row?.id", ui)
        self.assertIn("start(true)", ui)


class QuoteEnforcementTests(unittest.TestCase):
    """The quota failed as an instruction alone, so it is checked in code."""

    def block(self, tag, quotes):
        body = ' '.join(f'management said "phrase {i} of the answer"' for i in range(quotes))
        return f'[{tag}] **A claim.** {body} ' + 'Plain reported prose to give the block length. ' * 12

    def test_a_quote_stuffed_takeaway_block_is_named(self):
        text = self.block('COMPETITIVE POSITIONING', 9) + '\n' + self.block('M&A', 1)
        finding = summary_comparison.quote_findings('takeaways', text)
        self.assertIn('tagged takeaway', finding)
        self.assertIn('worst holding 9', finding)

    def test_restrained_prose_passes(self):
        text = self.block('GUIDANCE', 1) + '\n' + self.block('M&A', 1)
        self.assertEqual(summary_comparison.quote_findings('takeaways', text), '')

    def test_the_brief_has_an_absolute_ceiling(self):
        body = ' '.join(f'management said "committed figure {i}"' for i in range(9))
        finding = summary_comparison.quote_findings('brief', body + ' plain prose. ' * 200)
        self.assertIn('limit of 6', finding)

    def test_the_record_sections_are_exempt_because_verbatim_is_their_job(self):
        stuffed = ' '.join(f'"quoted answer {i}"' for i in range(40))
        self.assertEqual(summary_comparison.quote_findings('qa', stuffed), '')
        self.assertEqual(summary_comparison.quote_findings('record', stuffed), '')


class RepairPassTests(unittest.TestCase):
    def run_generate(self, drafts):
        """drafts: list of section texts returned in order, per section."""
        calls = []
        queue = list(drafts)

        def ask(rules, prompt, tokens):
            calls.append(prompt)
            if prompt.startswith('SOURCE PART'):
                return 'Evidence record [Part 1].'
            return queue.pop(0) if queue else 'clean prose without quotation.'

        state = {}
        summary_comparison.generate('Body text.', state, ask, lambda _s: None)
        return state, calls

    def test_a_violating_section_is_redrafted_once_and_the_repair_is_used(self):
        stuffed = ' '.join(f'"fragment {i}"' for i in range(30))
        # First section drafted is 'takeaways'; give it a stuffed draft, then a clean one.
        state, calls = self.run_generate([stuffed, 'clean rewritten takeaways.'])
        self.assertEqual(state['sections']['takeaways'], 'clean rewritten takeaways.')
        repair = state['quoteRepairs']['takeaways']
        self.assertTrue(repair['resolved'])
        self.assertIn('broke the quoting quota', [c for c in calls if 'broke the quoting quota' in c][0])

    def test_a_clean_section_is_not_redrafted(self):
        state, calls = self.run_generate([])
        self.assertEqual(state['quoteRepairs'], {})
        self.assertFalse(any('broke the quoting quota' in c for c in calls))

    def test_later_sections_are_told_what_earlier_ones_already_said(self):
        _state, calls = self.run_generate([])
        section_calls = [c for c in calls if not c.startswith('SOURCE PART')]
        self.assertNotIn('already written', section_calls[0])
        self.assertIn('already written', section_calls[-1])
