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
        # Neither wording of the rule held on a real note: "one quoted phrase
        # per point" was ignored, and a stated numeric quota was over-complied
        # with by deleting evidence. The ceiling is enforced in code instead.
        self.assertIn('should be reported instead', rules)

    def test_the_limit_is_not_stated_in_the_prompt(self):
        # Stating a numeric quota made the model comply by deleting evidence:
        # takeaways came back at 0.2 quotes per 1k against a limit of 3.0 and
        # the note lost 35% of its content. The ceiling lives in code instead.
        for key in summary_comparison.QUOTE_LIMITS:
            self.assertNotIn('QUOTA', summary_comparison.SECTIONS[key], key)

    def test_converting_a_quote_may_not_drop_its_content(self):
        self.assertIn('never means dropping the fact', summary_comparison.RULES)

    def test_sections_may_not_defer_to_each_other(self):
        self.assertIn('must stand on its own', summary_comparison.RULES)
        self.assertNotIn('refer to them in a short clause', summary_comparison.RULES)

    def test_numbered_lists_are_refused(self):
        self.assertIn('Never use numbered lists', summary_comparison.RULES)

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
        clean = 'Clean rewritten takeaways carrying the same content. ' * 20
        state, calls = self.run_generate([stuffed, clean])
        self.assertEqual(state['sections']['takeaways'], clean)
        repair = state['quoteRepairs']['takeaways']
        self.assertEqual(repair['kept'], 'repair')
        self.assertTrue(repair['resolved'])
        self.assertTrue(any('has a defect' in c for c in calls))

    def test_a_repair_that_shrinks_the_section_is_discarded(self):
        stuffed = ('Management said ' + ' '.join(f'"fragment {i}"' for i in range(30))
                   + ' plus substantial reported detail. ' * 30)
        state, _calls = self.run_generate([stuffed, 'Terse rewrite.'])
        # The repair traded content for form, which is worse than the defect.
        self.assertEqual(state['sections']['takeaways'], stuffed)
        self.assertEqual(state['quoteRepairs']['takeaways']['kept'], 'original')
        self.assertFalse(state['quoteRepairs']['takeaways']['resolved'])

    def test_a_clean_section_is_not_redrafted(self):
        state, calls = self.run_generate([])
        self.assertEqual(state['quoteRepairs'], {})
        self.assertFalse(any('has a defect' in c for c in calls))

    def test_no_section_is_shown_the_others(self):
        # Supplying earlier sections made later ones defer to them instead of
        # standing alone, and cost the Q&A log two thirds of its exchanges.
        _state, calls = self.run_generate([])
        self.assertFalse(any('already written' in c for c in calls))


class QaFloorTests(unittest.TestCase):
    def test_a_collapsed_qa_log_is_flagged_against_a_question_rich_source(self):
        source = 'Why is that? ' * 20
        finding = summary_comparison.qa_findings('Q: one\nA: answer', source)
        self.assertIn('only 1 exchange', finding)

    def test_a_full_qa_log_passes(self):
        source = 'Why is that? ' * 20
        log = '\n'.join(f'Q: question {i}\nA: answer {i}' for i in range(6))
        self.assertEqual(summary_comparison.qa_findings(log, source), '')

    def test_a_source_without_questions_is_not_flagged(self):
        self.assertEqual(summary_comparison.qa_findings('No exchanges here.', 'A document with no questions.'), '')


class FigureCoverageTests(unittest.TestCase):
    """A note silently dropped the 12-14% long-term algorithm and it was only
    caught by grepping for that one string."""

    def test_figures_normalise_across_spacing_and_dash_style(self):
        found = summary_comparison.figures('12 – 14 %, $5 billion, 99.8%, 80-90 percent')
        self.assertIn('12-14%', found)
        self.assertIn('$5billion', found)
        self.assertIn('99.8%', found)
        self.assertIn('80-90%', found)

    def test_a_figure_in_the_records_but_not_the_note_is_reported(self):
        coverage = summary_comparison.figure_coverage(
            'The note keeps 99.8% on-time.',
            'Records carry 12-14%, $5 billion and 99.8%.')
        self.assertEqual(coverage['missing'], ['$5billion', '12-14%'])
        self.assertEqual(coverage['checked'], 3)

    def test_a_note_carrying_every_figure_reports_nothing_missing(self):
        records = 'Records carry 12-14% and $5 billion.'
        self.assertEqual(summary_comparison.figure_coverage(records, records)['missing'], [])

    def test_generate_records_coverage_for_the_finished_note(self):
        def ask(rules, prompt, tokens):
            return 'Record mentions 12-14% growth.' if prompt.startswith('SOURCE PART') else 'Note prose.'
        state = {}
        summary_comparison.generate('Body.', state, ask, lambda _s: None)
        self.assertIn('12-14%', state['figureCoverage']['missing'])


class AssessmentBloatTests(unittest.TestCase):
    def test_an_assessment_longer_than_the_takeaways_is_flagged(self):
        # The real v4 note: 11,233 characters of assessment against 6,767 of
        # takeaways, opening with "MANAGEMENT STATEMENTS".
        finding = summary_comparison.assessment_findings('x' * 11233, 'y' * 6767)
        self.assertIn('1.7 times', finding)
        self.assertIn('restating the management record', finding)

    def test_a_proportionate_assessment_passes(self):
        self.assertEqual(summary_comparison.assessment_findings('x' * 3198, 'y' * 6088), '')

    def test_the_shrink_guard_does_not_veto_a_deliberate_trim(self):
        calls = []

        def ask(rules, prompt, tokens):
            calls.append(prompt)
            if prompt.startswith('SOURCE PART'):
                return 'Evidence record.'
            if prompt.startswith('Do not restate'):
                # First the bloated draft, then the intended shorter rewrite.
                return 'A' * 9000 if len([c for c in calls if c.startswith('Do not restate')]) == 1 else 'Tight assessment.'
            return 'B' * 4000

        state = {}
        summary_comparison.generate('Body.', state, ask, lambda _s: None)
        self.assertEqual(state['sections']['assessment'], 'Tight assessment.')
        self.assertEqual(state['quoteRepairs']['assessment']['kept'], 'repair')
