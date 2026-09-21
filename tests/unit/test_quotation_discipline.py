"""Key Takeaways were 21% quoted words, a third of the quotes three words or less.

The prompt said what a quotation mark means but never when to use one, while
telling the model four times to "preserve". Quoting is the defensively safe way
to preserve -- a paraphrase can break fidelity, a quote cannot -- so it quoted
everything worth keeping.
"""
import unittest

import summary_lab


class QuotingRuleTests(unittest.TestCase):
    def test_the_prompt_says_when_a_quotation_is_warranted(self):
        for clue in ('wording is itself the evidence', 'a hedge or a', 'non-answer'):
            self.assertIn(clue, summary_lab.RULES, clue)

    def test_short_fragments_are_called_out_as_the_common_case(self):
        self.assertIn('three words or fewer', summary_lab.RULES)
        self.assertIn('almost never earns its marks', summary_lab.RULES)

    def test_paraphrasing_is_named_as_the_work_rather_than_a_risk(self):
        self.assertIn('say what was meant in your own words', summary_lab.RULES)
        self.assertIn('preserving the meaning is what fidelity requires', summary_lab.RULES)

    def test_the_accuracy_rule_survives_the_restraint_rule(self):
        # Quoting less must not license quoting loosely.
        self.assertIn('Quotation marks mean exact source', summary_lab.RULES)
        self.assertIn('never a paraphrase or corrected transcription', summary_lab.RULES)

    def test_takeaways_states_that_it_is_a_conclusion_not_an_excerpt(self):
        spec = summary_lab.SECTIONS['takeaways']
        self.assertIn('your own words', spec)
        self.assertIn('not an excerpt', spec)

    def test_evidence_quoting_is_still_required_where_it_is_the_point(self):
        # The assessment must still quote the wording that shows an evasion.
        self.assertIn('quote the wording that shows it', summary_lab.SECTIONS['assessment'])

    def test_the_version_moves_so_a_reimport_does_not_match_an_old_row(self):
        self.assertEqual(summary_lab.VERSION, 'source-reviewed-lab-v4')


class TranscriptRepairTests(unittest.TestCase):
    """The ETN note flagged transcription artefacts in four separate sections
    and again in the unresolved list, so it read as if it were arguing with its
    own transcript while making a point."""

    def test_a_reading_settled_by_context_is_written_settled(self):
        self.assertIn('write the settled reading', summary_lab.RULES)
        self.assertIn('record what you changed in the review', summary_lab.RULES)

    def test_the_repair_is_not_narrated_in_the_prose(self):
        self.assertIn('Do not narrate the repair mid-paragraph', summary_lab.RULES)

    def test_a_genuine_ambiguity_is_raised_once_not_in_every_section(self):
        self.assertIn('would change what the reader concludes', summary_lab.RULES)
        self.assertIn('rather than again in every section that touches it', summary_lab.RULES)

    def test_domain_inference_is_protected_rather_than_suppressed(self):
        # Reading "three and a half" as $3.5m/MW is the conclusion the reader
        # wants. The rule must not turn restraint about repairs into timidity
        # about judgement.
        self.assertIn('This governs repairs, not judgement', summary_lab.RULES)
        self.assertIn('is your conclusion: state it plainly', summary_lab.RULES)
        self.assertIn('say once that it was inferred', summary_lab.RULES)
