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
