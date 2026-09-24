"""Verification runs on a cheaper model than composition.

Checking a draft against one source part reads a section and a passage and
reports discrepancies; the findings feed a revision rather than reaching the
reader. Those calls are 47% of a run and about 25% of its cost.
"""
import inspect
import unittest
from pathlib import Path

import summary_lab

SOURCE = Path('summary_lab.py').read_text()


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.state = {'parts': {}, 'sections': {}, 'checks': {}, 'reductions': {}}

    def test_the_two_models_are_different_and_the_cheaper_one_verifies(self):
        self.assertNotEqual(summary_lab.MODEL, summary_lab.CHECK_MODEL)
        self.assertIn('sonnet', summary_lab.CHECK_MODEL)
        self.assertIn('opus', summary_lab.MODEL)

    def test_the_check_model_is_overridable_without_a_deploy(self):
        self.assertIn('CHARLIE_SUMMARY_LAB_CHECK_MODEL', SOURCE)

    def test_generate_falls_back_to_one_caller_when_none_is_supplied(self):
        # Older callers pass four arguments; they must keep working.
        params = inspect.signature(summary_lab.generate).parameters
        self.assertIsNone(params['check'].default)
        source = inspect.getsource(summary_lab.generate)
        self.assertIn('check = check or ask', source)

    def test_only_the_part_check_uses_the_cheaper_caller(self):
        source = inspect.getsource(summary_lab.generate)
        self.assertIn("state['partChecks'][checkkey] = check(", source)
        # Drafting and revision stay on the composition model.
        self.assertEqual(source.count('= check('), 1)
        self.assertGreaterEqual(source.count('= ask('), 3)


class UsageLabellingTests(unittest.TestCase):
    """The ledger recorded a flat per-experiment cost, so where the money went
    inside a run had to be modelled rather than read."""

    def test_each_call_records_which_step_it_was(self):
        self.assertIn("usage_for('generate')", SOURCE)
        self.assertIn("usage_for('check')", SOURCE)
        self.assertIn("'step': step", SOURCE)

    def test_the_experiment_id_is_still_recorded(self):
        self.assertIn("'experiment': jid", SOURCE)
