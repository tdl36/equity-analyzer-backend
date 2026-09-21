"""Budgets tuned for a non-thinking model starved a thinking one.

Two Opus 5 experiments died at the part check: max_tokens caps thinking plus
text, and Opus 5 thinks at the default high effort, so a 3500-token budget was
spent before any output existed.
"""
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import summary_lab

SOURCE = Path('summary_lab.py').read_text()

# Room for a thinking model to think and still answer.
FLOOR = 12000


class BudgetTests(unittest.TestCase):
    def budgets(self):
        return {name: getattr(summary_lab, name) for name in dir(summary_lab)
                if name.startswith('TOKENS_')}

    def test_every_step_has_room_for_a_thinking_model(self):
        found = self.budgets()
        self.assertTrue(found, 'no named token budgets')
        for name, value in found.items():
            self.assertGreaterEqual(value, FLOOR, f'{name} is below the thinking floor')

    def test_the_step_that_failed_is_no_longer_the_smallest_budget(self):
        self.assertGreaterEqual(summary_lab.TOKENS_PART_CHECK, FLOOR)
        self.assertGreater(summary_lab.TOKENS_PART_CHECK, 3500)

    def test_long_sections_still_get_more_room_than_short_ones(self):
        self.assertGreater(summary_lab.TOKENS_LONG_SECTION, summary_lab.TOKENS_SECTION)

    def test_no_call_site_hardcodes_a_budget(self):
        # A literal is how the old cap escaped review; a name states its purpose.
        calls = re.findall(r'ask\(RULES,.*?,\s*([A-Za-z_0-9]+)\s*(?:if|\))', SOURCE, re.S)
        literals = [c for c in calls if c.isdigit()]
        self.assertEqual(literals, [], f'hardcoded budgets: {literals}')

    def test_budgets_stay_within_the_smallest_supported_output_limit(self):
        # Opus 4.6 and later allow 64k+ synchronous output; stay well inside.
        for name, value in self.budgets().items():
            self.assertLess(value, 64000, name)


class TruncationMessageTests(unittest.TestCase):
    """The failure said only "incomplete", which named neither cause nor cure."""

    def call(self, stop_reason):
        message = SimpleNamespace(
            stop_reason=stop_reason, model='claude-opus-5',
            content=[SimpleNamespace(type='text', text='partial')],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1))

        class Stream:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self): return iter(())
            def get_final_message(self): return message

        class Client:
            messages = SimpleNamespace(stream=lambda **kw: Stream())
            def __enter__(self): return self
            def __exit__(self, *a): return False

        return summary_lab.ask_with_recovery(
            'k', 'claude-opus-5', 'sys', 'prompt', 12000, {}, lambda state: None,
            client_factory=lambda **kw: Client(), sleep=lambda s: None)

    def test_hitting_the_limit_names_the_limit_and_the_reason(self):
        with self.assertRaises(Exception) as caught:
            self.call('max_tokens')
        message = str(caught.exception)
        self.assertIn('12,000', message)
        self.assertIn('thinking counts against that limit', message)

    def test_another_stop_reason_is_reported_as_itself(self):
        with self.assertRaises(Exception) as caught:
            self.call('refusal')
        self.assertIn('refusal', str(caught.exception))


class DefaultModelTests(unittest.TestCase):
    def test_the_default_is_unchanged(self):
        self.assertEqual(summary_lab.MODEL, 'claude-opus-4-6')
