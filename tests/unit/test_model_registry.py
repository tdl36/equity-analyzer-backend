"""Every offered model must have a price, and the price must not be a stale tier.

The table carried (15, 75) for the whole Opus family -- the retired Opus 4/4.1
tier -- so every Opus cost Charlie reported was three times the real figure.
The existing guard only caught a *missing* model (priced at 0); nothing caught
a wrong one. Rates verified against platform.claude.com on 2026-09-20.
"""
import ast
import unittest
from pathlib import Path

TREE = ast.parse(Path('app_v3.py').read_text())


def constant(name):
    namespace = {}
    for node in TREE.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'app_v3.py', 'exec'), namespace)
            return namespace[name]
    raise AssertionError(f'{name} not found')


PRICES = constant('LLM_PRICES')
PICKER = constant('PICKER_MODELS')

# Anthropic's published rates, input/output USD per million tokens.
PUBLISHED = {
    'claude-opus-5': (5.0, 25.0),
    'claude-opus-4-8': (5.0, 25.0),
    'claude-opus-4-7': (5.0, 25.0),
    'claude-opus-4-6': (5.0, 25.0),
    'claude-fable-5-1': (10.0, 50.0),
    'claude-fable-5': (10.0, 50.0),
    'claude-sonnet-5': (2.0, 10.0),
    'claude-sonnet-4-6': (3.0, 15.0),
    'claude-haiku-4-5-20251001': (1.0, 5.0),
}


class PriceTests(unittest.TestCase):
    def test_every_anthropic_price_matches_the_published_rate(self):
        for model, rate in PUBLISHED.items():
            self.assertEqual(PRICES.get(model), rate, model)

    def test_no_model_is_still_on_the_retired_opus_tier(self):
        # (15, 75) is Opus 4 / 4.1, retired. Nothing current costs that.
        self.assertNotIn((15.0, 75.0), PRICES.values())

    def test_every_offered_model_has_a_price(self):
        # An unpriced model records as 0, which reads as free rather than unknown.
        unpriced = [m['model'] for m in PICKER if m['model'] not in PRICES]
        self.assertEqual(unpriced, [])

    def test_output_always_costs_more_than_input(self):
        for model, (inp, out) in PRICES.items():
            self.assertGreater(out, inp, model)


class PickerTests(unittest.TestCase):
    def test_the_current_lineup_is_offered(self):
        models = {m['model'] for m in PICKER}
        for model in ('claude-opus-5', 'claude-opus-4-8', 'claude-fable-5-1', 'claude-sonnet-5'):
            self.assertIn(model, models, model)

    def test_keys_and_model_ids_are_unique(self):
        keys = [m['key'] for m in PICKER]
        models = [m['model'] for m in PICKER]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(models), len(set(models)))

    def test_the_default_is_a_model_the_picker_offers(self):
        self.assertIn(constant('PICKER_DEFAULT_MODEL'), {m['key'] for m in PICKER})

    def test_fable_is_not_described_as_merely_expressive(self):
        # Anthropic positions Fable 5.1 for demanding reasoning and long-horizon
        # agentic work, not as the expressive option.
        fable = next(m for m in PICKER if m['key'] == 'fable-5-1')
        self.assertNotIn('expressive', fable['note'].lower())
