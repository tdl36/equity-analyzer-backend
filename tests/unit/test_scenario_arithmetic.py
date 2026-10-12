import copy
import unittest
from scenario_arithmetic import check


class ScenarioArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.row = {k: {'statement': v} for k, v in {
            'eps_or_fcf': 'FY2027 adjusted diluted EPS USD 6.00/share',
            'valuation_multiple': '18.0x FY2027 adjusted diluted EPS; rationale: recovery versus peers',
            'implied_value': 'USD 108.00/share',
            'upside_downside': '+8.43% price return vs USD 99.60/share on 2026-10-09',
        }.items()}

    def test_valid_arithmetic_does_not_mutate_or_certify_assumptions(self):
        before = copy.deepcopy(self.row)
        result = check(self.row)
        self.assertEqual(result['status'], 'checked')
        self.assertIn('still require review', result['message'])
        self.assertEqual(self.row, before)

    def test_separate_value_and_return_errors(self):
        for field, value, error in [
            ('implied_value', 'USD 109.00/share', 'implied value'),
            ('upside_downside', '+9.43% price return vs USD 99.60/share on 2026-10-09', 'price return'),
        ]:
            with self.subTest(field=field):
                row = copy.deepcopy(self.row); row[field]['statement'] = value
                result = check(row)
                self.assertEqual(result['status'], 'mismatch'); self.assertIn(error, result['message'])

    def test_period_basis_and_currency_must_match(self):
        for field, old, new in [('valuation_multiple','2027','2028'), ('valuation_multiple','adjusted','GAAP'), ('implied_value','USD','EUR')]:
            row = copy.deepcopy(self.row); row[field]['statement'] = row[field]['statement'].replace(old,new)
            self.assertEqual(check(row)['status'],'mismatch')

    def test_no_inference_from_narrative_or_unsupported_method(self):
        for value in ('EPS $6', 'FY2027 adjusted diluted EPS USD 5-6/share', 'FY2027 adjusted diluted EPS USD 6.00/share if demand holds', 'FY2027 adjusted FCF USD 6.00/share', 'Unavailable', '', 'FY2027 adjusted diluted EPS USD 999999999999999999999999/share'):
            row = copy.deepcopy(self.row); row['eps_or_fcf']['statement'] = value
            self.assertEqual(check(row)['status'],'unverified')
        self.assertEqual(check({})['status'],'unverified')

    def test_invalid_date_and_nonpositive_inputs(self):
        for field, old, new in [('upside_downside','2026-10-09','2026-02-30'), ('upside_downside','99.60','0'), ('eps_or_fcf','6.00','0'), ('valuation_multiple','18.0','0')]:
            row = copy.deepcopy(self.row); row[field]['statement'] = row[field]['statement'].replace(old,new)
            self.assertEqual(check(row)['status'],'unverified')

    def test_rounding_and_price_return_only(self):
        self.row['upside_downside']['statement'] = '+8% price return vs USD 99.60/share on 2026-10-09'
        self.assertEqual(check(self.row)['status'],'checked')
        self.row['upside_downside']['statement'] = '+8% total return vs USD 99.60/share on 2026-10-09'
        self.assertEqual(check(self.row)['status'],'unverified')
        self.row['upside_downside']['statement'] = '-10.00% price return vs USD 120.00/share on 2026-10-09'
        self.assertEqual(check(self.row)['status'],'checked')
