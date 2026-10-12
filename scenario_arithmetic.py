"""Conservative arithmetic checks for explicitly formatted, cited P/E scenarios.

No number extraction from narrative, currency conversion, current-price rebasing,
or judgment about whether a multiple or earnings assumption is reasonable.
"""
import re
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

NUMBER = r'(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,4})?'
BASIS = r'(?:GAAP|adjusted) diluted EPS'
EPS = re.compile(rf'FY(?P<year>20[0-9]{{2}}) (?P<basis>{BASIS}) (?P<currency>[A-Z]{{3}}) (?P<value>{NUMBER})/share')
MULTIPLE = re.compile(rf'(?P<value>{NUMBER})x FY(?P<year>20[0-9]{{2}}) (?P<basis>{BASIS}); rationale: (?P<rationale>[^\n]+)')
PRICE = re.compile(rf'(?P<currency>[A-Z]{{3}}) (?P<value>{NUMBER})/share')
RETURN = re.compile(rf'(?P<value>[+-]?{NUMBER})% price return vs (?P<currency>[A-Z]{{3}}) (?P<price>{NUMBER})/share on (?P<date>20[0-9]{{2}}-[0-9]{{2}}-[0-9]{{2}})')


def check(row):
    """Return diagnostics only; caller supplies already-supported claim objects."""
    def parse(key, pattern):
        claim = row.get(key)
        text = claim.get('statement') if isinstance(claim, dict) else None
        return pattern.fullmatch(text) if isinstance(text, str) else None

    eps = parse('eps_or_fcf', EPS)
    multiple = parse('valuation_multiple', MULTIPLE)
    price = parse('implied_value', PRICE)
    ret = parse('upside_downside', RETURN)
    if not all((eps, multiple, price, ret)):
        return {'status': 'unverified', 'message': 'Arithmetic unverified: explicit P/E inputs, basis, year and dated price-return reference required.'}
    if not multiple['rationale'].strip():
        return {'status': 'unverified', 'message': 'Arithmetic unverified: multiple rationale is missing.'}
    if eps['year'] != multiple['year'] or eps['basis'] != multiple['basis']:
        return {'status': 'mismatch', 'message': 'Review required: earnings and multiple use different years or accounting bases.'}
    if len({eps['currency'], price['currency'], ret['currency']}) != 1:
        return {'status': 'mismatch', 'message': 'Review required: scenario currencies differ; no conversion performed.'}
    if eps['currency'] not in {'USD','EUR','GBP','JPY','CHF','CAD','AUD','HKD','SEK','NOK','DKK','SGD','CNY','KRW','TWD','INR','BRL','MXN','ZAR'}:
        return {'status': 'unverified', 'message': 'Arithmetic unverified: unsupported currency code.'}
    try:
        date.fromisoformat(ret['date'])
        earnings, mult, reference = (Decimal(eps['value']), Decimal(multiple['value']), Decimal(ret['price']))
        if min(earnings, mult, reference) <= 0:
            raise ValueError('Nonpositive P/E input')
        with localcontext() as ctx:
            ctx.prec = 64
            expected = earnings * mult
            expected_return = (expected / reference - 1) * 100
    except (InvalidOperation, ValueError, ArithmeticError):
        return {'status': 'unverified', 'message': 'Arithmetic unverified: invalid date or nonpositive P/E input.'}

    # Compare at the precision explicitly saved, rather than an arbitrary tolerance.
    def agrees(actual, expected):
        value = Decimal(actual)
        unit = Decimal(1).scaleb(value.as_tuple().exponent)
        with localcontext() as ctx:
            ctx.prec = 64
            return value == expected.quantize(unit, rounding=ROUND_HALF_UP)

    problems = []
    if not agrees(price['value'], expected):
        problems.append('implied value does not equal EPS × P/E')
    if not agrees(ret['value'], expected_return):
        problems.append('price return does not match that value and the saved reference price')
    if problems:
        return {'status': 'mismatch', 'message': 'Review required: ' + '; '.join(problems) + '.'}
    return {'status': 'checked', 'message': 'EPS × P/E and price return agree at saved precision. Assumptions and source accuracy still require review.'}
