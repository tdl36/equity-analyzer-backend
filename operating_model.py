"""Versioned analyst-entered EV/EBITDA scenarios; no model calls or market feeds."""
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import re

VERSION = 'ev-ebitda-v1'
NAMES = ('bear', 'base', 'bull')


def text(value, label, limit=1800):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'Enter {label} (up to {limit} characters).')
    return value.strip()


def number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)) or len(str(value)) > 32:
        raise ValueError(f'Enter a finite number for {label}.')
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f'Enter a number for {label}.')
    if not result.is_finite() or not Decimal(str(minimum)) <= result <= Decimal(str(maximum)):
        raise ValueError(f'{label} must be between {minimum} and {maximum}.')
    if result != result.quantize(Decimal('.000001')):
        raise ValueError(f'{label} supports up to six decimal places.')
    return result


def money(value):
    return str(value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def evaluate(data):
    """Validate inputs, normalize units, and replace any client-supplied results."""
    if not isinstance(data, dict):
        raise ValueError('Enter operating-model inputs.')
    if data.get('version') != VERSION:
        raise ValueError('Unsupported operating-model version.')
    if data.get('method') != 'ev_ebitda' or data.get('units') != 'millions':
        raise ValueError('This model requires EV/EBITDA, money in millions and shares in millions.')
    if data.get('suitable') is not True:
        raise ValueError('Confirm this EV/EBITDA method is suitable for the company.')
    currency = text(data.get('currency'), 'three-letter currency', 3).upper()
    if not re.fullmatch('[A-Z]{3}', currency):
        raise ValueError('Use a three-letter currency code.')
    years = []
    for key in ('baseYear', 'targetYear'):
        n = number(data.get(key), key, 1900, 2200)
        if n != n.to_integral_value():
            raise ValueError('Fiscal years must be whole numbers.')
        years.append(int(n))
    horizon = years[1] - years[0]
    if not 1 <= horizon <= 5:
        raise ValueError('Choose a forecast one to five fiscal years after the base year.')
    as_of = text(data.get('asOf'), 'reference-price date', 10)
    date.fromisoformat(as_of)
    base_revenue = number(data.get('baseRevenue'), 'base revenue', '.000001', '1000000000')
    price = number(data.get('referencePrice'), 'reference share price', '.000001', '1000000')
    raw = data.get('scenarios')
    if not isinstance(raw, dict) or set(raw) != set(NAMES):
        raise ValueError('Provide exactly bear, base and bull assumptions.')
    result = dict(version=VERSION, method='ev_ebitda', units='millions', suitable=True,
                  currency=currency, baseYear=years[0], targetYear=years[1], asOf=as_of,
                  baseRevenue=str(base_revenue), referencePrice=str(price),
                  revenueReference=text(data.get('revenueReference'), 'base revenue source / definition'),
                  priceReference=text(data.get('priceReference'), 'reference-price source'),
                  ebitdaBasis=text(data.get('ebitdaBasis'), 'EBITDA definition / adjustments'),
                  scenarios={}, results={})
    if data.get('baseRevenueObservation') is not None:
        from financial_observations import validate_model_link
        result['baseRevenueObservation'] = validate_model_link({**result, 'baseRevenueObservation': data['baseRevenueObservation']})
    with localcontext() as ctx:
        ctx.prec = 60
        for name in NAMES:
            row = raw[name]
            if not isinstance(row, dict):
                raise ValueError(f'Enter {name} assumptions.')
            if row.get('currency', currency) != currency or row.get('units', 'millions') != 'millions' or row.get('targetYear', years[1]) != years[1]:
                raise ValueError('All scenarios must use the model currency, millions and target fiscal year.')
            values = {key: number(row.get(key), f'{name} {key}', low, high) for key, low, high in (
                ('growthPct', '-99', '200'), ('marginPct', '.000001', '100'),
                ('multiple', '.000001', '100'), ('netDebt', '-1000000000', '1000000000'),
                ('otherClaims', '0', '1000000000'), ('nonOperatingAssets', '0', '1000000000'),
                ('shares', '.000001', '1000000000'))}
            result['scenarios'][name] = {key: str(value) for key, value in values.items()}
            result['scenarios'][name]['rationale'] = text(row.get('rationale'), f'{name} assumptions and source references')
            revenue = base_revenue * (1 + values['growthPct'] / 100) ** horizon
            ebitda = revenue * values['marginPct'] / 100
            ev = ebitda * values['multiple']
            residual = ev - values['netDebt'] - values['otherClaims'] + values['nonOperatingAssets']
            equity = max(Decimal(0), residual)
            target = equity / values['shares']
            implied_ebitda = (price * values['shares'] + values['netDebt'] + values['otherClaims'] - values['nonOperatingAssets']) / values['multiple']
            result['results'][name] = dict(revenue=money(revenue), ebitda=money(ebitda),
                enterpriseValue=money(ev), equityValue=money(equity), unflooredEquityValue=money(residual),
                impliedPrice=money(target), priceReturnPct=money((target / price - 1) * 100),
                impliedEbitdaAtReferencePrice=money(implied_ebitda),
                equityFloored=residual < 0, reverseValid=implied_ebitda > 0)
        # One-variable sensitivities around the base assumptions. Preserve full precision.
        base = result['scenarios']['base']
        ebitda = base_revenue * (1 + Decimal(base['growthPct']) / 100) ** horizon * Decimal(base['marginPct']) / 100
        multiple = Decimal(base['multiple'])
        adjustment = -Decimal(base['netDebt']) - Decimal(base['otherClaims']) + Decimal(base['nonOperatingAssets'])
        result['sensitivity'] = [{'ebitdaChangePct':str(shift), 'multiple':str(multiple + delta),
            'impliedPrice':money(max(Decimal(0), ebitda * (1 + Decimal(shift) / 100) * (multiple + delta) + adjustment) / Decimal(base['shares']))}
            for shift in (-10, 0, 10) for delta in (Decimal(-1), Decimal(0), Decimal(1)) if multiple + delta > 0]
    result['warnings'] = [
        'Analyst-entered inputs and manual references are not independently source-verified.',
        'Conditional valuation, not a forecast or trade instruction. No dividends, discounting, FX or probabilities.',
        'Net debt, other claims, nonoperating assets and diluted shares are target-period assumptions; avoid double-counting.',
        'Reverse EBITDA holds each scenario multiple and equity bridge fixed; it is not observed market consensus.'
    ]
    if any(row['equityFloored'] for row in result['results'].values()):
        result['warnings'].append('At least one equity residual is negative and is floored at zero; this is not a recovery or restructuring model.')
    if not result['results']['base']['reverseValid']:
        result['warnings'].append('Base reverse EBITDA is nonpositive; this reverse valuation is not interpretable with this method.')
    return result
