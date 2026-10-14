"""Conservative historical P/E diagnostics; no fetching, approval or source certification.

Inputs are explicit source-backed observations. This arithmetic layer cannot prove
that an asserted share basis or source extraction is correct. A later adapter must
verify originals and freeze the supplement before investor review/rendering.
"""
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import re

_DECIMAL = re.compile(r'-?(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,6})?')
_HASH = re.compile(r'[0-9a-f]{64}')


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'20\d{2}-\d{2}-\d{2}', value):
        raise ValueError('Invalid date')
    return date.fromisoformat(value)


def _number(value):
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        raise ValueError('Explicit decimal string required')
    return Decimal(value)


def _source(value):
    # Identity receipts only. Do not fetch a caller-supplied URL or trust its body.
    return (isinstance(value, dict) and isinstance(value.get('sha256'), str)
            and bool(_HASH.fullmatch(value['sha256']))
            and isinstance(value.get('id'), str) and 0 < len(value['id']) <= 200)


def check_observation(observation):
    """Diagnose one explicit trailing annual GAAP P/E pair without mutating inputs.

    Same-day publication is withheld because a date alone cannot establish whether
    earnings were public before the close. No split adjustment, TTM construction,
    restatement selection or forward-estimate substitution is inferred here.
    """
    def gap(reason):
        return {'status': 'unverified', 'message': reason}
    if not isinstance(observation, dict):
        return gap('A dated price and earnings observation are required.')
    price, earnings = observation.get('price'), observation.get('earnings')
    if not isinstance(price, dict) or not isinstance(earnings, dict):
        return gap('A dated price and earnings observation are required.')
    if not _source(price.get('source')) or not _source(earnings.get('source')):
        return gap('Both inputs require source identity and SHA-256 receipts; originals still require verification.')
    for key in ('securityId', 'currency', 'shareBasis'):
        a, b = price.get(key), earnings.get(key)
        if not isinstance(a, str) or not a.strip() or a != b:
            return gap('Price and earnings must explicitly share security, currency and share basis.')
    if price.get('basis') != 'unadjusted_close' or earnings.get('basis') != 'GAAP diluted EPS':
        return gap('Only unadjusted close and annual GAAP diluted EPS are supported.')
    if earnings.get('periodType') != 'annual' or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', str(earnings.get('accession', ''))):
        return gap('Annual earnings require a filing accession; no TTM or forward earnings are inferred.')
    try:
        observed = _date(price.get('date'))
        published = _date(earnings.get('publishedDate'))
        start, end = _date(earnings.get('periodStart')), _date(earnings.get('periodEnd'))
        if not 350 <= (end - start).days <= 380:
            return gap('An annual earnings period of 350–380 days is required.')
        if not end <= published < observed:
            return gap('The selected earnings filing must be published before the price date; same-day timing is unresolved.')
        if (observed - end).days > 550:
            return gap('The annual earnings period is too old for this bounded historical comparison.')
        close, eps = _number(price.get('value')), _number(earnings.get('value'))
        if close <= 0:
            return gap('A positive closing price is required.')
        if eps <= 0:
            return {'status': 'not_meaningful', 'message': 'P/E is not meaningful for zero or negative earnings.'}
        with localcontext() as ctx:
            ctx.prec = 40
            multiple = close / eps
    except (ValueError, InvalidOperation, ArithmeticError):
        return gap('Invalid dates or decimal values; no valuation inferred.')
    return {'status': 'calculated', 'multiple': format(multiple, '.4f'),
            'priceDate': price['date'], 'earningsPeriodEnd': earnings['periodEnd'],
            'message': 'Close ÷ annual GAAP diluted EPS only. Source accuracy, latest available filing and share-basis compatibility still require review.'}


def check_supplement(supplement, originals):
    """Check a bounded inventory against caller-supplied original bytes.

    No path/URL is opened here. Byte equality establishes source identity only, not
    extraction accuracy, share compatibility or latest-filing completeness. The
    returned digest identifies all submitted inputs, including invalid rows.
    """
    import hashlib
    import json
    if not isinstance(supplement, dict) or set(supplement) != {'version', 'securityId', 'observations'}:
        raise ValueError('A versioned security and observation inventory is required.')
    if type(supplement['version']) is not int or supplement['version'] != 1:
        raise ValueError('Unsupported historical valuation supplement version.')
    security = supplement['securityId']
    rows = supplement['observations']
    if not isinstance(security, str) or not security.strip() or len(security) > 200:
        raise ValueError('Explicit bounded security identity required.')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 60:
        raise ValueError('Provide 1–60 explicit observations; no automatic sampling.')
    if not isinstance(originals, dict):
        raise ValueError('An original-byte inventory is required.')
    try:
        encoded = json.dumps(supplement, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError('Supplement must be finite JSON data.') from exc
    if len(encoded) > 250000:
        raise ValueError('Historical supplement exceeds 250 KB.')
    dates = []
    for row in rows:
        price = row.get('price') if isinstance(row, dict) else None
        value = price.get('date') if isinstance(price, dict) else None
        dates.append(value if isinstance(value, str) else None)
    repeated = {value for value in dates if value is not None and dates.count(value) > 1}
    diagnoses = []
    for index, row in enumerate(rows):
        diagnosis = check_observation(row)
        if isinstance(row, dict):
            parts = [row.get('price'), row.get('earnings')]
            if dates[index] in repeated:
                diagnosis = {'status': 'unverified', 'message': 'Duplicate price date: all conflicting or repeated rows are withheld.'}
            elif any(not isinstance(part, dict) or part.get('securityId') != security for part in parts):
                diagnosis = {'status': 'unverified', 'message': 'Observation security differs from the frozen supplement identity.'}
            else:
                for part in parts:
                    source = part.get('source')
                    if not _source(source):
                        break  # check_observation already withholds malformed receipts.
                    content = originals.get(source['id'])
                    if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != source['sha256']:
                        diagnosis = {'status': 'unverified', 'message': 'Original bytes are missing or do not match the observation receipt.'}
                        break
        diagnoses.append({'index': index, **diagnosis})
    return {'version': 1, 'digest': hashlib.sha256(encoded).hexdigest(), 'securityId': security,
            'observations': diagnoses,
            'sourceReviewRequired': True, 'investorReviewRequired': True}
