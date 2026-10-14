"""Calendar anchors from frozen financials, not inferred trading sessions.

Year ends are requests for exact-date coverage only. Weekends/holidays stay gaps;
no previous-session substitution, current date, fresh quote or future filing is
introduced. This policy never supplies earnings amounts from restated annuals.
"""
import hashlib
import json
from historical_valuation import _date
from historical_valuation_review import _baseline
from onepager_corrections import digest


def frozen_dates(fundamentals, baseline, *, assignment_input):
    baseline = _baseline(baseline)
    if (not isinstance(fundamentals, dict) or not isinstance(assignment_input, dict)
            or digest({'input': assignment_input, 'fundamentals': fundamentals,
                       'vintage': fundamentals}) != baseline['fundamentalsDigest']):
        raise ValueError('Frozen financials do not match the edition baseline.')
    cutoff = _date(fundamentals.get('asOf'))
    annuals = fundamentals.get('annuals')
    if not isinstance(annuals, list) or len(annuals) > 30:
        raise ValueError('A bounded frozen annual history is required.')
    anchors, gaps = {}, []
    for index, row in enumerate(annuals):
        try:
            if not isinstance(row, dict):
                raise ValueError('Invalid period')
            start, end = _date(row.get('start')), _date(row.get('end'))
            if not 330 <= (end-start).days <= 380 or end >= cutoff:
                raise ValueError('Period not completed before snapshot')
            day = end.isoformat()
            if day in anchors:
                raise ValueError('Duplicate annual end')
            anchors[day] = {'date': day, 'basis': 'fiscal_period_end',
                            'periodStart': start.isoformat(), 'periodEnd': day}
        except (ValueError, TypeError):
            # Never quietly form a complete-looking series from malformed history.
            gaps.append({'index': index, 'code': 'invalid_or_conflicting_period'})
    if gaps:
        anchors = {}
    if not annuals:
        gaps.append({'code': 'no_frozen_annual_periods'})
    payload = {'version': 1, 'policy': 'exact_fiscal_period_end_v1',
               'baseline': baseline, 'asOf': cutoff.isoformat(),
               'dates': sorted(anchors), 'anchors': [anchors[d] for d in sorted(anchors)],
               'gaps': gaps, 'status': 'dates_available' if anchors else 'needs_date_review',
               'tradingDatesVerified': False, 'valuationReady': False,
               'caveat': 'Fiscal period ends are calendar anchors, not verified trading days. Missing exact-date prices remain gaps; no adjacent session is substituted.'}
    encoded = json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {'datePolicyDigest': hashlib.sha256(encoded).hexdigest(), 'payload': json.loads(encoded)}
