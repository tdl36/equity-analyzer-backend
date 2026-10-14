"""Bounded coordinator checkpoint of retained inputs; never permission to dispatch.

The coordinator supplies frozen dates and trusted storage records. Every date is
planned against the same complete inventory and baseline. Replay rejects even a
new ambiguous receipt instead of silently refreshing a previously selected pack.
"""
import hashlib
import json
from historical_valuation_selection import plan_selection


def validate_dates(dates):
    if (not isinstance(dates, list) or not 1 <= len(dates) <= 30
            or any(not isinstance(day, str) for day in dates)
            or len(set(dates)) != len(dates)):
        raise ValueError('Supply one to thirty distinct frozen valuation dates.')
    from historical_valuation import _date
    for day in dates:
        _date(day)
    return sorted(dates)


def readiness(records, baseline, *, cik, symbol, dates):
    dates = validate_dates(dates)
    plans = [plan_selection(records, baseline, cik=cik, symbol=symbol, price_date=day)
             for day in sorted(dates)]
    observations = []
    for plan in plans:
        p = plan['payload']
        row = {'date': p['date'], 'planDigest': plan['planDigest'],
               'status': p['status'], 'gaps': p['gaps']}
        if p['status'] == 'selected':
            candidate = p['candidate']
            row.update(candidateDigest=candidate['candidateDigest'],
                       selection=candidate['payload']['selections'][0])
        observations.append(row)
    # Input coverage is separate from valuation validity and investor approval.
    payload = {'version': 1, 'baseline': plans[0]['payload']['baseline'],
               'cik': cik, 'symbol': symbol, 'dates': sorted(dates),
               'status': 'inputs_selected' if all(r['status'] == 'selected' for r in observations)
                         else 'needs_evidence_review',
               'observations': observations, 'valuationReady': False,
               'dispatchAllowed': False, 'investorReviewRequired': True,
               'unresolvedChecks': ['security_currency_mapping', 'corporate_actions',
                                    'filing_inventory_completeness', 'source_provenance', 'investor_review']}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded) > 500000:
        raise ValueError('Historical readiness checkpoint exceeds the bound.')
    return {'readinessDigest': hashlib.sha256(encoded).hexdigest(), 'payload': json.loads(encoded)}


def replay(snapshot, records, baseline, *, cik, symbol):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('payload'), dict):
        raise ValueError('A complete historical readiness checkpoint is required.')
    current = readiness(records, baseline, cik=cik, symbol=symbol,
                        dates=snapshot['payload'].get('dates'))
    if current != snapshot:
        raise ValueError('Historical evidence or target changed; prepare a new checkpoint.')
    return current
