"""Assemble source-backed inputs without inventing security mappings or valuations.

Internal pure adapter, not an endpoint. Records must come from trusted retained
storage; this function cannot establish provider provenance, fetch, save or approve.
"""
import hashlib
import json
import re
from historical_valuation import _date, _source
from historical_valuation_review import _baseline
from historical_valuation_originals import load_originals
from historical_valuation_evidence import annual_eps_before, daily_close_from_original


def assemble_candidates(selections, records, baseline, *, cik, symbol):
    frozen = _baseline(baseline)
    if type(cik) is not int or not 0 < cik < 10**10:
        raise ValueError('A verified issuer CIK is required.')
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,29}', symbol):
        raise ValueError('A verified provider symbol is required.')
    if not isinstance(selections, list) or not 1 <= len(selections) <= 30:
        raise ValueError('Select one to thirty exact historical dates.')
    refs, dates = [], set()
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != {'date', 'priceSource', 'earningsSource'}:
            raise ValueError('Each selection requires a date and two exact source receipts.')
        day = _date(selection['date']).isoformat()
        if day in dates:
            raise ValueError('Duplicate historical dates require explicit resolution.')
        dates.add(day)
        for field in ('priceSource', 'earningsSource'):
            receipt = selection[field]
            if not _source(receipt) or set(receipt) != {'id', 'sha256'}:
                raise ValueError('Select exact original receipts, not proposed financial values.')
            refs.append(receipt)
    originals, retrieval_dates = load_originals(refs, records, assignment_id=frozen['assignmentId'])
    kinds = {record['id']: record['kind'] for record in records}
    observations = []
    for selection in selections:
        row = {'date': selection['date'], 'status': 'incomplete', 'gaps': []}
        for part, field, kind in [('price', 'priceSource', 'daily_prices'),
                                  ('earnings', 'earningsSource', 'sec_companyfacts')]:
            receipt = selection[field]
            if receipt['id'] not in originals:
                row['gaps'].append({'input': part, 'code': 'missing_original'})
                continue
            if kinds[receipt['id']] != kind:
                row['gaps'].append({'input': part, 'code': 'wrong_original_kind'})
                continue
            raw = originals[receipt['id']]
            result = (daily_close_from_original(raw, receipt, symbol, selection['date'],
                                               retrieval_dates[receipt['id']]) if part == 'price'
                      else annual_eps_before(raw, receipt, cik, selection['date']))
            if result['status'] != 'extracted':
                row['gaps'].append({'input': part, 'code': 'unverified_original', 'message': result['message']})
            else:
                row[part] = result[part]
        if not row['gaps']:
            row['status'] = 'inputs_extracted'
        observations.append(row)
    payload = {'version': 1, 'baseline': frozen, 'cik': cik, 'symbol': symbol,
               'selections': selections, 'observations': observations,
               'valuationReady': False, 'investorReviewRequired': True,
               'unresolvedChecks': ['security_currency_mapping', 'corporate_actions',
                                    'filing_inventory_completeness', 'source_provenance', 'investor_review'],
               'retainedHashes': {key: hashlib.sha256(value).hexdigest()
                                  for key, value in sorted(originals.items())},
               'priceRetrievalDates': retrieval_dates}
    # Canonical serialization detaches caller-owned receipts and baseline, so
    # subsequent navigation or mutable inputs cannot alter an assembled version.
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded) > 500000:
        raise ValueError('Historical candidate pack exceeds the bound.')
    return {'candidateDigest': hashlib.sha256(encoded).hexdigest(), 'payload': json.loads(encoded)}


def replay_candidates(snapshot, records, baseline, *, cik, symbol):
    """Recheck a saved candidate pack; evidence/context changes need a new version."""
    if not isinstance(snapshot, dict) or set(snapshot) != {'candidateDigest', 'payload'}:
        raise ValueError('A complete candidate snapshot is required.')
    payload = snapshot.get('payload')
    if not isinstance(payload, dict):
        raise ValueError('A candidate payload is required.')
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded) > 500000 or hashlib.sha256(encoded).hexdigest() != snapshot['candidateDigest']:
        raise ValueError('Candidate snapshot content changed.')
    current = assemble_candidates(payload.get('selections'), records, baseline, cik=cik, symbol=symbol)
    if current != snapshot:
        raise ValueError('Historical evidence or baseline changed; assemble a new candidate version.')
    return current
