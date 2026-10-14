"""Internal pending security-mapping evidence contract; no fetch, save or approval.

A caller must supply assignment-scoped records from trusted storage. This module
checks that proposed identifiers occur at exact top-level fields in retained JSON;
it does not establish provider authority, historical applicability or share basis.
No production endpoint accepts this contract yet. Ingestion remains disabled.
"""
from decimal import Decimal
import hashlib
import json
import re
from historical_valuation import _date, _source
from historical_valuation_review import _baseline
from historical_valuation_provenance import provenance


def _encode(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('Invalid mapping JSON.') from exc
    if len(raw) > 50000:
        raise ValueError('Mapping snapshot exceeds the bound.')
    return raw


def prepare_mapping(proposal, records, baseline, *, candidate_digest, cik, symbol):
    """Freeze a reviewable claim, exact source fields and target candidate identity.

    Price currency is a claim about the quoted security, not the reporting
    currency of an issuer. Even a matching field requires investor/provenance
    review before it could be used; this receipt never changes valuation readiness.
    """
    frozen = _baseline(baseline)
    if not isinstance(candidate_digest, str) or not re.fullmatch('[0-9a-f]{64}', candidate_digest):
        raise ValueError('An exact candidate digest is required.')
    if type(cik) is not int or not 0 < cik < 10**10:
        raise ValueError('A verified issuer CIK is required.')
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,29}', symbol):
        raise ValueError('A verified provider symbol is required.')
    if not isinstance(proposal, dict) or set(proposal) != {'source', 'fields', 'priceCurrency', 'asOf'}:
        raise ValueError('Mapping requires an exact source, field citations, currency and as-of date.')
    receipt = proposal['source']
    if not _source(receipt) or set(receipt) != {'id', 'sha256'}:
        raise ValueError('An exact mapping original receipt is required.')
    day = _date(proposal['asOf']).isoformat()
    currency = proposal['priceCurrency']
    if not isinstance(currency, str) or not re.fullmatch('[A-Z]{3}', currency):
        raise ValueError('A three-letter proposed currency is required.')
    fields = proposal['fields']
    if not isinstance(fields, dict) or set(fields) != {'cik', 'symbol', 'priceCurrency'}:
        raise ValueError('Cite each identifier and proposed price currency explicitly.')
    if any(not isinstance(v, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,79}', v) for v in fields.values()) or len(set(fields.values())) != 3:
        raise ValueError('Three distinct bounded top-level JSON field citations are required.')
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError('Exactly one mapping original must be loaded from trusted storage.')
    record = records[0]
    if not isinstance(record, dict) or record.get('assignmentId') != frozen['assignmentId'] or record.get('kind') != 'security_reference':
        raise ValueError('The security reference must belong to the frozen assignment.')
    content = record.get('content')
    if (not isinstance(content, bytes) or not 0 < len(content) <= 1000000 or
            record.get('id') != receipt['id'] or record.get('sha256') != receipt['sha256'] or
            hashlib.sha256(content).hexdigest() != receipt['sha256']):
        raise ValueError('Mapping original bytes differ from the selected receipt.')
    retrieved = _date(record.get('retrievedDate')).isoformat()
    if day > retrieved:
        raise ValueError('Mapping as-of date cannot be later than original retrieval.')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Ambiguous duplicate JSON field.')
            out[key] = value
        return out
    def reject(value):
        raise ValueError('Nonfinite original JSON.')
    try:
        data = json.loads(content, object_pairs_hook=unique, parse_constant=reject, parse_float=Decimal)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('Mapping original is not unambiguous JSON.') from exc
    if not isinstance(data, dict):
        raise ValueError('Mapping original must be a JSON object.')
    # Exact matching avoids ticker aliases, numeric coercions and inferred USD.
    if (type(data.get(fields['cik'])) is not int or data[fields['cik']] != cik or
            data.get(fields['symbol']) != symbol or data.get(fields['priceCurrency']) != currency):
        raise ValueError('Cited mapping values do not match the frozen issuer/security/currency claim.')
    payload = {'version': 1, 'status': 'pending', 'baseline': frozen,
               'candidateDigest': candidate_digest, 'cik': cik, 'symbol': symbol,
               'proposal': proposal, 'retrievedDate': retrieved,
               'acquisition': provenance(record.get('provenance')),
               'valuationReady': False, 'investorReviewRequired': True,
               'unresolvedChecks': ['price_currency_semantics', 'historical_applicability',
                   'share_class_mapping', 'corporate_actions', 'source_provenance', 'investor_review']}
    raw = _encode(payload)
    return {'mappingDigest': hashlib.sha256(raw).hexdigest(), 'payload': json.loads(raw)}


def replay_mapping(snapshot, records, baseline, *, candidate_digest, cik, symbol):
    if not isinstance(snapshot, dict) or set(snapshot) != {'mappingDigest', 'payload'}:
        raise ValueError('A complete pending mapping snapshot is required.')
    raw = _encode(snapshot['payload'])
    if hashlib.sha256(raw).hexdigest() != snapshot['mappingDigest']:
        raise ValueError('Mapping snapshot content changed.')
    if not isinstance(snapshot['payload'], dict):
        raise ValueError('A mapping payload is required.')
    current = prepare_mapping(snapshot['payload'].get('proposal'), records, baseline,
                              candidate_digest=candidate_digest, cik=cik, symbol=symbol)
    if snapshot != current:
        raise ValueError('Candidate, evidence or baseline changed; prepare a new mapping version.')
    return current
