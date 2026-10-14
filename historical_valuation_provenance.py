"""Bounded acquisition metadata, never proof that a provider or source is trusted."""
import re
from datetime import datetime
from urllib.parse import urlsplit


def provenance(value):
    if value is None:
        return None  # Legacy receipt: do not invent acquisition evidence.
    fields = {'provider', 'observedUrl', 'retrievedAt', 'collectorVersion'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Invalid historical acquisition metadata.')
    for key in ('provider', 'collectorVersion'):
        if not isinstance(value[key], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', value[key]):
            raise ValueError('Invalid historical acquisition identity.')
    url = value['observedUrl']
    if not isinstance(url, str) or not 1 <= len(url) <= 2048 or any(ord(c) <= 32 for c in url):
        raise ValueError('Invalid historical observed URL.')
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == 'https' and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.query and not parsed.fragment
                 and parsed.port in (None, 443))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Historical observed URL must be HTTPS without credentials or query secrets.')
    stamp = value['retrievedAt']
    if not isinstance(stamp, str) or len(stamp) > 40:
        raise ValueError('Invalid historical acquisition time.')
    try:
        moment = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        if moment.tzinfo is None:
            raise ValueError()
    except ValueError:
        raise ValueError('Historical acquisition time requires a timezone.') from None
    return dict(value)


def public_provenance(value):
    """Expose only validated fields, suppress damaged metadata without hiding bytes."""
    try:
        metadata = provenance(value)
    except ValueError:
        return {'status': 'unavailable', 'verified': False}
    if metadata is None:
        return {'status': 'missing', 'verified': False}
    return {'status': 'recorded_unverified', 'verified': False, **metadata}


def acquisition_check(record, *, cik):
    """Check recorded claims against adapter policy; never certify their origin.

    Caller has already bound exact original bytes and assignment identity. No URL
    is opened. The digest binds all metadata, including fields not used by checks.
    """
    import hashlib
    import json
    from datetime import timezone
    metadata = provenance(record.get('provenance'))
    result = {'verified': False, 'status': 'missing', 'metadataDigest': None, 'gaps': []}
    if metadata is None:
        result['gaps'] = ['missing_acquisition_metadata']
        return result
    result['metadataDigest'] = hashlib.sha256(json.dumps(metadata, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    expected = {
        'sec_companyfacts': ('sec', f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json', 'historical-sec-v1'),
        'daily_prices': ('alphavantage', 'https://www.alphavantage.co/query', 'historical-prices-v1'),
    }.get(record.get('kind'))
    if expected is None:
        result['gaps'].append('unsupported_acquisition_kind')
    else:
        for field, value in zip(('provider', 'observedUrl', 'collectorVersion'), expected):
            if metadata[field] != value:
                result['gaps'].append({'provider':'provider_mismatch', 'observedUrl':'endpoint_or_issuer_mismatch',
                                       'collectorVersion':'collector_version_mismatch'}[field])
    day = datetime.fromisoformat(metadata['retrievedAt'].replace('Z', '+00:00')).astimezone(timezone.utc).date().isoformat()
    if day != record.get('retrievedDate'):
        result['gaps'].append('retrieval_date_mismatch')
    result['status'] = 'mismatch' if result['gaps'] else 'consistent_unverified'
    return result
