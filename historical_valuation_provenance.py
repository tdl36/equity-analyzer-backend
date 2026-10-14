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
