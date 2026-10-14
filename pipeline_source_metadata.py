"""Public inventory fields only; never return original bodies or arbitrary metadata."""
import json
import re
from datetime import date
from urllib.parse import urlsplit, parse_qsl


def observed_url(value):
    """Only the collector's public research permalink, never sign-in or credential URLs."""
    if not isinstance(value,str) or len(value)>2000 or any(ord(c)<33 for c in value):return ''
    try:
        u=urlsplit(value)
        if (u.scheme!='https' or u.hostname!='research.alpha-sense.com' or u.username or u.password
                or u.port not in (None,443) or u.fragment
                or re.search(r'login|authenticate|authorize',u.path,re.I)
                or any(re.search(r'token|password|secret|code|state|key|signature|credential',k,re.I) for k,_ in parse_qsl(u.query))):return ''
        return value
    except ValueError:return ''


def fields(value):
    try:meta=json.loads(value) if isinstance(value,str) else value
    except ValueError:meta={}
    if not isinstance(meta,dict):meta={}
    published=meta.get('published') or meta.get('publishedDate')
    try:published=date.fromisoformat(published).isoformat() if isinstance(published,str) else None
    except ValueError:published=None
    publisher=meta.get('publisher')
    receipt=meta.get('source')=='alphasense_collection' and isinstance(meta.get('sha256'),str) and bool(re.fullmatch('[0-9a-f]{64}',meta['sha256']))
    return {'sourceUrl':observed_url(meta.get('sourceUrl')) if receipt else '',
            'collectionRun':meta.get('collectionRun') if receipt and isinstance(meta.get('collectionRun'),str) and re.fullmatch('[0-9a-f]{12}',meta['collectionRun']) else '',
            'recordedSha256':meta.get('sha256') if receipt else '',
            'publishedDate':published,'publisher':publisher[:200] if isinstance(publisher,str) else '',
            'originalReceiptRecorded':receipt}
