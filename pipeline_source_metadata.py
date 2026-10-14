"""Public inventory fields only; never return original bodies or arbitrary metadata."""
import json
import re
from datetime import date


def fields(value):
    try:meta=json.loads(value) if isinstance(value,str) else value
    except ValueError:meta={}
    if not isinstance(meta,dict):meta={}
    published=meta.get('published') or meta.get('publishedDate')
    try:published=date.fromisoformat(published).isoformat() if isinstance(published,str) else None
    except ValueError:published=None
    publisher=meta.get('publisher')
    return {'publishedDate':published,'publisher':publisher[:200] if isinstance(publisher,str) else '',
            'originalReceiptRecorded':meta.get('source')=='alphasense_collection' and isinstance(meta.get('sha256'),str) and bool(re.fullmatch('[0-9a-f]{64}',meta['sha256']))}
