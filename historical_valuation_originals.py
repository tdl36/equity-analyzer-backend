"""Bind historical diagnostics to server-loaded, assignment-scoped originals.

Not an upload endpoint or a provenance certification. Callers must obtain these
records from locked trusted storage, never deserialize them from request bodies.
No URLs or filesystem paths supplied by a proposal are opened.
"""
import hashlib
from historical_valuation import _date, _source


def load_originals(references, records, *, assignment_id):
    """Return exact referenced bytes and trusted market-data retrieval dates.

    Missing originals remain missing for the pending diagnostic. A present but
    conflicting receipt is an error, never permission to use another version.
    The trusted record contract includes assignmentId, id, sha256, content,
    kind (sec_companyfacts or daily_prices), and retrievedDate.
    """
    if not isinstance(assignment_id, str) or not assignment_id:
        raise ValueError('Assignment scope is required.')
    if not isinstance(references, list) or not 1 <= len(references) <= 60:
        raise ValueError('One to sixty original references are required.')
    wanted = {}
    for receipt in references:
        if not _source(receipt):
            raise ValueError('Invalid original reference.')
        ident = receipt['id']
        if ident in wanted and wanted[ident] != receipt['sha256']:
            raise ValueError('Conflicting versions of one original were requested.')
        wanted[ident] = receipt['sha256']
    if not isinstance(records, list) or len(records) > 60:
        raise ValueError('Original inventory exceeds the bound.')
    originals, dates, seen = {}, {}, set()
    total = 0
    for record in records:
        if not isinstance(record, dict) or not _source(record):
            raise ValueError('Invalid stored original receipt.')
        ident = record['id']
        if ident not in wanted:
            raise ValueError('Storage returned an unrequested original.')
        if ident in seen:
            raise ValueError('Storage returned ambiguous original versions.')
        seen.add(ident)
        if record.get('assignmentId') != assignment_id:
            raise ValueError('Original belongs to a different assignment.')
        raw = record.get('content')
        if not isinstance(raw, bytes) or not 0 < len(raw) <= 15000000:
            raise ValueError('Stored original is empty, unavailable or oversized.')
        total += len(raw)
        if total > 60000000:
            raise ValueError('Combined historical originals exceed the bound.')
        if (record['sha256'] != wanted[ident]
                or hashlib.sha256(raw).hexdigest() != wanted[ident]):
            raise ValueError('Original bytes differ from the frozen receipt.')
        if record.get('kind') not in ('sec_companyfacts', 'daily_prices'):
            raise ValueError('Unsupported historical original kind.')
        retrieved = record.get('retrievedDate')
        _date(retrieved)
        originals[ident] = raw
        if record['kind'] == 'daily_prices':
            dates[ident] = retrieved
    return originals, dates
