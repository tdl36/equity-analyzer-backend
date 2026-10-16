"""Bounded metadata inventory and frozen source roles for processing proposals."""
import re
from company_research import eligible
from pipeline_source_metadata import fields


def supplemental_ids(value, collected_count):
    if (not isinstance(value, list) or len(value) + collected_count > 1000
            or any(not isinstance(i, str) or not re.fullmatch(r'[1-9][0-9]{0,18}', i) for i in value)
            or len(set(value)) != len(value)):
        raise ValueError('Choose distinct stored original identities; at most 1000 originals per selected pack.')
    return sorted(value)


def inventory(cur, ticker, collected_names, after=None):
    if after is not None and (not isinstance(after, str) or not re.fullmatch(r'[1-9][0-9]{0,18}', after) or int(after) > 9223372036854775807):
        raise ValueError('Invalid source-library cursor. Reload the library.')
    cur.execute("SELECT id,filename,metadata FROM document_files WHERE ticker=%s AND NOT(filename=ANY(%s)) AND (%s::bigint IS NULL OR id<%s::bigint) ORDER BY id DESC LIMIT 201", (ticker, collected_names, after, after))
    rows = list(cur.fetchall())
    return {'documents': [dict(id='thesis_' + str(r['id']), filename=r['filename'],
                               **fields(r['metadata']), eligible=eligible(r['metadata'] or {})) for r in rows[:200]],
            'nextCursor': str(rows[199]['id']) if len(rows) > 200 else None,
            'scope': 'Stored originals for this company, excluding this collection. Metadata is not byte verification or evidence of thesis incorporation.'}


def context(cur, ticker, sources, collected_names, baseline_package):
    cur.execute('SELECT id::text AS id,metadata FROM document_files WHERE ticker=%s AND id::text=ANY(%s)', (ticker, [s['documentId'] for s in sources]))
    metadata = {r['id']: fields(r['metadata']) for r in cur.fetchall()}
    referenced = {r.get('sha256') or r.get('hash') for r in baseline_package.get('sourceRegister', []) if isinstance(r, dict)} - {None, ''}
    return {s['documentId']: {'role': 'collection' if s['filename'] in collected_names else 'supplemental',
                            'publishedDate': metadata.get(s['documentId'], {}).get('publishedDate'),
                            'publisher': metadata.get(s['documentId'], {}).get('publisher', ''),
                            'sourceKind': metadata.get(s['documentId'], {}).get('sourceKind', ''),
                            'referencedBySavedThesis': s['sha256'] in referenced} for s in sources}


def collected_inventory(cur, ticker, receipts):
    """Metadata for the frozen collection receipt, not a fresh byte-verification claim."""
    if (not isinstance(receipts, list) or not 1 <= len(receipts) <= 1000
            or any(not isinstance(r, dict) or not isinstance(r.get('filename'), str)
                   or not r['filename'] or not isinstance(r.get('sha256'), str)
                   or not re.fullmatch('[0-9a-f]{64}', r['sha256']) for r in receipts)
            or len({r['filename'] for r in receipts}) != len(receipts)):
        raise ValueError('Collection source receipts are missing or ambiguous. Inspect the saved collection.')
    names = [r['filename'] for r in receipts]
    cur.execute('SELECT id,filename,metadata FROM document_files WHERE ticker=%s AND filename=ANY(%s) ORDER BY id LIMIT 2001', (ticker, names))
    rows = list(cur.fetchall())
    if len(rows) > 2000:
        raise ValueError('Stored source identities are ambiguous. Inspect the collection before selecting evidence.')
    by_name = {}
    for row in rows:
        by_name.setdefault(row['filename'], []).append(row)
    documents = []
    for receipt in receipts:
        matches = by_name.get(receipt['filename'], [])
        row = matches[0] if len(matches) == 1 else None
        state = 'missing' if not matches else 'ambiguous' if row is None else 'available' if eligible(row['metadata'] or {}) else 'restricted'
        documents.append(dict(id='thesis_' + str(row['id']) if row else None,
                              filename=receipt['filename'], receiptSha256=receipt['sha256'],
                              availability=state, **fields(row['metadata'] if row else {})))
    return documents


def selected_inventory(cur, ticker, collected_names, selected):
    """Exact metadata recovery, bounded separately from browsing; no original bodies."""
    ids = selected.split(',') if isinstance(selected, str) else []
    if not 1 <= len(ids) <= 100:
        raise ValueError('Choose 1–100 stored identities per metadata read.')
    ids = supplemental_ids(ids, 0)
    if any(int(value) > 9223372036854775807 for value in ids):
        raise ValueError('Invalid stored source identity.')
    cur.execute('SELECT id,filename,metadata FROM document_files WHERE ticker=%s AND id=ANY(%s) AND NOT(filename=ANY(%s)) ORDER BY id DESC', (ticker, [int(value) for value in ids], collected_names))
    rows = list(cur.fetchall())
    found = {str(r['id']) for r in rows}
    return {'selectedIds': ids,
            'documents': [dict(id='thesis_' + str(r['id']), filename=r['filename'],
                               **fields(r['metadata']), eligible=eligible(r['metadata'] or {})) for r in rows],
            'missingIds': [value for value in ids if value not in found]}
