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
