"""Immutable historical originals for trusted ingestion, never proposal uploads.

Caller authenticates ingestion, verifies provider identity, and holds the assignment
lock. This module preserves receipts and exact bytes; it does not certify provenance,
fetch URLs, approve valuation, commit transactions or create a public endpoint.
"""
import json
from historical_valuation_provenance import provenance
from historical_valuation_originals import load_originals
from historical_valuation_store import _identity


def save_original(cur, record):
    """Retain one verified ingestion record; conflicting identities cannot overwrite."""
    metadata = provenance(record.get('provenance'))
    aid = _identity(record.get('assignmentId'))
    reference = {k: record.get(k) for k in ('id', 'sha256')}
    load_originals([reference], [record], assignment_id=aid)
    cur.execute("SELECT pg_advisory_xact_lock(hashtext('historical-originals-schema'))")
    cur.execute('''CREATE TABLE IF NOT EXISTS historical_valuation_originals (
        assignment_id UUID NOT NULL, source_id TEXT NOT NULL, sha256 TEXT NOT NULL,
        kind TEXT NOT NULL, retrieved_date DATE NOT NULL, content BYTEA NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY(assignment_id,source_id))''')
    cur.execute('ALTER TABLE historical_valuation_originals ADD COLUMN IF NOT EXISTS provenance JSONB')
    cur.execute('''INSERT INTO historical_valuation_originals
        (assignment_id,source_id,sha256,kind,retrieved_date,content,provenance)
        VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING RETURNING source_id''',
        (aid, record['id'], record['sha256'], record['kind'],
         record['retrievedDate'], record['content'], json.dumps(metadata)))
    inserted = cur.fetchone() is not None
    cur.execute('''SELECT sha256,kind,retrieved_date,content,provenance
        FROM historical_valuation_originals WHERE assignment_id=%s AND source_id=%s FOR SHARE''',
        (aid, record['id']))
    old = cur.fetchone()
    if (old['sha256'] != record['sha256'] or old['kind'] != record['kind']
            or str(old['retrieved_date']) != record['retrievedDate']
            or bytes(old['content']) != record['content']
            or old['provenance'] != metadata):
        raise ValueError('Original identity already belongs to different historical evidence.')
    return {'id': record['id'], 'sha256': record['sha256'], 'reused': not inserted}


def read_records(cur, references, *, assignment_id):
    """Lock only requested rows; missing evidence remains missing, with no DDL.

    SQL checks byte lengths before transferring bodies to bound memory even if
    storage is damaged. The existing adapter then rechecks every exact hash.
    """
    aid = _identity(assignment_id)
    load_originals(references, [], assignment_id=aid)
    cur.execute("SELECT to_regclass('historical_valuation_originals') AS name")
    if not cur.fetchone()['name']:
        return []
    cur.execute("""SELECT EXISTS(SELECT 1 FROM pg_attribute
        WHERE attrelid='historical_valuation_originals'::regclass
        AND attname='provenance' AND NOT attisdropped) AS present""")
    metadata_column = 'provenance' if cur.fetchone()['present'] else 'NULL::jsonb'
    identifiers = sorted({r['id'] for r in references})
    cur.execute('''SELECT source_id,octet_length(content) AS size
        FROM historical_valuation_originals
        WHERE assignment_id=%s AND source_id=ANY(%s) ORDER BY source_id FOR SHARE''',
        (aid, identifiers))
    sizes = cur.fetchall()
    if (any(not 0 < r['size'] <= 15000000 for r in sizes)
            or sum(r['size'] for r in sizes) > 60000000):
        raise ValueError('Stored historical originals exceed the byte bound.')
    # Query the locked inventory, not newly appearing rows under READ COMMITTED.
    cur.execute(f'''SELECT source_id,sha256,kind,retrieved_date,content,
        {metadata_column} AS provenance
        FROM historical_valuation_originals
        WHERE assignment_id=%s AND source_id=ANY(%s) ORDER BY source_id FOR SHARE''',
        (aid, [r['source_id'] for r in sizes]))
    records = [{'assignmentId': aid, 'id': r['source_id'], 'sha256': r['sha256'],
                'kind': r['kind'], 'retrievedDate': str(r['retrieved_date']),
                'content': bytes(r['content']), 'provenance': provenance(r['provenance']),
                'provenanceVerified': False} for r in cur.fetchall()]
    load_originals(references, records, assignment_id=aid)
    return records


def read_originals(cur, references, *, assignment_id):
    records = read_records(cur, references, assignment_id=assignment_id)
    return load_originals(references, records, assignment_id=assignment_id)
