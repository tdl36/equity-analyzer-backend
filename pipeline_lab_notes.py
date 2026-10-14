"""Draft-only version ledger for the pending Lab note workflow.

Not registered with production. No publication, thesis mutation, model invocation,
legacy research_notes writes or file export occurs here. Call in one transaction.
"""
import hashlib
import json
import re
import uuid


def schema(cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS pipeline_lab_note_versions (
        note_id UUID NOT NULL, revision INTEGER NOT NULL CHECK(revision>0),
        event_id UUID NOT NULL UNIQUE, ticker VARCHAR(20) NOT NULL,
        payload JSONB NOT NULL, payload_hash CHAR(64) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY(note_id,revision))''')


def normalized(value):
    if not isinstance(value,dict) or set(value)!={'title','markdown','sourcePack','qualityIssues'}:
        raise ValueError('A title, draft, frozen source pack and explicit quality issues are required.')
    title=value['title'];markdown=value['markdown'];sources=value['sourcePack'];issues=value['qualityIssues']
    if not isinstance(title,str) or not 1<=len(title.strip())<=240:raise ValueError('A bounded note title is required.')
    if not isinstance(markdown,str) or not markdown.strip() or len(markdown.encode('utf-8'))>2_000_000:raise ValueError('A nonempty draft up to 2 MB is required.')
    if not isinstance(issues,list) or len(issues)>100 or any(not isinstance(i,str) or not 1<=len(i)<=2000 for i in issues):
        raise ValueError('Bounded quality issues are required; missing review must remain explicit.')
    if not isinstance(sources,list) or not 1<=len(sources)<=8:raise ValueError('One to eight frozen original identities are required.')
    result=[]
    for s in sources:
        if not isinstance(s,dict) or set(s)!={'reference','filename','sha256','bytes'}:raise ValueError('Invalid original identity.')
        if (not isinstance(s['reference'],str) or not re.fullmatch(r'charlie:thesis_[0-9]+',s['reference'])
                or not isinstance(s['filename'],str) or not 1<=len(s['filename'])<=255
                or not isinstance(s['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',s['sha256'])
                or type(s['bytes']) is not int or not 0<s['bytes']<=15_000_000):raise ValueError('Invalid original identity, hash or size.')
        result.append(dict(s))
    if len({s['reference'] for s in result})!=len(result) or len({s['filename'] for s in result})!=len(result) or sum(s['bytes'] for s in result)>60_000_000:raise ValueError('Duplicate or oversized source pack.')
    return {'title':title.strip(),'markdown':markdown,'sourcePack':sorted(result,key=lambda s:s['reference']),
            'qualityIssues':issues,'status':'draft','investorApproved':False}


def obj(value):return json.loads(value) if isinstance(value,str) else value


def append(cur,note_id,ticker,event_id,expected_revision,payload):
    try:note_id=str(uuid.UUID(note_id));event_id=str(uuid.UUID(event_id))
    except (ValueError,TypeError,AttributeError):raise ValueError('Stable note and edit identities are required.')
    if not isinstance(ticker,str) or not re.fullmatch('[A-Z0-9][A-Z0-9.-]{0,19}',ticker):raise ValueError('Confirmed ticker required.')
    if type(expected_revision) is not int or expected_revision<0:raise ValueError('Expected draft revision required.')
    value=normalized(payload)
    signature={'noteId':note_id,'ticker':ticker,'expectedRevision':expected_revision,'payload':value}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    # Lock event first, then note, to serialize both replay and concurrent edits.
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('lab-note-event:'+event_id,))
    cur.execute('SELECT note_id,revision,payload_hash FROM pipeline_lab_note_versions WHERE event_id=%s',(event_id,));prior=cur.fetchone()
    if prior:
        if prior['payload_hash']!=digest:raise ValueError('Edit identity already used with different content.')
        return {'noteId':str(prior['note_id']),'revision':prior['revision'],'replayed':True,'status':'draft'}
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('lab-note:'+note_id,))
    cur.execute('SELECT revision,ticker,payload FROM pipeline_lab_note_versions WHERE note_id=%s ORDER BY revision DESC LIMIT 1',(note_id,));latest=cur.fetchone()
    revision=latest['revision'] if latest else 0
    if revision!=expected_revision:raise ValueError('A newer draft exists. Reload and compare before saving.')
    if latest and latest['ticker']!=ticker:raise ValueError('Note belongs to a different ticker.')
    if latest and obj(latest['payload'])['sourcePack']!=value['sourcePack']:
        raise ValueError('Editing cannot substitute the frozen source pack. Start a separate sourced draft.')
    cur.execute('INSERT INTO pipeline_lab_note_versions(note_id,revision,event_id,ticker,payload,payload_hash) VALUES(%s,%s,%s,%s,%s::jsonb,%s)',
                (note_id,revision+1,event_id,ticker,json.dumps(value),digest))
    return {'noteId':note_id,'revision':revision+1,'replayed':False,'status':'draft'}


def _note_identity(note_id):
    try:
        return str(uuid.UUID(note_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError('A valid note identity is required.')


def read(cur, note_id, revision=None):
    """Read one immutable draft; never fall back when a revision is missing."""
    note_id = _note_identity(note_id)
    if revision is not None and (type(revision) is not int or revision < 1):
        raise ValueError('A positive draft revision is required.')
    query = 'SELECT note_id,revision,ticker,payload,created_at FROM pipeline_lab_note_versions WHERE note_id=%s'
    params = [note_id]
    if revision is not None:
        query += ' AND revision=%s'
        params.append(revision)
    cur.execute(query + ' ORDER BY revision DESC LIMIT 1', params)
    row = cur.fetchone()
    if not row:
        return None
    return {'noteId': str(row['note_id']), 'revision': row['revision'],
            'ticker': row['ticker'], 'createdAt': row['created_at'].isoformat(),
            **obj(row['payload'])}


def history(cur, note_id, before=None, limit=20):
    """Bounded revision metadata only; bodies and source packs are fetched singly."""
    note_id = _note_identity(note_id)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('History page size must be between 1 and 50.')
    if before is not None and (type(before) is not int or before < 1):
        raise ValueError('A positive history cursor is required.')
    query = '''SELECT revision,ticker,created_at,payload->>'title' AS title
               FROM pipeline_lab_note_versions WHERE note_id=%s'''
    params = [note_id]
    if before is not None:
        query += ' AND revision<%s'
        params.append(before)
    params.append(limit + 1)
    cur.execute(query + ' ORDER BY revision DESC LIMIT %s', params)
    rows = cur.fetchall()
    items = [{'revision': r['revision'], 'ticker': r['ticker'], 'title': r['title'],
              'createdAt': r['created_at'].isoformat(), 'status': 'draft'}
             for r in rows[:limit]]
    return {'noteId': note_id, 'versions': items,
            'nextBefore': items[-1]['revision'] if len(rows) > limit else None}


def edit(cur, note_id, event_id, expected_revision, changes):
    """Edit prose only, retaining server-owned provenance and quality caveats.

    Read the specified base, not latest: a lost-response retry must reproduce its
    original signature even after subsequent revisions. append serializes writes
    and rejects a stale base unless this exact edit already committed.
    """
    if not isinstance(changes, dict) or set(changes) != {'title', 'markdown'}:
        raise ValueError('Only the title and draft text can be edited.')
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError('An existing draft revision is required.')
    base = read(cur, note_id, expected_revision)
    if base is None:
        raise ValueError('The requested draft revision does not exist.')
    payload = {**changes, 'sourcePack': base['sourcePack'],
               'qualityIssues': base['qualityIssues']}
    # Edits retain existing caveats; they never inherit a claim of fresh review.
    # Approval remains false on every revision regardless of quality issue count.
    return append(cur, note_id, base['ticker'], event_id, expected_revision, payload)
