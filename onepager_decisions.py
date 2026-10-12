"""Internal transactional decision store; no public route or automatic caller.

The future HTTP boundary must authenticate an investor and obtain investor_id
from trusted server authentication, never the request body. Tests use disposable
synthetic records only. Original editions/research remain immutable.
"""
import json
from onepager_corrections import digest
from onepager_edition_contract import _uuid
from onepager_decision_contract import validate_decision


def record_decision(get_db, aid, eid, body, *, investor_id):
    if not isinstance(investor_id, str) or not investor_id.strip() or len(investor_id) > 200:
        raise ValueError('An authenticated investor identity is required.')
    aid, eid = _uuid(aid), _uuid(eid)
    if not isinstance(body, dict) or len(json.dumps(body)) > 12000:
        raise ValueError('Invalid or oversized decision request.')
    ident = _uuid(body.get('requestId'))
    if body.get('assignmentId') != aid or body.get('editionId') != eid:
        raise ValueError('The decision belongs to a different assignment or edition.')
    signature = digest(body)
    with get_db(commit=True) as (_, cur):
        # One lock serializes all decisions for this assignment, including when
        # no accepted head exists yet. Lock before schema/evidence consistently.
        cur.execute("SELECT id,ticker,status,input,result FROM mp_jobs WHERE id=%s AND stage='research_assignment' FOR UPDATE", (aid,))
        assignment = cur.fetchone()
        if not assignment:
            raise ValueError('Saved assignment not found.')
        assignment = dict(assignment)
        cur.execute("SELECT to_regclass('onepager_editions') AS name")
        if not cur.fetchone()['name']:
            raise ValueError('Saved edition not found.')
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('onepager-decisions-schema'))")
        cur.execute('''CREATE TABLE IF NOT EXISTS onepager_decisions (
            id UUID PRIMARY KEY, assignment_id TEXT NOT NULL,
            edition_id UUID NOT NULL UNIQUE, investor_id TEXT NOT NULL,
            payload_hash TEXT NOT NULL, record JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
            CREATE TABLE IF NOT EXISTS onepager_accepted_heads (
            assignment_id TEXT PRIMARY KEY, edition_id UUID NOT NULL,
            decision_id UUID NOT NULL UNIQUE)''')
        cur.execute('SELECT assignment_id,edition_id,investor_id,payload_hash,record FROM onepager_decisions WHERE id=%s', (ident,))
        previous = cur.fetchone()
        if previous:
            if (previous['assignment_id'] != aid or str(previous['edition_id']) != eid or
                    previous['investor_id'] != investor_id or previous['payload_hash'] != signature):
                raise ValueError('This decision identity belongs to a different request or investor.')
            # Acknowledgement of an already committed decision, not a fresh
            # approval. Remains safe after evidence drift or a later decision.
            return {'decision': previous['record'], 'reused': True}
        cur.execute('SELECT id FROM onepager_decisions WHERE edition_id=%s', (eid,))
        if cur.fetchone():
            raise ValueError('This edition already has a decision. Inspect its saved review.')
        cur.execute('SELECT run_id,record FROM onepager_editions WHERE id=%s AND assignment_id=%s FOR SHARE', (eid, aid))
        saved = cur.fetchone()
        if not saved or str(saved['run_id']) != str((assignment.get('result') or {}).get('reportId')):
            raise ValueError('Saved edition does not belong to the current assignment report.')
        record = saved['record']
        if not isinstance(record, dict) or record.get('id') != eid:
            raise ValueError('Saved edition identity changed.')
        cur.execute('SELECT edition_id FROM onepager_accepted_heads WHERE assignment_id=%s FOR UPDATE', (aid,))
        head = cur.fetchone()
        parent = str(head['edition_id']) if head else None
        cur.execute('SELECT id,ticker,status,input,baseline,sources,state FROM stock_analysis_runs WHERE id=%s FOR SHARE', (saved['run_id'],))
        row = cur.fetchone()
        if not row:
            raise ValueError('Saved report not found.')
        run = dict(row)
        ids = record.get('correctionIds')
        if not isinstance(ids, list) or not 1 <= len(ids) <= 20:
            raise ValueError('Invalid saved correction selection.')
        ids = [_uuid(value) for value in ids]
        cur.execute('SELECT id,run_id,fingerprint,payload_hash,proposal FROM onepager_corrections WHERE id=ANY(%s::uuid[]) FOR SHARE', (ids,))
        corrections = [dict(row) for row in cur.fetchall()]
        documents = {}
        for filename in sorted({source['filename'] for source in run.get('sources', [])}):
            cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE', (run['ticker'], filename))
            rows = cur.fetchall()
            if len(rows) != 1:
                raise ValueError('A frozen original is missing or ambiguous.')
            documents[filename] = dict(rows[0])
        intent = validate_decision(run, assignment, record, body, corrections, documents,
                                   assignment['result'].get('visualFundamentals'), parent_id=parent)
        cur.execute('SELECT clock_timestamp() AS decided_at')
        decision = {**intent, 'investorId': investor_id,
                    'decidedAt': cur.fetchone()['decided_at'].isoformat(),
                    'reviewOrigin': 'investor',
                    'caveat': 'Investor editorial decision only; original research and thesis remain unchanged.'}
        cur.execute('INSERT INTO onepager_decisions(id,assignment_id,edition_id,investor_id,payload_hash,record) VALUES(%s,%s,%s,%s,%s,%s::jsonb)',
                    (ident, aid, eid, investor_id, signature, json.dumps(decision)))
        if intent['action'] == 'accept':
            cur.execute('''INSERT INTO onepager_accepted_heads(assignment_id,edition_id,decision_id)
                VALUES(%s,%s,%s) ON CONFLICT(assignment_id) DO UPDATE
                SET edition_id=EXCLUDED.edition_id,decision_id=EXCLUDED.decision_id''', (aid, eid, ident))
        return {'decision': decision, 'reused': False}
