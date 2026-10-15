"""Durable, immutable collection-to-processing proposals; never a second worker."""
import hashlib
import json
import uuid
from flask import jsonify, request
from research_assignment_plan import plan
from pipeline_lab import check_originals

STAGE = 'processing_proposal'


def obj(value):
    return json.loads(value) if isinstance(value, str) else value or {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assignment_id(ident):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'charlie:processing-proposal:' + ident))


def original_pack(cur, ticker, filenames):
    cur.execute("WITH selected AS MATERIALIZED (SELECT id,filename,metadata,file_data FROM document_files WHERE ticker=%s AND filename=ANY(%s) FOR SHARE) SELECT id::text AS id,filename,metadata,CASE WHEN octet_length(file_data)<=20000000 AND sum(octet_length(file_data)) OVER ()<=80000000 THEN file_data ELSE NULL END AS file_data FROM selected", (ticker, filenames))
    documents = {'charlie:thesis_' + str(r['id']): r for r in cur.fetchall()}
    if len(documents) != len(filenames):
        raise ValueError('Selected original identities are missing or ambiguous. Refresh the source library.')
    checked = check_originals(ticker, 'processing', 1, sorted(documents), documents)
    if checked['blockers']:
        raise ValueError(' '.join(checked['blockers']))
    return [dict(r, documentId=r['reference'][len('charlie:thesis_'):]) for r in checked['verified']]


def dispatch(cur, proposed_plan, ident):
    """Called inside the existing assignment insertion transaction; all or nothing."""
    ref = proposed_plan['processingProposal']
    cur.execute('SELECT * FROM mp_jobs WHERE id=%s AND stage=%s FOR UPDATE', (ref['id'], STAGE))
    row = cur.fetchone()
    if not row or row['status'] != 'pending':
        raise ValueError('Processing proposal is unavailable or already decided. Reload its saved job.')
    frozen = obj(row['input'])
    if ref != {'id': row['id'], 'revision': 1, 'digest': frozen['digest']} or ident != assignment_id(row['id']):
        raise ValueError('Processing proposal revision or request identity changed.')
    if plan({k: v for k, v in proposed_plan.items() if k != 'processingProposal'}) != frozen['plan']:
        raise ValueError('Processing choices changed. Save a new proposal before approving.')
    if proposed_plan['thesisBaseline'].get('baseline') != frozen['baseline'] or proposed_plan['caseRevision'] != frozen['caseRevision']:
        raise ValueError('The thesis or research baseline changed. Save a fresh processing proposal.')
    from thesis_imports import current, row_hash
    if row_hash(current(cur, row['ticker'])) != frozen['baseline']['hash']:
        raise ValueError('The accepted thesis changed. Save a fresh processing proposal.')
    pack = original_pack(cur, row['ticker'], [s['filename'] for s in frozen['sources']])
    if pack != frozen['sources']:
        raise ValueError('Selected originals changed. Save a fresh proposal before processing.')
    cur.execute("UPDATE mp_jobs SET status='dispatched',result=%s::jsonb,updated_at=NOW() WHERE id=%s", (json.dumps({'assignmentId': ident}), row['id']))
    return pack


def public(row):
    frozen = obj(row['input'])
    return dict(id=row['id'], ticker=row['ticker'], status=row['status'], revision=1,
                digest=frozen['digest'], collectionId=frozen['collectionId'], plan=frozen['plan'],
                sources=frozen['sources'], baseline=frozen['baseline'], caseRevision=frozen['caseRevision'],
                assignmentId=obj(row['result']).get('assignmentId'), created_at=row['created_at'])


def register(co):
    bp = co.bp

    @bp.route('/api/research/assignments/<collection_id>/processing-proposals', methods=['GET', 'POST'])
    def processing_proposals(collection_id):
        parent = co.read(collection_id)
        if not parent or parent['status'] != 'complete' or not obj(parent['input']).get('collectionOnly'):
            return jsonify(error='Choose a completed source-only collection.'), 409
        if request.method == 'GET':
            with co.db() as (_, cur):
                cur.execute("SELECT * FROM mp_jobs WHERE stage=%s AND input->>'collectionId'=%s ORDER BY created_at DESC LIMIT 50", (STAGE, collection_id))
                rows = [public(r) for r in cur.fetchall()]
            response = jsonify(proposals=rows)
            response.headers['Cache-Control'] = 'private, no-store'
            return response
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or set(data) != {'requestId', 'filenames', 'plan'}:
                raise ValueError('Choose originals, purpose and outputs before saving a proposal.')
            ident = str(uuid.UUID(data['requestId']))
            filenames = data['filenames']
            if not isinstance(filenames, list) or not 1 <= len(filenames) <= 1000 or any(not isinstance(n, str) for n in filenames) or len(set(filenames)) != len(filenames):
                raise ValueError('Choose distinct originals (1–1000) from this collection.')
            choices = data['plan']
            allowed = {'researchPurpose', 'outputs', 'horizon', 'instruction', 'meeting', 'visualFundamentals'}
            if not isinstance(choices, dict) or set(choices) - allowed or 'researchPurpose' not in choices:
                raise ValueError('An explicit thesis-review or event-note purpose is required.')
            parent_plan = obj(parent['input'])
            normalized = plan({**choices, **{k: parent_plan[k] for k in ('ticker', 'since', 'until')}})
            intent = {'collectionId': collection_id, 'plan': normalized, 'filenames': sorted(filenames)}
            intent_hash = digest(intent)
            # Exact replay returns the frozen proposal even if evidence has since changed.
            with co.db() as (_, cur):
                cur.execute('SELECT * FROM mp_jobs WHERE id=%s', (ident,))
                old = cur.fetchone()
            if old:
                if old['stage'] != STAGE or obj(old['input']).get('intentHash') != intent_hash:
                    raise ValueError('Proposal identity already used for different choices.')
                return jsonify(public(old)), 200
            baseline = co.call('thesis_imports.prepare', ticker=parent['ticker'])['package']['baseline']
            revision = co.call('stock_analysis.listing', ticker=parent['ticker'])['baseline']['revision']
            receipts = obj(parent['result']).get('sources', [])
            hashes = {s['filename']: s['sha256'] for s in receipts}
            if len(hashes) != len(receipts) or any(n not in hashes for n in filenames):
                raise ValueError('Select originals from this collection receipt only.')
            with co.db(True) as (_, cur):
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('assignment-submit:' + ident,))
                cur.execute('SELECT * FROM mp_jobs WHERE id=%s', (ident,))
                old = cur.fetchone()
                if old:
                    if old['stage'] != STAGE or obj(old['input']).get('intentHash') != intent_hash:
                        raise ValueError('Proposal identity already used for different choices.')
                    return jsonify(public(old)), 200
                sources = original_pack(cur, parent['ticker'], filenames)
                if any(s['sha256'] != hashes[s['filename']] for s in sources):
                    raise ValueError('Original bytes no longer match the completed collection receipt.')
                frozen = dict(collectionId=collection_id, plan=normalized, sources=sources,
                              baseline=baseline, caseRevision=revision, intentHash=intent_hash)
                frozen['digest'] = digest(frozen)
                cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result) VALUES(%s,%s,%s,'pending',%s::jsonb,'{}'::jsonb) RETURNING *", (ident, STAGE, parent['ticker'], json.dumps(frozen)))
                saved = public(cur.fetchone())
            return jsonify(saved), 201
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return jsonify(error=str(exc) if isinstance(exc, ValueError) else 'Invalid processing proposal.'), 409

    @bp.post('/api/research/processing-proposals/<ident>/<action>')
    def decide_processing_proposal(ident, action):
        if action not in ('approve', 'cancel'):
            return jsonify(error='Unknown processing action.'), 400
        with co.db() as (_, cur):
            cur.execute('SELECT * FROM mp_jobs WHERE id=%s AND stage=%s', (ident, STAGE))
            row = cur.fetchone()
        if not row:
            return jsonify(error='Processing proposal not found.'), 404
        frozen = obj(row['input'])
        if request.get_json(silent=True) != {'revision': 1, 'digest': frozen['digest']}:
            return jsonify(error='Reload the saved proposal before deciding.'), 409
        if action == 'cancel':
            with co.db(True) as (_, cur):
                cur.execute("UPDATE mp_jobs SET status='cancelled',updated_at=NOW() WHERE id=%s AND stage=%s AND status='pending' RETURNING id", (ident, STAGE))
                cancelled = cur.fetchone()
            if not cancelled:
                return jsonify(error='Proposal already decided. Inspect its saved job.'), 409
            return jsonify(cancelled=True)
        if row['status'] not in ('pending', 'dispatched'):
            return jsonify(error='This proposal was cancelled. Save a new proposal to process.'), 409
        body = dict(frozen['plan'], requestId=assignment_id(ident),
                    processingProposal={'id': ident, 'revision': 1, 'digest': frozen['digest']})
        # Existing submission owns credentials, baseline, ticker exclusion, reservation,
        # deterministic replay and worker dispatch. Viewing/saving never calls this.
        with co.app.test_request_context('/api/research/assignments', method='POST', json=body):
            return co.app.view_functions['research_assignments.assignments']()
