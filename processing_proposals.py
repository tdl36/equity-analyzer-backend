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


def original_pack(cur, ticker, filenames, supplemental_ids=()):
    # Downstream workers address frozen originals by filename. A supplemental ID
    # must not save successfully only to discover an unselected namesake at approval.
    cur.execute('''WITH selected AS MATERIALIZED (
        SELECT d.id,d.filename,d.metadata,d.file_data,
            (SELECT COUNT(*) FROM document_files identities
             WHERE identities.ticker=d.ticker AND identities.filename=d.filename) AS identity_count
        FROM document_files d
        WHERE d.ticker=%s AND (d.filename=ANY(%s) OR d.id::text=ANY(%s)) FOR SHARE OF d)
        SELECT id::text AS id,filename,metadata,identity_count,
            CASE WHEN identity_count=1 AND octet_length(file_data)<=20000000
                 AND sum(octet_length(file_data)) OVER ()<=80000000
                 THEN file_data ELSE NULL END AS file_data FROM selected''', (ticker, filenames, list(supplemental_ids)))
    documents = {'charlie:thesis_' + str(r['id']): r for r in cur.fetchall()}
    if len(documents) != len(filenames) + len(supplemental_ids):
        raise ValueError('Selected original identities are missing or ambiguous. Refresh the source library.')
    if any(r['identity_count'] != 1 for r in documents.values()):
        raise ValueError('A selected original shares its filename with another stored original for this company. Resolve the ambiguous identity before saving or approving processing; originals are preserved.')
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
    if frozen['plan'].get('labFullNote'):
        if not frozen.get('labNotePlan'):
            raise ValueError('Full-note settings are missing. Save a fresh proposal.')
        from processing_proposal_note import verify_sources
        verify_sources(cur, row['ticker'], pack, frozen['labNotePlan'])
        proposed_plan['labNotePlan'] = frozen['labNotePlan']
    if frozen.get('sourceContext'):
        proposed_plan['processingSourceContext'] = frozen['sourceContext']
    if frozen.get('sourceEvidence') is not None:
        proposed_plan['processingEvidenceContext'] = frozen['sourceEvidence']
    cur.execute("UPDATE mp_jobs SET status='dispatched',result=%s::jsonb,updated_at=NOW() WHERE id=%s", (json.dumps({'assignmentId': ident}), row['id']))
    return pack


def public(row):
    frozen = obj(row['input'])
    from processing_proposal_note import public as note_public
    from processing_proposal_coverage import summarize
    return dict(id=row['id'], ticker=row['ticker'], status=row['status'], revision=1,
                digest=frozen['digest'], collectionId=frozen['collectionId'], plan=frozen['plan'],
                sources=frozen['sources'], coverage=summarize(frozen), fullNote=note_public(frozen.get('labNotePlan')), sourceContext=frozen.get('sourceContext', {}), baseline=frozen['baseline'], caseRevision=frozen['caseRevision'],
                assignmentId=obj(row['result']).get('assignmentId'), created_at=row['created_at'])



def history_page(cur, collection_id, ticker, args):
    """Bounded, scoped reads; exact recovery never depends on the first page."""
    if (set(args) - {'after', 'requestId'} or len(args) > 1
            or any(len(args.getlist(key)) != 1 for key in args)):
        raise ValueError('Invalid proposal history filters. Refresh saved proposals.')
    key = next(iter(args), None)
    ident = None
    if key:
        try:
            ident = str(uuid.UUID(args[key]))
        except (ValueError, TypeError, AttributeError):
            raise ValueError('Invalid proposal identity. Refresh saved proposals.') from None
    scope = (STAGE, collection_id, ticker)
    if key == 'requestId':
        cur.execute("SELECT * FROM mp_jobs WHERE stage=%s AND input->>'collectionId'=%s AND ticker=%s AND id=%s", (*scope, ident))
        row = cur.fetchone()
        return {'proposals': [public(row)] if row else [], 'nextCursor': None}
    boundary = None
    if key == 'after':
        cur.execute("SELECT created_at,id FROM mp_jobs WHERE stage=%s AND input->>'collectionId'=%s AND ticker=%s AND id=%s", (*scope, ident))
        boundary = cur.fetchone()
        if not boundary:
            raise ValueError('Proposal history cursor is unavailable for this collection. Refresh saved proposals.')
    sql = "SELECT * FROM mp_jobs WHERE stage=%s AND input->>'collectionId'=%s AND ticker=%s"
    values = scope
    if boundary:
        sql += ' AND (created_at,id)<(%s,%s)'
        values += (boundary['created_at'], boundary['id'])
    cur.execute(sql + ' ORDER BY created_at DESC,id DESC LIMIT 51', values)
    rows = list(cur.fetchall())
    return {'proposals': [public(row) for row in rows[:50]],
            'nextCursor': rows[49]['id'] if len(rows) > 50 else None}


def register(co):
    bp = co.bp

    @bp.get('/api/research/assignments/<collection_id>/processing-sources')
    def processing_sources(collection_id):
        parent = co.read(collection_id)
        if not parent or parent['status'] != 'complete' or not obj(parent['input']).get('collectionOnly'):
            return jsonify(error='Choose a completed source-only collection.'), 409
        try:
            if request.args:
                raise ValueError('Collected source inventory does not accept filters.')
            from processing_proposal_sources import collected_inventory
            with co.db() as (_, cur):
                documents = collected_inventory(cur, parent['ticker'], obj(parent['result']).get('sources'))
            response = jsonify(collectionId=collection_id, ticker=parent['ticker'], documents=documents)
            response.headers['Cache-Control'] = 'private, no-store'
            return response
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @bp.get('/api/research/assignments/<collection_id>/processing-library')
    def processing_library(collection_id):
        parent = co.read(collection_id)
        if not parent or parent['status'] != 'complete' or not obj(parent['input']).get('collectionOnly'):
            return jsonify(error='Choose a completed source-only collection.'), 409
        try:
            if set(request.args) - {'after', 'selected'} or len(request.args) > 1 or any(len(request.args.getlist(k)) > 1 for k in request.args):
                raise ValueError('Invalid source-library filters.')
            from processing_proposal_sources import inventory, selected_inventory
            names = [s['filename'] for s in obj(parent['result']).get('sources', [])]
            with co.db() as (_, cur):
                if 'selected' in request.args:
                    data = selected_inventory(cur, parent['ticker'], names, request.args['selected'])
                else:
                    data = inventory(cur, parent['ticker'], names, request.args.get('after'))
            response = jsonify(**data, collectionId=collection_id, ticker=parent['ticker'], after=request.args.get('after')); response.headers['Cache-Control'] = 'private, no-store'
            return response
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @bp.route('/api/research/assignments/<collection_id>/processing-proposals', methods=['GET', 'POST'])
    def processing_proposals(collection_id):
        parent = co.read(collection_id)
        if not parent or parent['status'] != 'complete' or not obj(parent['input']).get('collectionOnly'):
            return jsonify(error='Choose a completed source-only collection.'), 409
        if request.method == 'GET':
            try:
                with co.db() as (_, cur):
                    page = history_page(cur, collection_id, parent['ticker'], request.args)
                response = jsonify(page)
                response.headers['Cache-Control'] = 'private, no-store'
                return response
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or set(data) not in ({'requestId', 'filenames', 'plan'}, {'requestId', 'filenames', 'plan', 'supplementalIds'}):
                raise ValueError('Choose originals, purpose and outputs before saving a proposal.')
            ident = str(uuid.UUID(data['requestId']))
            filenames = data['filenames']
            if not isinstance(filenames, list) or not 1 <= len(filenames) <= 1000 or any(not isinstance(n, str) for n in filenames) or len(set(filenames)) != len(filenames):
                raise ValueError('Choose distinct originals (1–1000) from this collection.')
            from processing_proposal_sources import supplemental_ids, context
            supplements = supplemental_ids(data.get('supplementalIds', []), len(filenames))
            choices = data['plan']
            allowed = {'researchPurpose', 'outputs', 'horizon', 'instruction', 'meeting', 'visualFundamentals', 'labFullNote'}
            if not isinstance(choices, dict) or set(choices) - allowed or 'researchPurpose' not in choices:
                raise ValueError('An explicit thesis-review or event-note purpose is required.')
            parent_plan = obj(parent['input'])
            normalized = plan({**choices, **{k: parent_plan[k] for k in ('ticker', 'since', 'until')}})
            intent = {'collectionId': collection_id, 'plan': normalized, 'filenames': sorted(filenames)}
            if 'supplementalIds' in data:
                intent['supplementalIds'] = supplements
            intent_hash = digest(intent)
            # Exact replay returns the frozen proposal even if evidence has since changed.
            with co.db() as (_, cur):
                cur.execute('SELECT * FROM mp_jobs WHERE id=%s', (ident,))
                old = cur.fetchone()
            if old:
                if old['stage'] != STAGE or obj(old['input']).get('intentHash') != intent_hash:
                    raise ValueError('Proposal identity already used for different choices.')
                return jsonify(public(old)), 200
            from processing_proposal_note import freeze as freeze_note
            note_config = freeze_note(co, normalized)
            baseline_package = co.call('thesis_imports.prepare', ticker=parent['ticker'])['package']
            baseline = baseline_package['baseline']
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
                sources = original_pack(cur, parent['ticker'], filenames, supplements)
                if any(s['filename'] not in filenames and s['filename'] in hashes for s in sources):
                    raise ValueError('An added original belongs to this collection. Select it in the collected evidence list.')
                if any(s['sha256'] != hashes[s['filename']] for s in sources if s['filename'] in filenames):
                    raise ValueError('Original bytes no longer match the completed collection receipt.')
                if len({s['sha256'] for s in sources}) != len(sources):
                    raise ValueError('Selected originals contain identical bytes under different identities. Keep one copy before saving.')
                frozen = dict(collectionId=collection_id, plan=normalized, sources=sources,
                              baseline=baseline, caseRevision=revision, intentHash=intent_hash)
                if 'supplementalIds' in data:
                    frozen['sourceContext'] = context(cur, parent['ticker'], sources, filenames, baseline_package)
                    from research_source_context import freeze as freeze_source_context
                    frozen['sourceEvidence'] = freeze_source_context(sources, frozen['sourceContext'])
                if note_config is not None:
                    from research_source_context import note_context
                    note_context({'processingEvidenceContext':frozen.get('sourceEvidence'),'thesisBaseline':baseline_package}, [{'id':s['documentId'],'filename':s['filename'],'sha256':s['sha256']} for s in sorted(sources,key=lambda s:s['filename'])])
                    from processing_proposal_note import source_plan
                    note_config['sourcePreparation'] = source_plan(cur, parent['ticker'], sources, note_config)
                    frozen['labNotePlan'] = note_config
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
