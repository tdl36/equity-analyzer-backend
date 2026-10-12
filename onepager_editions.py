"""Persist explicit pending one-pager selections with explicitly selected, revalidated pending previews."""
import json
import uuid
from flask import Blueprint, jsonify, request, Response
from onepager_corrections import digest, evidence_items
from onepager_edition_contract import baseline, prepare_edition
from onepager_decisions import read_lineage, record_decision


def create_blueprint(get_db, investor_identity=lambda: None):
    bp = Blueprint('onepager_editions', __name__)

    @bp.after_request
    def private_response(response):
        response.headers['Cache-Control'] = 'private, no-store'
        return response

    @bp.post('/api/research/assignments/<aid>/onepager-editions/<eid>/decision')
    def decide(aid, eid):
        investor = investor_identity()
        if not investor:
            return jsonify(error='Sign in as the investor to record an editorial decision. Agent credentials cannot approve editions.'), 403
        try:
            result = record_decision(get_db, aid, eid, request.get_json(silent=True), investor_id=investor)
            decision = result['decision']
            return jsonify(reused=result['reused'], decision={k:decision[k] for k in
                           ('requestId','editionId','action','decidedAt','editionDigest')})
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            return jsonify(error=str(exc)), 409

    @bp.route('/api/research/assignments/<aid>/onepager-editions', methods=['GET', 'POST'])
    def editions(aid):
        try:
            aid = str(uuid.UUID(aid))
            writing = request.method == 'POST'
            body = request.get_json(silent=True) if writing else None
            if writing and (not isinstance(body, dict) or len(json.dumps(body)) > 12000):
                raise ValueError('Invalid or oversized edition selection.')
            with get_db(commit=writing) as (_, cur):
                # Lock the coordinating assignment before its report and evidence.
                # Shared lock also gives read responses a coherent lineage while
                # the separate decision store serializes on this assignment.
                cur.execute("SELECT id,ticker,status,input,result FROM mp_jobs WHERE id=%s AND stage='research_assignment' FOR SHARE", (aid,))
                assignment = cur.fetchone()
                if not assignment:
                    return jsonify(error='Saved assignment not found.'), 404
                assignment = dict(assignment)
                parent, decisions = read_lineage(cur, aid)
                if writing and assignment['status'] in ('cancelled', 'stopped'):
                    return jsonify(error='This assignment was stopped. No new edition was saved.'), 409
                if writing:
                    cur.execute("SELECT pg_advisory_xact_lock(hashtext('onepager-editions-schema'))")
                    cur.execute('''CREATE TABLE IF NOT EXISTS onepager_editions (
                        id UUID PRIMARY KEY, assignment_id TEXT NOT NULL, run_id TEXT NOT NULL,
                        payload_hash TEXT NOT NULL, record JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')
                    cur.execute('CREATE INDEX IF NOT EXISTS onepager_editions_assignment_idx ON onepager_editions(assignment_id,created_at DESC)')
                    ident = str(uuid.UUID(body.get('requestId', '')))
                    cur.execute('SELECT assignment_id,payload_hash,record FROM onepager_editions WHERE id=%s', (ident,))
                    previous = cur.fetchone()
                    if previous:
                        if previous['assignment_id'] != aid or previous['payload_hash'] != digest(body):
                            return jsonify(error='This submission identity belongs to a different edition selection.'), 409
                        decision = decisions.get(ident)
                        return jsonify(id=ident, status=('accepted' if decision['action']=='accept' else 'rejected') if decision else 'pending', reused=True)
                result = assignment.get('result') or {}
                rid = result.get('reportId')
                if not rid:
                    return jsonify(error='The assignment has no saved report.'), 409
                cur.execute('SELECT id,ticker,status,input,baseline,sources,state FROM stock_analysis_runs WHERE id=%s' + (' FOR SHARE' if writing else ''), (rid,))
                run = cur.fetchone()
                if not run:
                    return jsonify(error='Saved report not found.'), 404
                run = dict(run)
                financials = result.get('visualFundamentals')
                pack = baseline(run, assignment, financials, parent_id=parent)
                if not writing:
                    cur.execute("SELECT to_regclass('onepager_editions') AS name")
                    rows = []
                    if cur.fetchone()['name']:
                        cur.execute('SELECT id,record,created_at FROM onepager_editions WHERE assignment_id=%s ORDER BY created_at DESC LIMIT 20', (aid,))
                        rows = [dict(r) for r in cur.fetchall()]
                        for item in rows:
                            item['decision'] = decisions.get(str(item['id']))
                            item['stale'] = any(item['record'].get(k) != v for k, v in pack.items()
                                                if k != 'parentId' or not item['decision'])
                            item['status'] = ('accepted' if item['decision']['action']=='accept' else 'rejected') if item['decision'] else 'pending'
                            item['currentAccepted'] = str(item['id']) == parent
                            if item['decision'] and digest(item['record']) != item['decision']['editionDigest']:
                                raise ValueError('The decided edition content changed. Inspect saved lineage.')
                    accepted = None
                    if parent:
                        cur.execute('SELECT record FROM onepager_editions WHERE id=%s AND assignment_id=%s', (parent, aid))
                        accepted = cur.fetchone()['record']
                    return jsonify(**pack, editions=rows, acceptedEdition=accepted, limit=20, status='pending_only',
                                   canDecide=bool(investor_identity()),
                                   caveat='Decisions require investor sign-in and fresh evidence checks. Reading this list does not reverify original bytes or usage restrictions.')
                ids = body.get('correctionIds')
                if not isinstance(ids, list) or not 1 <= len(ids) <= 20:
                    raise ValueError('Select between one and twenty corrections.')
                ids = [str(uuid.UUID(value)) for value in ids]
                cur.execute("SELECT to_regclass('onepager_corrections') AS name")
                if not cur.fetchone()['name']:
                    raise ValueError('No saved corrections are available for this edition.')
                cur.execute('SELECT id,run_id,fingerprint,payload_hash,proposal FROM onepager_corrections WHERE id=ANY(%s::uuid[]) FOR SHARE', (ids,))
                corrections = [dict(r) for r in cur.fetchall()]
                # Only fetch originals named by this report; validation rejects
                # unrelated correction IDs, ambiguous documents and stale records.
                files = sorted({s['filename'] for s in run.get('sources', [])
                                if s.get('id') in {e.get('sourceId') for r in corrections for e in evidence_items(r['proposal'])}})
                documents = {}
                for filename in files:
                    cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE', (run['ticker'], filename))
                    matches = cur.fetchall()
                    if len(matches) != 1:
                        raise ValueError('An original is missing or ambiguous. Inspect the saved evidence.')
                    documents[filename] = dict(matches[0])
                record = prepare_edition(run, assignment, body, corrections, documents, financials, parent_id=parent)
                cur.execute('INSERT INTO onepager_editions(id,assignment_id,run_id,payload_hash,record) VALUES(%s,%s,%s,%s,%s::jsonb)',
                            (record['id'], aid, str(rid), digest(body), json.dumps(record)))
                return jsonify(id=record['id'], status='pending', reused=False), 201
        except (ValueError, TypeError, AttributeError) as exc:
            return jsonify(error=str(exc)), 400
    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/preview')
    def preview(aid, eid):
        return render_version(aid, eid, False)

    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/export')
    def export(aid, eid):
        return render_version(aid, eid, True)

    def render_version(aid, eid, exporting):
        try:
            aid, eid = str(uuid.UUID(aid)), str(uuid.UUID(eid))
            layout = request.args.get('layout', 'illustrated')
            if set(request.args) - {'layout'} or layout not in ('illustrated', 'brief'):
                return jsonify(error='Choose the illustrated or brief pending preview.'), 400
        except (ValueError, TypeError, AttributeError):
            return jsonify(error='Invalid assignment or edition identity.'), 400
        try:
            # Read-only transaction; shared locks keep every checked source and
            # baseline stable through rendering. No schema creation or dispatch.
            with get_db() as (_, cur):
                cur.execute("SELECT id,ticker,status,input,result FROM mp_jobs WHERE id=%s AND stage='research_assignment' FOR SHARE", (aid,))
                assignment = cur.fetchone()
                if not assignment:
                    return jsonify(error='Saved assignment not found.'), 404
                assignment = dict(assignment)
                parent, decisions = read_lineage(cur, aid)
                if exporting and decisions.get(eid, {}).get('action') != 'accept':
                    raise ValueError('Only an explicitly accepted edition can be exported.')
                if not exporting and eid in decisions:
                    raise ValueError('This edition already has an investor decision. Pending preview is unavailable; inspect its saved review.')
                cur.execute("SELECT to_regclass('onepager_editions') AS name")
                if not cur.fetchone()['name']:
                    return jsonify(error='Saved edition not found.'), 404
                cur.execute('SELECT run_id,record FROM onepager_editions WHERE id=%s AND assignment_id=%s FOR SHARE', (eid, aid))
                saved = cur.fetchone()
                if not saved:
                    return jsonify(error='Saved edition not found for this assignment.'), 404
                result = assignment.get('result') or {}
                rid = result.get('reportId')
                record = saved['record']
                if not isinstance(record, dict) or record.get('id') != eid or str(saved['run_id']) != str(rid):
                    raise ValueError('Edition identity changed. Inspect its saved assignment.')
                cur.execute('SELECT id,ticker,status,input,baseline,sources,state FROM stock_analysis_runs WHERE id=%s FOR SHARE', (rid,))
                run = cur.fetchone()
                if not run:
                    return jsonify(error='Saved report not found.'), 404
                run = dict(run)
                ids = record.get('correctionIds')
                if not isinstance(ids, list) or not 1 <= len(ids) <= 20:
                    raise ValueError('Invalid saved correction selection.')
                ids = [str(uuid.UUID(value)) for value in ids]
                cur.execute("SELECT to_regclass('onepager_corrections') AS name")
                if not cur.fetchone()['name']:
                    raise ValueError('Saved corrections are unavailable. Inspect the edition.')
                cur.execute('SELECT id,run_id,fingerprint,payload_hash,proposal FROM onepager_corrections WHERE id=ANY(%s::uuid[]) FOR SHARE', (ids,))
                corrections = [dict(row) for row in cur.fetchall()]
                # A preview also repeats unchanged claims. Reverify the entire
                # frozen pack, not just the sources behind proposed corrections.
                from company_research import eligible
                from research_originals import verified_original
                documents = {}
                sources = run.get('sources', [])
                for filename in sorted({source['filename'] for source in sources}):
                    cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE', (run['ticker'], filename))
                    matches = cur.fetchall()
                    if len(matches) != 1:
                        raise ValueError('An original is missing or ambiguous. Inspect the saved evidence.')
                    document = dict(matches[0])
                    if not eligible(document.get('metadata') or {}):
                        raise ValueError('A frozen source is restricted. Review usage restrictions before previewing.')
                    documents[filename] = document
                for source in sources:
                    verified_original(source, documents[source['filename']], (run.get('input', {}).get('hashes') or {}).get(source['filename']))
                from onepager_edition_evidence import pending_evidence, accepted_evidence
                from research_onepager import render as brief
                from research_infographic import render as illustrated
                financials = result.get('visualFundamentals')
                context = (accepted_evidence(run, assignment, record, corrections, documents, financials, decision=decisions[eid]) if exporting else
                           pending_evidence(run, assignment, record, corrections, documents, financials, parent_id=parent))
                html = (illustrated(run, assignment['input'], financials, preview=context) if layout == 'illustrated'
                        else brief(run, assignment['input'], preview=context))
                if exporting:
                    response = Response(html, mimetype='text/html')
                    response.headers['Content-Disposition'] = f'attachment; filename="edition-{eid}-{layout}.html"'
                    response.headers['Content-Security-Policy'] = "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data: https:; font-src data: https:"
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    return response
                return jsonify(id=eid, assignmentId=aid, runId=str(rid), ticker=run['ticker'],
                               status='pending', layout=layout, html=html,
                               caveat='Pending preview only. Source bytes and usage restrictions rechecked; claim meaning still requires investor review. Original outputs remain unchanged.')
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            return jsonify(error=str(exc)), 409
    return bp
