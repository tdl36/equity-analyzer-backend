"""Persist explicit pending one-pager selections without accepting or rendering them."""
import json
import uuid
from flask import Blueprint, jsonify, request
from onepager_corrections import digest
from onepager_edition_contract import baseline, prepare_edition


def create_blueprint(get_db):
    bp = Blueprint('onepager_editions', __name__)

    @bp.after_request
    def private_response(response):
        response.headers['Cache-Control'] = 'private, no-store'
        return response

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
                # This is a pending record only: no accepted lineage exists yet.
                cur.execute("SELECT id,ticker,status,input,result FROM mp_jobs WHERE id=%s AND stage='research_assignment'" + (' FOR SHARE' if writing else ''), (aid,))
                assignment = cur.fetchone()
                if not assignment:
                    return jsonify(error='Saved assignment not found.'), 404
                assignment = dict(assignment)
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
                        return jsonify(id=ident, status='pending', reused=True)
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
                pack = baseline(run, assignment, financials)
                if not writing:
                    cur.execute("SELECT to_regclass('onepager_editions') AS name")
                    rows = []
                    if cur.fetchone()['name']:
                        cur.execute('SELECT id,record,created_at FROM onepager_editions WHERE assignment_id=%s ORDER BY created_at DESC LIMIT 20', (aid,))
                        rows = [dict(r) for r in cur.fetchall()]
                        for item in rows:
                            item['stale'] = any(item['record'].get(k) != v for k, v in pack.items())
                    return jsonify(**pack, editions=rows, limit=20, status='pending_only',
                                   caveat='No edition can be accepted or used as an export yet. Reading does not reverify original bytes or usage restrictions.')
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
                                if s.get('id') in {r['proposal'].get('sourceId') for r in corrections}})
                documents = {}
                for filename in files:
                    cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE', (run['ticker'], filename))
                    matches = cur.fetchall()
                    if len(matches) != 1:
                        raise ValueError('An original is missing or ambiguous. Inspect the saved evidence.')
                    documents[filename] = dict(matches[0])
                record = prepare_edition(run, assignment, body, corrections, documents, financials)
                cur.execute('INSERT INTO onepager_editions(id,assignment_id,run_id,payload_hash,record) VALUES(%s,%s,%s,%s,%s::jsonb)',
                            (record['id'], aid, str(rid), digest(body), json.dumps(record)))
                return jsonify(id=record['id'], status='pending', reused=False), 201
        except (ValueError, TypeError, AttributeError) as exc:
            return jsonify(error=str(exc)), 400
    return bp
