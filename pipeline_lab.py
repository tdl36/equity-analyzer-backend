"""Isolated, centrally saved Pipeline Lab selections. Selections and exact-original checks for coordinated research."""
import base64
import hashlib
import json
import re
from flask import Blueprint, jsonify, request


def config(value):
    if not isinstance(value, dict) or set(value) != {'thesis', 'note'}:
        raise ValueError('Separate thesis and note selections are required.')
    out = {}
    for task in ('thesis', 'note'):
        rows = value[task]
        if not isinstance(rows, list) or len(rows) > 1000:
            raise ValueError('Select at most 1000 source references.')
        if any(not isinstance(x, str) or not 1 <= len(x) <= 1200 for x in rows) or len(set(rows)) != len(rows):
            raise ValueError('Source references must be distinct bounded identifiers.')
        out[task] = sorted(rows)
    return out


def check_originals(ticker, task, revision, references, documents):
    """Read-only byte preflight, never a provenance receipt or dispatch permit."""
    blockers = []
    verified = []
    if not references:
        blockers.append('Select at least one original for this task.')
    if len(references) > 1000:
        blockers.append('Select at most 1000 source references.')
        return {'verified': [], 'blockers': blockers, 'digest': None}
    total = 0
    for ref in references:
        if not ref.startswith('charlie:thesis_'):
            blockers.append('A selected source needs a supported original import: ' + ref)
            continue
        row = documents.get(ref)
        if not row:
            blockers.append('A selected stored original is missing: ' + ref)
            continue
        from company_research import eligible
        if not eligible(row.get('metadata') or {}):
            blockers.append('A selected original is excluded or reference-only: ' + ref)
            continue
        try:
            encoded = row.get('file_data')
            if not isinstance(encoded, str) or not 0 < len(encoded) <= 20_000_000:
                raise ValueError()
            raw = base64.b64decode(encoded, validate=True)
            if not 0 < len(raw) <= 15_000_000:
                raise ValueError()
        except (ValueError, TypeError):
            blockers.append('A selected original is empty, invalid, exceeds 15 MB, or the selected pack exceeds 60 MB: ' + ref)
            continue
        total += len(raw)
        verified.append({'reference': ref, 'filename': row['filename'], 'bytes': len(raw),
                         'sha256': hashlib.sha256(raw).hexdigest()})
    if total > 60_000_000:
        blockers.append('The selected originals exceed the 60 MB pack limit.')
    names = [r['filename'] for r in verified]
    if len(names) != len(set(names)):
        blockers.append('Selected originals share a filename; resolve their identities before generation.')
    payload = {'ticker': ticker, 'task': task, 'revision': revision,
               'originals': sorted(verified, key=lambda r: r['reference'])}
    digest = None if blockers else hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {'verified': verified, 'blockers': blockers, 'digest': digest}


def resolve_selection(cur, ticker, selection):
    """Freeze IDs and byte hashes from the saved task, inside assignment submission."""
    cur.execute('SELECT value FROM app_settings WHERE key=%s FOR SHARE', ('pipeline_lab_v1:' + ticker,))
    row = cur.fetchone()
    current = json.loads(row['value']) if row and isinstance(row['value'], str) else (row['value'] if row else None)
    if not current or current['revision'] != selection['revision']:
        raise ValueError('Lab settings changed. Reload and check your saved selection again.')
    refs = config(current['selection'])[selection['task']]
    ids = [r[len('charlie:thesis_'):] for r in refs if r.startswith('charlie:thesis_')]
    documents = {}
    if ids and len(refs) <= 1000:
        cur.execute("WITH selected AS MATERIALIZED (SELECT id,filename,metadata,file_data FROM document_files WHERE ticker=%s AND id::text=ANY(%s) FOR SHARE) SELECT id::text AS id, filename, metadata, CASE WHEN octet_length(file_data)<=20000000 AND sum(octet_length(file_data)) OVER ()<=80000000 THEN file_data ELSE NULL END AS file_data FROM selected", (ticker, ids))
        documents = {'charlie:thesis_' + str(d['id']): d for d in cur.fetchall()}
    result = check_originals(ticker, selection['task'], selection['revision'], refs, documents)
    if result['blockers']:
        raise ValueError(' '.join(result['blockers']))
    if result['digest'] != selection['digest']:
        raise ValueError('Selected original bytes changed since your check. Check them again before starting.')
    return [dict(r, documentId=r['reference'][len('charlie:thesis_'):]) for r in result['verified']]


def create_blueprint(get_db):
    bp = Blueprint('pipeline_lab', __name__)
    @bp.after_request
    def private(response):
        response.headers['Cache-Control'] = 'private, no-store'
        return response

    @bp.get('/api/pipeline-lab/original/<ticker>/<source_id>')
    def original(ticker, source_id):
        # Exact storage identity and company scope; no filename fallback or model call.
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}', ticker) or not re.fullmatch(r'thesis_[1-9][0-9]{0,18}', source_id):
            return jsonify(error='Choose a supported stored original identity.'), 400
        if request.args:
            return jsonify(error='Original downloads do not accept query options.'), 400
        with get_db() as (_, cur):
            cur.execute("SELECT filename,metadata,CASE WHEN octet_length(file_data)<=20000000 THEN file_data ELSE NULL END AS file_data FROM document_files WHERE ticker=%s AND id::text=%s", (ticker, source_id[7:]))
            document = cur.fetchone()
        if not document:
            return jsonify(error='Stored original not found for this company.'), 404
        ref = 'charlie:' + source_id
        checked = check_originals(ticker, 'note', 0, [ref], {ref: document})
        if checked['blockers']:
            return jsonify(error=' '.join(checked['blockers'])), 409
        from flask import send_file
        from io import BytesIO
        from werkzeug.utils import secure_filename
        raw = base64.b64decode(document['file_data'], validate=True)
        response = send_file(BytesIO(raw), mimetype='application/octet-stream', as_attachment=True,
                             download_name=secure_filename(document['filename']) or 'original', conditional=False, max_age=0)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "sandbox; default-src 'none'"
        response.headers['X-Original-SHA256'] = checked['verified'][0]['sha256']
        return response

    @bp.route('/api/pipeline-lab/config/<ticker>', methods=['GET', 'PUT'])
    def settings(ticker):
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}', ticker):
            return jsonify(error='Invalid ticker.'), 400
        key = 'pipeline_lab_v1:' + ticker
        body = request.get_json(silent=True) if request.method == 'PUT' else None
        if request.method == 'PUT':
            try:
                if not isinstance(body, dict) or set(body) != {'revision', 'selection'} or type(body['revision']) is not int or body['revision'] < 0:
                    raise ValueError('A saved configuration revision is required.')
                selection = config(body['selection'])
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
        with get_db(commit=request.method == 'PUT') as (_, cur):
            if request.method == 'PUT':
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (key,))
            cur.execute('SELECT value FROM app_settings WHERE key=%s', (key,))
            row = cur.fetchone()
            saved = json.loads(row['value']) if row and isinstance(row['value'], str) else (row['value'] if row else None)
            current = saved or {'revision': 0, 'selection': {'thesis': [], 'note': []}}
            if request.method == 'PUT':
                if body['revision'] != current['revision']:
                    # A lost response may be retried without creating another version.
                    if body['revision'] + 1 == current['revision'] and selection == current['selection']:
                        return jsonify(current)
                    return jsonify(error='This Lab configuration changed on another device. Reload saved settings before saving.'), 409
                current = {'revision': current['revision'] + 1, 'selection': selection}
                cur.execute('INSERT INTO app_settings(key,value,updated_at) VALUES(%s,%s,NOW()) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()', (key, json.dumps(current)))
        return jsonify(current)
    @bp.route('/api/pipeline-lab/check/<ticker>', methods=['POST'])
    def check(ticker):
        body = request.get_json(silent=True)
        if (not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}', ticker)
                or not isinstance(body, dict) or set(body) != {'task', 'revision'}
                or body['task'] not in ('thesis', 'note')
                or type(body['revision']) is not int or body['revision'] < 0):
            return jsonify(error='A task and saved revision are required.'), 400
        with get_db() as (_, cur):
            cur.execute('SELECT value FROM app_settings WHERE key=%s', ('pipeline_lab_v1:' + ticker,))
            row = cur.fetchone()
            current = json.loads(row['value']) if row and isinstance(row['value'], str) else (row['value'] if row else None)
            if not current or current['revision'] != body['revision']:
                return jsonify(error='Saved settings changed or are unavailable. Reload before checking originals.'), 409
            refs = config(current['selection'])[body['task']]
            documents = {}
            ids = [r[len('charlie:thesis_'):] for r in refs if r.startswith('charlie:thesis_')]
            if ids and len(refs) <= 1000:
                # Bound returned base64 before loading it into Python. Oversize rows
                # remain visible as blockers rather than disappearing from selection.
                cur.execute("SELECT id::text AS id, filename, metadata, CASE WHEN octet_length(file_data)<=20000000 AND sum(octet_length(file_data)) OVER ()<=80000000 THEN file_data ELSE NULL END AS file_data FROM document_files WHERE ticker=%s AND id::text=ANY(%s)", (ticker, ids))
                documents = {'charlie:thesis_' + str(d['id']): d for d in cur.fetchall()}
            result = check_originals(ticker, body['task'], current['revision'], refs, documents)
        return jsonify(**result, task=body['task'], revision=current['revision'],
                       checkedOnly=True, generationAvailable=not result['blockers'])
    from pipeline_lab_import import register
    register(bp,get_db)
    return bp
