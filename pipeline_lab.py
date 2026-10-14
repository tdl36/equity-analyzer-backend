"""Isolated, centrally saved Pipeline Lab selections. No research dispatch."""
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
    if len(references) > 8:
        blockers.append('Select at most eight originals for a bounded research pack.')
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
        try:
            encoded = row.get('file_data')
            if not isinstance(encoded, str) or not 0 < len(encoded) <= 20_000_000:
                raise ValueError()
            raw = base64.b64decode(encoded, validate=True)
            if not 0 < len(raw) <= 15_000_000:
                raise ValueError()
        except (ValueError, TypeError):
            blockers.append('A selected original is empty, invalid or exceeds 15 MB: ' + ref)
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


def create_blueprint(get_db):
    bp = Blueprint('pipeline_lab', __name__)
    @bp.after_request
    def private(response):
        response.headers['Cache-Control'] = 'private, no-store'
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
            if ids and len(refs) <= 8:
                # Bound returned base64 before loading it into Python. Oversize rows
                # remain visible as blockers rather than disappearing from selection.
                cur.execute("SELECT id::text AS id, filename, CASE WHEN octet_length(file_data)<=20000000 THEN file_data ELSE NULL END AS file_data FROM document_files WHERE ticker=%s AND id::text=ANY(%s)", (ticker, ids))
                documents = {'charlie:thesis_' + str(d['id']): d for d in cur.fetchall()}
            result = check_originals(ticker, body['task'], current['revision'], refs, documents)
        return jsonify(**result, task=body['task'], revision=current['revision'],
                       checkedOnly=True, generationAvailable=False)
    return bp
