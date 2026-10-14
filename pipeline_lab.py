"""Isolated, centrally saved Pipeline Lab selections. No research dispatch."""
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
    return bp
