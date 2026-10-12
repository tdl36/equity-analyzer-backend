"""Monotonic restriction of a frozen original; never edits bytes or research."""
import json
import uuid
from datetime import datetime, timezone
from flask import jsonify, request
from research_originals import verified_original


def restrict(cur, table, ident, source_id, body):
    if table not in ('stock_analysis_runs', 'investment_committee_runs', 'company_research_runs'):
        raise ValueError('Unknown research workflow.')
    uuid.UUID(ident)
    if not isinstance(body, dict) or set(body) != {'sha256', 'reason', 'locator'}:
        raise ValueError('Supply the frozen hash, restriction reason and original locator.')
    for key, limit in (('reason', 1000), ('locator', 200)):
        if not isinstance(body[key], str) or not body[key].strip() or len(body[key]) > limit:
            raise ValueError('Restriction reason and locator are required and bounded.')
    cur.execute(f'SELECT ticker,input,sources FROM {table} WHERE id=%s FOR SHARE', (ident,))
    run = cur.fetchone()
    if not run:
        raise ValueError('Saved research not found.')
    source = next((s for s in run['sources'] if s.get('id') == source_id), None)
    if not source or body['sha256'] != source.get('originalHash'):
        raise ValueError('Select the exact frozen original.')
    cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR UPDATE',
                (run['ticker'], source['filename']))
    document = cur.fetchone()
    verified_original(source, document, run['input'].get('hashes', {}).get(source['filename']))
    metadata = document.get('metadata') or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise ValueError('Malformed source metadata; inspect before restricting.')
    # A retry or concurrent request cannot replace an existing restriction record.
    if metadata.get('aiAllowed') is False:
        return {'restricted': True, 'reused': True, 'sourceId': source_id}
    metadata = dict(metadata)
    metadata['aiAllowed'] = False
    metadata['restrictionObservation'] = {
        'reason': body['reason'].strip(), 'locator': body['locator'].strip(),
        'sha256': body['sha256'], 'runId': ident,
        'observedAt': datetime.now(timezone.utc).isoformat()}
    cur.execute('UPDATE document_files SET metadata=%s::jsonb WHERE ticker=%s AND filename=%s',
                (json.dumps(metadata), run['ticker'], source['filename']))
    return {'restricted': True, 'reused': False, 'sourceId': source_id}


def register(bp, get_db, table, route):
    @bp.post(f'/api/research/{route}-run/<ident>/source/<source_id>/restrict')
    def restrict_original(ident, source_id):
        try:
            with get_db(commit=True) as (_, cur):
                result = restrict(cur, table, ident, source_id, request.get_json(silent=True))
            response = jsonify(result)
            response.headers['Cache-Control'] = 'private, no-store'
            return response
        except (ValueError, TypeError, AttributeError):
            return jsonify(error='Restriction was not recorded. Check the saved source identity, bytes, metadata and bounded reason/locator.'), 400
