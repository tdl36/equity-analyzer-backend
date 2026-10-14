"""Pending Lab draft review adapter; deliberately not registered in production.

No generation, publication, legacy-note writes or automatic schema creation.
The caller must provide the application's authenticated-user predicate.
"""
import json
import re
from flask import Blueprint, jsonify, request
from pipeline_lab_notes import edit, history, read


def create_blueprint(get_db, authenticated):
    bp = Blueprint('pipeline_lab_notes', __name__)

    @bp.before_request
    def require_identity():
        if not authenticated():
            return jsonify(error='Sign in to review Lab drafts.'), 401

    @bp.after_request
    def private(response):
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    def integer(name, default=None):
        values = request.args.getlist(name)
        if not values:
            return default
        if len(values) != 1 or not re.fullmatch(r'[1-9][0-9]{0,8}', values[0]):
            raise ValueError('Use one positive integer for ' + name + '.')
        return int(values[0])

    @bp.get('/api/pipeline-lab/notes/<note_id>')
    def draft(note_id):
        try:
            if set(request.args) - {'revision'}:
                raise ValueError('Unknown draft query option.')
            revision = integer('revision')
            with get_db() as (_, cur):
                result = read(cur, note_id, revision)
            if result is None:
                return jsonify(error='Draft revision not found.'), 404
            return jsonify(result)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @bp.get('/api/pipeline-lab/notes/<note_id>/history')
    def versions(note_id):
        try:
            if set(request.args) - {'before', 'limit'}:
                raise ValueError('Unknown history query option.')
            before, limit = integer('before'), integer('limit', 20)
            with get_db() as (_, cur):
                result = history(cur, note_id, before, limit)
            return jsonify(result)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @bp.post('/api/pipeline-lab/notes/<note_id>/revisions')
    def save(note_id):
        try:
            if request.args:
                raise ValueError('Revision edits do not accept query options.')
            if request.mimetype != 'application/json':
                raise ValueError('Send a JSON draft edit.')
            limit = 2_010_000
            if request.content_length is not None and request.content_length > limit:
                return jsonify(error='Draft edit exceeds its size limit.'), 413
            raw = request.stream.read(limit + 1)
            if len(raw) > limit:
                return jsonify(error='Draft edit exceeds its size limit.'), 413
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ValueError('Invalid draft edit JSON.')
            if not isinstance(data, dict) or set(data) != {'eventId', 'expectedRevision', 'title', 'markdown'}:
                raise ValueError('Edit identity, baseline revision, title and text are required.')
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        try:
            # Let conflicts unwind the transaction before preparing the response.
            with get_db(commit=True) as (_, cur):
                result = edit(cur, note_id, data['eventId'], data['expectedRevision'],
                              {'title': data['title'], 'markdown': data['markdown']})
            return jsonify(result)
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    return bp
