"""Read the exact meeting output retained by a coordinated assignment."""
import json
import uuid
from flask import jsonify


def register(coordinator):
    @coordinator.bp.get('/api/research/assignments/<ident>/saved-meeting')
    def saved_meeting(ident):
        try:
            ident = str(uuid.UUID(ident))
        except ValueError:
            return jsonify(error='Invalid assignment.'), 400
        with coordinator.db() as (_, cur):
            # Match the retained delivery receipt, the handoff, and the company.
            # Never use the newest question set or dispatch/resume any work.
            cur.execute("SELECT ticker,result FROM mp_jobs WHERE id=%s AND stage='research_assignment'", (ident,))
            assignment = cur.fetchone()
            if not assignment:
                return jsonify(error='Saved assignment not found.'), 404
            state = assignment['result'] or {}
            if isinstance(state, str):
                state = json.loads(state)
            mid, qid, rid = (state.get(k) for k in ('meetingId', 'meetingQuestionSetId', 'reportId'))
            if type(mid) is not int or type(qid) is not int or not rid:
                return jsonify(error='An exact meeting-version receipt is unavailable. No current version has been substituted.'), 409
            cur.execute('''SELECT q.id,q.meeting_id,q.version,q.status,q.topics_json,
                       q.created_at,q.generation_model,c.ticker
                FROM mp_question_sets q
                JOIN mp_meetings m ON m.id=q.meeting_id
                JOIN mp_companies c ON c.id=m.company_id
                WHERE q.id=%s AND m.id=%s AND c.ticker=%s
                  AND EXISTS (SELECT 1 FROM research_meeting_handoffs h
                      WHERE h.run_id::text=%s AND h.meeting_id=m.id AND h.question_set_id=q.id)''',
                (qid, mid, assignment['ticker'], str(rid)))
            row = cur.fetchone()
            if not row:
                return jsonify(error='The delivered meeting version is unavailable or its identity cannot be verified. No newer version has been substituted.'), 409
            data = dict(row)
            try:
                raw = data.pop('topics_json')
                topics = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(topics, list) or any(not isinstance(t, dict) or not isinstance(t.get('questions', []), list) or any(not isinstance(q, dict) for q in t.get('questions', [])) for t in topics):
                    raise ValueError()
            except (ValueError, TypeError):
                return jsonify(error='The saved meeting-version content is unavailable.'), 409
        response = jsonify(assignmentId=ident, reportId=rid, questionSet={**data, 'topics': topics})
        response.headers['Cache-Control'] = 'private, no-store'
        return response
