"""Read-only meeting work inventory and version/answer context."""
import json
from flask import Blueprint,jsonify


def create_blueprint(get_db):
    bp=Blueprint('meeting_workspace',__name__)
    @bp.get('/api/mp/work')
    def work():
        with get_db() as (_,cur):
            cur.execute("""SELECT id,ticker,status,error,input->>'meetingId' AS meeting_id,
                input->'meetingProfile' AS profile,result->>'stage' AS step,
                result->>'completed' AS completed,result->>'total' AS total,created_at,updated_at
                FROM mp_jobs WHERE stage='pipeline' ORDER BY created_at DESC LIMIT 100""")
            rows=[dict(r) for r in cur.fetchall()]
        for r in rows:
            for k in ('created_at','updated_at'):r[k]=r[k].isoformat() if r[k] else None
        return jsonify(jobs=rows)
    @bp.get('/api/mp/meetings/<int:mid>/versions')
    def versions(mid):
        with get_db() as (_,cur):
            cur.execute('SELECT id,version,topics_json,generation_model,created_at FROM mp_question_sets WHERE meeting_id=%s ORDER BY version DESC LIMIT 30',(mid,))
            rows=[dict(r) for r in cur.fetchall()]
        for r in rows:
            value=r.pop('topics_json');r['topics']=value if isinstance(value,list) else json.loads(value or '[]')
            r['created_at']=r['created_at'].isoformat() if r['created_at'] else None
        return jsonify(versions=rows)
    @bp.get('/api/mp/meetings/<int:mid>/quality')
    def quality(mid):
        from meeting_pack_quality import inspect
        with get_db() as (_,cur):
            cur.execute('SELECT topics_json FROM mp_question_sets WHERE meeting_id=%s ORDER BY version DESC LIMIT 1',(mid,))
            row=cur.fetchone()
            if not row:return jsonify(error='No saved question pack yet.'),404
            cur.execute('SELECT filename FROM mp_documents WHERE meeting_id=%s',(mid,))
            names=[r['filename'] for r in cur.fetchall()]
        value=row['topics_json'];topics=value if isinstance(value,list) else json.loads(value or '[]')
        return jsonify(inspect(topics,names))
    @bp.get('/api/mp/meetings/<int:mid>/answers')
    def answers(mid):
        with get_db() as (_,cur):
            cur.execute('SELECT id,question,response_notes,status FROM mp_past_questions WHERE meeting_id=%s ORDER BY id DESC',(mid,))
            rows=[dict(r) for r in cur.fetchall()]
        return jsonify(answers=rows)
    return bp
