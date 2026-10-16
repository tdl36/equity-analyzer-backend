"""Read-only meeting work inventory and version/answer context."""
import json
from flask import Blueprint,jsonify,request


def create_blueprint(get_db):
    bp=Blueprint('meeting_workspace',__name__)
    @bp.after_request
    def private_context(response):
        response.headers['Cache-Control']='private, no-store'
        return response
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
            cur.execute('SELECT id,meeting_id,version,topics_json,generation_model,created_at FROM mp_question_sets WHERE meeting_id=%s ORDER BY version DESC LIMIT 31',(mid,))
            rows=[dict(r) for r in cur.fetchall()]
        truncated=len(rows)>30;rows=rows[:30]
        for r in rows:
            value=r.pop('topics_json');r['topics']=value if isinstance(value,list) else json.loads(value or '[]')
            r['created_at']=r['created_at'].isoformat() if r['created_at'] else None
        return jsonify(meetingId=mid,versions=rows,truncated=truncated)
    @bp.get('/api/mp/meetings/<int:mid>/quality')
    def quality(mid):
        from meeting_pack_quality import inspect
        selected=request.args.get('questionSetId')
        if selected is not None and (len(selected)>10 or not selected.isascii() or not selected.isdigit() or not 0<int(selected)<2147483648):
            return jsonify(error='Invalid saved question pack.'),400
        with get_db() as (_,cur):
            if selected is not None:
                cur.execute('SELECT id,topics_json FROM mp_question_sets WHERE meeting_id=%s AND id=%s',(mid,int(selected)))
            else:
                cur.execute('SELECT id,topics_json FROM mp_question_sets WHERE meeting_id=%s ORDER BY version DESC LIMIT 1',(mid,))
            row=cur.fetchone()
            if not row:return jsonify(error='No saved question pack yet.'),404
            cur.execute('SELECT filename FROM mp_documents WHERE meeting_id=%s',(mid,))
            names=[r['filename'] for r in cur.fetchall()]
        value=row['topics_json'];topics=value if isinstance(value,list) else json.loads(value or '[]')
        result=inspect(topics,names)
        result['scope']+=' Source counts use currently attached meeting documents, not a reconstructed historical inventory.'
        return jsonify(**result,meetingId=mid,questionSetId=row['id'])
    @bp.get('/api/mp/meetings/<int:mid>/answers')
    def answers(mid):
        with get_db() as (_,cur):
            cur.execute('SELECT id,meeting_id,question,response_notes,status FROM mp_past_questions WHERE meeting_id=%s ORDER BY id DESC',(mid,))
            rows=[dict(r) for r in cur.fetchall()]
        return jsonify(meetingId=mid,answers=rows)
    @bp.get('/api/mp/meetings/<int:mid>/answers/<int:aid>')
    def answer(mid,aid):
        from meeting_answer_capture import read
        with get_db() as (_,cur):row=read(cur,aid,mid)
        response=jsonify(answer=row) if row else jsonify(error='Recorded question not found for this meeting.')
        response.status_code=200 if row else 404
        response.headers['Cache-Control']='private, no-store'
        return response
    return bp
