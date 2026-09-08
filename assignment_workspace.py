"""Read-only assignment inspection, linked by exact command identity."""
import json,uuid
from research_commands import obj

def routes(bp,get_db):
    from flask import jsonify
    @bp.route('/api/research/assignments/<cid>',methods=['GET'])
    def detail(cid):
        try:uuid.UUID(cid)
        except ValueError:return jsonify(error='Invalid assignment'),400
        with get_db() as (_,cur):
            cur.execute("SELECT id,ticker,status,input,result,error,created_at,updated_at FROM mp_jobs WHERE id=%s AND stage='collection_control'",(cid,));row=cur.fetchone()
            if not row:return jsonify(error='Assignment not found'),404
            command=dict(row);payload=obj(row['input']).get('payload',{});result=obj(row['result']) or {}
            cur.execute("SELECT value FROM app_settings WHERE key='collection_control_snapshot'");snapshot=cur.fetchone();snapshot=obj(snapshot['value']) if snapshot else {}
            collection=next((r for r in snapshot.get('requests',[]) if r['id']==result.get('refreshRequestId')),None)
            cur.execute('SELECT value FROM app_settings WHERE key=%s',('source_shortlist:'+cid,));selection=cur.fetchone()
            cur.execute("SELECT id,status,error,input,result,created_at,updated_at FROM mp_jobs WHERE stage='command_meeting' AND input->>'commandId'=%s",(cid,));prep=cur.fetchone()
            documents=[];versions=[];meeting=None
            if prep:
                meeting=obj(prep['input'])['meetingId']
                cur.execute('SELECT filename,doc_type FROM mp_documents WHERE meeting_id=%s ORDER BY upload_order',(meeting,));documents=[dict(r) for r in cur.fetchall()]
                cur.execute('SELECT id,version,status,created_at,generation_model FROM mp_question_sets WHERE meeting_id=%s ORDER BY version DESC',(meeting,));versions=[dict(r) for r in cur.fetchall()]
                checkpoint=obj(prep['result']) or {};prep={k:prep[k] for k in ('id','status','error','created_at','updated_at')}|{'stage':checkpoint.get('stage'),'completed':checkpoint.get('completed'),'total':checkpoint.get('total'),'cachedSources':checkpoint.get('cachedSources',sum(bool(x and x.get('_cacheHit')) for x in checkpoint.get('analyses',[])))}
            return jsonify(command={k:command[k] for k in ('id','ticker','status','error','created_at','updated_at')},assignment=payload,collection=collection,sourceReuse=result.get('savedSources',False),selection=obj(selection['value']) if selection else None,preparation=prep,meetingId=meeting,documents=documents,versions=versions)
