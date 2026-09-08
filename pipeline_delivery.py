"""Durable recap result receipt; retries cannot overwrite a different result."""
import hashlib
import json
import uuid


def obj(value):
    if isinstance(value,str):
        try:value=json.loads(value)
        except ValueError:return {}
    return value if isinstance(value,dict) else {}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate(result,detail):
    if not isinstance(result,dict) or not isinstance(result.get('markdown'),str) or not result['markdown'].strip():
        raise ValueError('A nonempty recap result is required')
    if len(json.dumps(result))>8_000_000:raise ValueError('Recap receipt exceeds 8 MB')
    if result.get('topic')!=detail.get('topic'):raise ValueError('Recap topic does not match its assigned job')
    return result


def create_blueprint(get_db):
    from flask import Blueprint, jsonify, request
    bp=Blueprint('pipeline_delivery',__name__)
    @bp.route('/api/pipeline/jobs/<job_id>/result',methods=['POST'])
    def receive(job_id):
        try:uuid.UUID(job_id)
        except (ValueError,TypeError):return jsonify(error='Invalid job ID'),400
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or data.get('status')!='complete':return jsonify(error='A complete result receipt is required'),400
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT job_type,status,steps_detail,result,agent_managed,agent_owner FROM research_pipeline_jobs WHERE id=%s FOR UPDATE',(job_id,));job=cur.fetchone()
            if not job:return jsonify(error='Research job not found; local result retained'),404
            if job['job_type']!='synthesis':return jsonify(error='This receipt is for catalyst synthesis only'),400
            if job.get('agent_managed') and (not data.get('claimToken') or job.get('agent_owner')!=data.get('claimToken')):
                return jsonify(error='Worker claim was superseded; local result retained'),409
            detail=obj(job['steps_detail'])
            try:result=validate(data.get('result'),detail)
            except ValueError as e:return jsonify(error=str(e)),400
            previous=obj(job['result'])
            if previous.get('markdown') and digest(previous)!=digest(result):return jsonify(error='A different result already exists; local original retained for inspection'),409
            if job['status'] not in ('queued','running','failed','complete'):return jsonify(error='Job is no longer eligible for delivery; local original retained'),409
            # Result and linked Inbox output commit in the same transaction.
            # This can recover a receipt after the old startup timeout marked it failed.
            cur.execute("UPDATE research_pipeline_jobs SET status='complete',result=%s::jsonb,error=NULL,progress=100,current_step='Recap delivered',updated_at=NOW(),completed_at=COALESCE(completed_at,NOW()) WHERE id=%s",(json.dumps(result),job_id))
            activity_id=detail.get('activityId');linked=False
            if activity_id:
                cur.execute('SELECT status,output FROM analyst_activities WHERE id=%s FOR UPDATE',(activity_id,));activity=cur.fetchone()
                out=obj(activity['output']) if activity else {}
                if activity and out.get('catalystJobId')==job_id and activity['status'] in ('running','failed'):
                    for key,source in [('synthesisMarkdown','markdown'),('sourceFiles','sourceFiles'),('sourceProvenance','sourceProvenance'),('evidenceSnapshot','evidenceSnapshot'),('claimReview','claimReview'),('processingRecovery','processingRecovery'),('coordination','coordination'),('fileCount','fileCount')]:out[key]=result.get(source)
                    from datetime import datetime,timezone
                    out['completedAt']=datetime.now(timezone.utc).isoformat()
                    cur.execute("UPDATE analyst_activities SET status='pending_review',output=%s::jsonb,error=NULL,updated_at=NOW() WHERE id=%s",(json.dumps(out),activity_id));linked=True
        return jsonify(received=True,jobId=job_id,resultHash=digest(result),inboxLinked=linked)
    return bp
