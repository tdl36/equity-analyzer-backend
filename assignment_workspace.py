"""Read-only assignment inspection, linked by exact command identity."""
import json,uuid
from research_commands import obj


def delivery_contract(payload, command, collection, source_reuse, prep, versions, reports, proposal, dispatch):
    """Delivery means a stored output exists, not that work was dispatched."""
    def stage(key,label,state,detail):return dict(key=key,label=label,state=state,detail=detail)
    failed={'failed','error','blocked','cancelled','canceled','needs_attention','attention_required','needs_review','dismissed','reverted'}
    collection=collection or {};prep=prep or {};proposal=proposal or {};dispatch=dispatch or {}
    collected=bool(source_reuse or collection.get('status')=='complete')
    steps=[stage('collection','Source collection','delivered' if collected else 'attention' if collection.get('status') in failed or command.get('status') in failed else 'pending',
                 'Saved sources reused' if source_reuse else collection.get('issue') or collection.get('status') or 'Waiting for the Mac collection snapshot')]
    has_report=any(r.get('has_report') for r in reports)
    recap_failed=bool(reports and reports[0].get('status') in failed)
    if not source_reuse:
        steps.append(stage('recap','Analyst recap','attention' if recap_failed else 'delivered' if has_report else 'pending',
                           'Latest recap needs attention; an earlier saved report may still be available in Analyst Inbox' if recap_failed else 'Report saved in Analyst Inbox; approval is separate' if has_report else 'Recap not saved yet; dispatch alone is not completion'))
    if payload.get('meetingPrep'):
        ready=prep.get('status')=='done' and any(v.get('status')=='ready' for v in versions)
        steps.append(stage('questions','Meeting question pack','delivered' if ready else 'attention' if prep.get('status') in failed else 'pending',
                           'Question pack saved; open questions to inspect its supporting passages' if ready else prep.get('error') or prep.get('stage') or ('Preparing from saved originals; no new recap requested' if source_reuse else 'Waiting for verified sources and the analyst recap')))
    if payload.get('autoProposal'):
        status=proposal.get('status')
        steps.append(stage('proposal','Thesis change proposal','delivered' if status in ('awaiting_approval','applied') else 'attention' if status in failed or dispatch.get('status') in failed else 'pending',
                           'Ready for your review; no automatic thesis acceptance' if status=='awaiting_approval' else 'Applied by analyst decision' if status=='applied' else proposal.get('error') or dispatch.get('error') or 'No reviewable proposal saved yet'))
    state='delivered' if all(s['state']=='delivered' for s in steps) else 'attention' if any(s['state']=='attention' for s in steps) else 'in_progress'
    return {'state':state,'steps':steps,'delivered':sum(s['state']=='delivered' for s in steps),'expected':len(steps)}

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
            reports=[]
            if result.get('topic'):
                cur.execute("SELECT id,status,activity_type,updated_at,(NULLIF(BTRIM(output->>'synthesisMarkdown'),'') IS NOT NULL) AS has_report FROM analyst_activities WHERE ticker=%s AND input->>'topic'=%s ORDER BY created_at DESC LIMIT 10",(command['ticker'],result['topic']))
                reports=[dict(r) for r in cur.fetchall()]
            cur.execute("SELECT id,status,error FROM mp_jobs WHERE stage='evidence_amendment' AND input->>'commandId'=%s ORDER BY created_at DESC LIMIT 1",(cid,));proposal=cur.fetchone()
            cur.execute("SELECT status,error FROM mp_jobs WHERE stage='command_proposal_dispatch' AND input->>'commandId'=%s ORDER BY created_at DESC LIMIT 1",(cid,));dispatch=cur.fetchone()
            delivery=delivery_contract(payload,command,collection,result.get('savedSources'),prep,versions,reports,proposal,dispatch)
            response=jsonify(command={k:command[k] for k in ('id','ticker','status','error','created_at','updated_at')},assignment=payload,collection=collection,sourceReuse=result.get('savedSources',False),selection=obj(selection['value']) if selection else None,preparation=prep,meetingId=meeting,documents=documents,versions=versions,reports=reports,proposal=proposal,delivery=delivery)
            response.headers['Cache-Control']='no-store';return response
