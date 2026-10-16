"""Read-only assignment inspection, linked by exact command identity."""
import json,uuid,re
from research_commands import obj


def delivery_contract(payload, command, collection, source_reuse, prep, versions, reports, proposal, dispatch, meeting_issue=None):
    """Delivery means a stored output exists, not that work was dispatched."""
    def stage(key,label,state,detail):return dict(key=key,label=label,state=state,detail=detail)
    failed={'failed','error','blocked','cancelled','canceled','needs_attention','attention','needs_auth','attention_required','needs_review','dismissed','reverted','stopped'}
    collection=collection or {};prep=prep or {};proposal=proposal or {};dispatch=dispatch or {}
    collected=bool(source_reuse or collection.get('status')=='complete')
    collection_unknown=not source_reuse and not collection and command.get('status') in ('applied','done')
    collection_detail=collection.get('issue') or command.get('error') or (
        'AlphaSense sign-in needs attention. Open signed-in Chrome and inspect Collection controls; saved originals are retained.'
        if collection.get('status')=='needs_auth' else collection.get('status') or ('Collection status is unavailable in the current snapshot and saved receipts. Completion or continued collection cannot be inferred.' if collection_unknown else 'Waiting for the Mac collection snapshot'))
    steps=[stage('collection','Source collection','delivered' if collected else 'attention' if collection.get('status') in failed or command.get('status') in failed else 'unknown' if collection_unknown else 'pending',
                 'Saved sources reused' if source_reuse else collection_detail)]
    # Rows are newest-first. Earlier saved outputs remain available, but cannot
    # establish delivery of the current attempt.
    has_report=bool(reports and reports[0].get('has_report'))
    older_report=any(r.get('has_report') for r in reports[1:])
    recap_failed=bool(reports and reports[0].get('status') in failed)
    if not source_reuse:
        steps.append(stage('recap','Analyst recap','attention' if recap_failed else 'delivered' if has_report else 'pending',
                           'Latest recap needs attention; an earlier saved report may still be available in Analyst Inbox' if recap_failed else 'Report saved in Analyst Inbox; approval is separate' if has_report else 'Latest recap is not saved yet; an earlier report remains available in Analyst Inbox' if older_report else 'Recap not saved yet; dispatch alone is not completion'))
    if payload.get('meetingPrep'):
        ready=not meeting_issue and prep.get('status')=='done' and bool(versions and versions[0].get('status')=='ready')
        questions_failed=bool(meeting_issue) or prep.get('status') in failed or bool(versions and versions[0].get('status') in failed)
        steps.append(stage('questions','Meeting question pack','delivered' if ready else 'attention' if questions_failed else 'pending',
                           'Question pack saved; open questions to inspect its supporting passages' if ready else meeting_issue or prep.get('error') or ('Latest question version needs attention; earlier saved versions remain available' if questions_failed else None) or prep.get('stage') or ('Preparing from saved originals; no new recap requested' if source_reuse else 'Waiting for verified sources and the analyst recap')))
    if payload.get('autoProposal'):
        status=proposal.get('status')
        steps.append(stage('proposal','Thesis change proposal','delivered' if status in ('awaiting_approval','applied') else 'attention' if status in failed or dispatch.get('status') in failed else 'pending',
                           'Ready for your review; no automatic thesis acceptance' if status=='awaiting_approval' else 'Applied by analyst decision' if status=='applied' else proposal.get('error') or dispatch.get('error') or 'No reviewable proposal saved yet'))
    state='delivered' if all(s['state']=='delivered' for s in steps) else 'attention' if any(s['state']=='attention' for s in steps) else 'unknown' if any(s['state']=='unknown' for s in steps) else 'in_progress'
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
            from collection_history import lookup
            collection=lookup(cur,snapshot,result.get('refreshRequestId'),command['ticker'])
            cur.execute('SELECT value FROM app_settings WHERE key=%s',('source_shortlist:'+cid,));selection=cur.fetchone()
            cur.execute("SELECT id,ticker,status,error,input,result,created_at,updated_at FROM mp_jobs WHERE stage='command_meeting' AND input->>'commandId'=%s ORDER BY created_at DESC,id DESC LIMIT 2",(cid,));preparations=list(cur.fetchall())
            documents=[];versions=[];meeting=None;meeting_issue=None;prep=None
            if len(preparations)>1:
                meeting_issue='Multiple preparation jobs are linked to this assignment. The meeting destination needs reconciliation; no job was selected or restarted.'
            elif preparations:
                prep=preparations[0]
                raw_meeting=obj(prep['input']).get('meetingId')
                if prep['ticker']!=command['ticker']:
                    meeting_issue='The preparation job company does not match this assignment. Meeting sources and links are withheld.'
                elif type(raw_meeting) not in (str,int) or not re.fullmatch(r'[1-9][0-9]{0,9}',str(raw_meeting)) or int(raw_meeting)>2147483647:
                    meeting_issue='The saved preparation has no valid meeting identity. Sources and links are withheld until it is reconciled.'
                else:
                    cur.execute('SELECT m.id,c.ticker FROM mp_meetings m JOIN mp_companies c ON c.id=m.company_id WHERE m.id=%s',(int(raw_meeting),));saved_meeting=cur.fetchone()
                    if not saved_meeting or saved_meeting['ticker']!=command['ticker']:
                        meeting_issue='The saved meeting could not be verified for this company. Meeting sources and links are withheld.'
                    else:meeting=saved_meeting['id']
                if meeting:
                    cur.execute('SELECT filename,doc_type FROM mp_documents WHERE meeting_id=%s ORDER BY upload_order',(meeting,));documents=[dict(r) for r in cur.fetchall()]
                    cur.execute('SELECT id,version,status,created_at,generation_model FROM mp_question_sets WHERE meeting_id=%s ORDER BY version DESC,id DESC',(meeting,));versions=[dict(r) for r in cur.fetchall()]
                checkpoint=obj(prep['result']) or {};prep={k:prep[k] for k in ('id','status','error','created_at','updated_at')}|{'stage':checkpoint.get('stage'),'completed':checkpoint.get('completed'),'total':checkpoint.get('total'),'cachedSources':checkpoint.get('cachedSources',sum(bool(x and x.get('_cacheHit')) for x in checkpoint.get('analyses',[])))}
            reports=[]
            if result.get('topic'):
                cur.execute("SELECT id,status,activity_type,updated_at,(NULLIF(BTRIM(output->>'synthesisMarkdown'),'') IS NOT NULL) AS has_report FROM analyst_activities WHERE ticker=%s AND input->>'topic'=%s ORDER BY created_at DESC LIMIT 10",(command['ticker'],result['topic']))
                reports=[dict(r) for r in cur.fetchall()]
            cur.execute("SELECT id,status,error FROM mp_jobs WHERE stage='evidence_amendment' AND input->>'commandId'=%s ORDER BY created_at DESC LIMIT 1",(cid,));proposal=cur.fetchone()
            cur.execute("SELECT status,error FROM mp_jobs WHERE stage='command_proposal_dispatch' AND input->>'commandId'=%s ORDER BY created_at DESC LIMIT 1",(cid,));dispatch=cur.fetchone()
            delivery=delivery_contract(payload,command,collection,result.get('savedSources'),prep,versions,reports,proposal,dispatch,meeting_issue)
            response=jsonify(command={k:command[k] for k in ('id','ticker','status','error','created_at','updated_at')},assignment=payload,collection=collection,sourceReuse=result.get('savedSources',False),selection=obj(selection['value']) if selection else None,preparation=prep,meetingId=meeting,meetingIssue=meeting_issue,documents=documents,versions=versions,reports=reports,proposal=proposal,delivery=delivery)
            response.headers['Cache-Control']='private, no-store';return response

    @bp.route('/api/research/assignments/<cid>/meeting-versions/<int:version_id>', methods=['GET'])
    def meeting_version(cid, version_id):
        """Read one retained version through its exact assignment, never latest."""
        try:
            uuid.UUID(cid)
        except ValueError:
            return jsonify(error='Invalid assignment'), 400
        with get_db() as (_, cur):
            cur.execute("""
                SELECT q.id, q.meeting_id, q.version, q.status, q.topics_json,
                       q.created_at, q.generation_model, c.ticker
                FROM mp_jobs command
                JOIN mp_jobs prep ON prep.stage='command_meeting'
                    AND prep.input->>'commandId'=command.id::text AND prep.ticker=command.ticker
                JOIN mp_meetings m ON m.id::text=prep.input->>'meetingId'
                JOIN mp_companies c ON c.id=m.company_id AND c.ticker=command.ticker
                JOIN mp_question_sets q ON q.meeting_id=m.id
                WHERE command.id=%s AND command.stage='collection_control' AND q.id=%s
                  AND (SELECT COUNT(*) FROM mp_jobs sibling WHERE sibling.stage='command_meeting' AND sibling.input->>'commandId'=command.id::text)=1
            """, (cid, version_id))
            row=cur.fetchone()
            if not row:
                response=jsonify(error='Saved meeting version not found for this assignment')
                response.status_code=404
            else:
                data=dict(row)
                raw=data.pop('topics_json')
                try:
                    topics=json.loads(raw) if isinstance(raw,str) else raw
                    if not isinstance(topics,list):
                        raise ValueError('Invalid stored topics')
                except (ValueError,TypeError):
                    response=jsonify(error='Saved meeting version content is unavailable')
                    response.status_code=409
                else:
                    response=jsonify(assignmentId=cid,questionSet=data|{'topics':topics})
            response.headers['Cache-Control']='private, no-store'
            return response
