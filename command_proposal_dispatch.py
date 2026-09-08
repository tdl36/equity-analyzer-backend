"""Bounded command-to-proposal orchestration; stable IDs fence repeated ticks."""
import json
import uuid


def identifiers(command_id):
    uuid.UUID(command_id)
    return (str(uuid.uuid5(uuid.NAMESPACE_URL,'charlie:command-proposal:'+command_id)),
            str(uuid.uuid5(uuid.NAMESPACE_URL,'charlie:command-dispatch:'+command_id)))


def proposal_input(bridge):
    if bridge['blocked']:raise ValueError('Some recap originals are missing or changed. Open the command comparison to inspect them.')
    if not 1<=len(bridge['ready'])<=10:raise ValueError('Automatic comparison requires 1–10 verified sources. Select a smaller set manually.')
    proposal_id,_=identifiers(bridge['commandId'])
    return {'requestId':proposal_id,'commandId':bridge['commandId'],'commandRevision':bridge['revision'],
            'filenames':[d['filename'] for d in bridge['ready']],'instructions':bridge['instructions']}


def advance(get_db,submit):
    from command_thesis_bridge import resolve
    with get_db() as (_,cur):
        cur.execute("""SELECT c.id,c.ticker FROM mp_jobs c LEFT JOIN mp_jobs d ON d.stage='command_proposal_dispatch' AND d.input->>'commandId'=c.id WHERE c.stage='collection_control'
            AND c.status='applied' AND c.input->>'action'='research_task'
            AND c.input->'payload'->>'autoProposal'='true'
            AND NOT EXISTS(SELECT 1 FROM mp_jobs p WHERE p.stage='evidence_amendment' AND p.input->>'commandId'=c.id)
            ORDER BY d.updated_at ASC NULLS FIRST,c.created_at ASC LIMIT 3""")
        commands=list(cur.fetchall() or [])
    outcomes=[]
    for command in commands:
        proposal_id,dispatch_id=identifiers(command['id'])
        try:
            with get_db() as (_,cur):bridge=resolve(cur,command['id'],command['ticker'])
            payload=proposal_input(bridge)
            response=submit(command['ticker'],payload)
            if isinstance(response,tuple):body,status=response[0],response[1]
            else:body,status=response,response.status_code
            result=body.get_json()
            if status not in (200,202):raise ValueError(result.get('error','Proposal submission did not confirm success'))
            if result.get('jobId')!=proposal_id:raise ValueError('Unexpected proposal receipt; inspect before retrying')
            state='complete';record={'proposalId':proposal_id,'message':'Thesis proposal submitted; review it before applying any edits.'};error=None
        except (ValueError,TypeError,KeyError) as exc:
            state='blocked';record={'proposalId':proposal_id,'message':str(exc)};error=str(exc)
        with get_db(commit=True) as (_,cur):
            cur.execute("""INSERT INTO mp_jobs(id,stage,ticker,status,input,result,error)
                VALUES(%s,'command_proposal_dispatch',%s,%s,%s::jsonb,%s::jsonb,%s)
                ON CONFLICT(id) DO UPDATE SET status=EXCLUDED.status,result=EXCLUDED.result,error=EXCLUDED.error,updated_at=NOW()""",
                (dispatch_id,command['ticker'],state,json.dumps({'commandId':command['id']}),json.dumps(record),error))
        outcomes.append({'commandId':command['id'],'status':state,**record})
    return outcomes
