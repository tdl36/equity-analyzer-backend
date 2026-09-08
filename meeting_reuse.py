"""Create a new managed brief from immutable originals of a completed pack."""
import copy,json,uuid
from command_thesis_bridge import obj,file_hash
from meeting_command_plan import meeting_options,profile_instruction,FOCUSES

def create(get_db,base_id,data):
    request_id=str(uuid.UUID(data.get('requestId','')))
    options=meeting_options(data)
    identity={'baseJobId':base_id,'options':options}
    jid=str(uuid.uuid5(uuid.UUID(request_id),'saved-meeting-pack'))
    with get_db(commit=True) as (_,cur):
        cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('meeting-reuse:'+request_id,))
        cur.execute('SELECT input FROM mp_jobs WHERE id=%s',(request_id,));old=cur.fetchone()
        if old:
            if obj(old['input']).get('reuseIdentity')!=identity:raise ValueError('This request ID belongs to different meeting settings.')
            return {'commandId':request_id,'jobId':jid,'reused':True}
        cur.execute("SELECT input,status FROM mp_jobs WHERE id=%s AND stage='command_meeting'",(base_id,));base=cur.fetchone()
        if not base or base['status']!='done':raise ValueError('Choose a completed meeting pack to reuse.')
        inp=copy.deepcopy(obj(base['input']))
        cur.execute("SELECT input FROM mp_jobs WHERE id=%s AND stage='collection_control'",(inp['commandId'],));command=cur.fetchone()
        if not command:raise ValueError('Original assignment is unavailable.')
        payload=copy.deepcopy(obj(command['input'])['payload'])
        cur.execute('SELECT * FROM mp_documents WHERE meeting_id=%s ORDER BY upload_order',(inp['meetingId'],));rows={r['id']:dict(r) for r in cur.fetchall()}
        docs=inp['docs']
        if not 1<=len(docs)<=20:raise ValueError('Saved pack has no eligible verified source set.')
        for d in docs:
            if d['id'] not in rows or file_hash(rows[d['id']])!=d['sha256']:raise ValueError('A saved original changed or is missing. Restore it before reuse.')
        instruction=(profile_instruction(options)+' '.join(FOCUSES[f] for f in options['focuses'])+' '+options['note']+
                     f" Use only the verified saved originals from {payload['since']} through {payload['until']}. No fresh search was performed; disclose stale or missing coverage.")
        payload.update(meetingPrep=options,instruction=instruction,autoProposal=False,coordinated=False)
        cur.execute('SELECT company_id FROM mp_meetings WHERE id=%s',(inp['meetingId'],));company=cur.fetchone()
        if not company:raise ValueError('Original meeting is unavailable.')
        cur.execute("INSERT INTO mp_meetings(company_id,meeting_date,meeting_type,notes) VALUES(%s,%s,'management',%s) RETURNING id",(company['company_id'],options['meetingDate'],instruction));mid=cur.fetchone()['id']
        for i,d in enumerate(docs):
            original=rows[d['id']]
            cur.execute('INSERT INTO mp_documents(meeting_id,filename,file_data,doc_type,extracted_text,upload_order,file_size,token_estimate) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id',
                        (mid,d['filename'],original['file_data'],original.get('doc_type','document'),original.get('extracted_text',''),i,original.get('file_size'),original.get('token_estimate')))
            d['id']=cur.fetchone()['id']
        inp.update(commandId=request_id,meetingId=mid,docs=docs,meetingProfile={k:options[k] for k in ('format','audience') if k in options},
                   timeframe=payload['since']+' through '+payload['until']+'. Meeting assignment: '+instruction,recoveryAttempts=0)
        inp.pop('workerToken',None)
        cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result) VALUES(%s,'collection_control',%s,'applied',%s::jsonb,%s::jsonb)",
                    (request_id,inp['ticker'],json.dumps({'action':'saved_meeting','payload':payload,'reuseIdentity':identity}),json.dumps({'savedSources':True,'baseJobId':base_id,'topic':'Saved-source meeting brief'})))
        cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,'command_meeting',%s,'queued',%s::jsonb)",(jid,inp['ticker'],json.dumps(inp,default=str)))
    return {'commandId':request_id,'jobId':jid,'meetingId':mid,'reused':False}
