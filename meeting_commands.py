"""Guided conference batches and verified command-to-meeting handoff."""
import base64
import json
import threading
import uuid
from command_thesis_bridge import resolve, file_hash, obj
from meeting_command_plan import batch

_WORKER = threading.Lock()


def job_id(command_id):
    return str(uuid.uuid5(uuid.UUID(command_id), 'meeting-pack'))


def source_documents(bridge, rows):
    if bridge['blocked'] or not 1 <= len(bridge['ready']) <= 20:
        raise ValueError('Meeting prep needs 1–20 verified originals and no missing sources. Narrow the collection or resolve missing originals.')
    docs=[]; total=0
    for source in bridge['ready']:
        matches=[r for r in rows if r['filename']==source['filename']]
        if len(matches)!=1 or file_hash(matches[0])!=source['sha256']:
            raise ValueError('A meeting source changed after the recap. Resolve source integrity before generation.')
        row=matches[0];raw=base64.b64decode(row['file_data'],validate=True);total+=len(raw)
        if total>60_000_000:raise ValueError('Meeting sources exceed 60 MB; narrow the collection.')
        text=''
        if not row['filename'].lower().endswith('.pdf'):
            text=raw.decode('utf-8',errors='strict')
            if row['filename'].lower().endswith(('.html','.htm')):
                from bs4 import BeautifulSoup
                text=BeautifulSoup(text,'html.parser').get_text(' ',strip=True)
            text=text.replace('\x00','')
            if not text.strip():raise ValueError('A source contains no readable text.')
            if len(text)>200_000:raise ValueError('A source exceeds the meeting text limit; narrow the source set.')
        docs.append({'filename':row['filename'],'fileData':row['file_data'],'extractedText':text,
                     'docType':'document','sha256':source['sha256'],'sourceUrl':obj(row.get('metadata')).get('sourceUrl')})
    return docs


def validate_pack(topics, docs):
    names={d['filename'] for d in docs}
    if not isinstance(topics,list) or not 1<=len(topics)<=20:raise ValueError('Meeting output has no valid topic groups.')
    count=0
    for topic in topics:
        if not isinstance(topic,dict) or not isinstance(topic.get('topic'),str) or not topic['topic'].strip():raise ValueError('Meeting topic is incomplete.')
        questions=topic.get('questions')
        if not isinstance(questions,list) or not questions:raise ValueError('Meeting topic has no questions.')
        for q in questions:
            if not isinstance(q,dict) or any(not isinstance(q.get(k),str) or not q[k].strip() for k in ('question','context','source','follow_up_angle')):
                raise ValueError('A meeting question is missing its rationale, source or follow-up.')
            cited=q.get('source_filenames')
            if not isinstance(cited,list) or not cited or any(not isinstance(n,str) or n not in names for n in cited):
                raise ValueError('A meeting question cites an unverified source filename. Retry generation after inspecting the output.')
            if q.get('priority') not in ('high','medium','low'):raise ValueError('Meeting question priority is missing.')
            count+=1
    if not 1<=count<=60:raise ValueError('Meeting output must contain 1–60 questions.')
    return topics


def prepare(get_db, command):
    cid=command['id']; jid=job_id(cid)
    with get_db(commit=True) as (_,cur):
        cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('meeting-command:'+cid,))
        cur.execute('SELECT id FROM mp_jobs WHERE id=%s',(jid,))
        if cur.fetchone():return jid
        bridge=resolve(cur,cid,command['ticker'])
        cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=ANY(%s)',
                    (command['ticker'],[s['filename'] for s in bridge['ready']]))
        docs=source_documents(bridge,list(cur.fetchall()))
        options=obj(command['input'])['payload']['meetingPrep']
        cur.execute("INSERT INTO mp_companies(ticker,name,sector) VALUES(%s,%s,'unknown') ON CONFLICT(ticker) DO UPDATE SET ticker=EXCLUDED.ticker RETURNING id,name,sector",(command['ticker'],command['ticker']))
        company=dict(cur.fetchone())
        cur.execute("SELECT pq.*,m.meeting_date FROM mp_past_questions pq LEFT JOIN mp_meetings m ON m.id=pq.meeting_id WHERE pq.company_id=%s AND m.meeting_date<=%s AND pq.status IN ('asked','answered','resolved') ORDER BY m.meeting_date DESC,pq.created_at DESC LIMIT 30",(company['id'],obj(command['input'])['payload']['until']))
        past=[dict(r) for r in cur.fetchall()]
        notes=bridge['instructions']+'\nSource register (verified originals):\n'+'\n'.join(d['filename']+' | SHA256 '+d['sha256']+' | '+(d.get('sourceUrl') or 'Source URL unavailable') for d in docs)
        cur.execute("INSERT INTO mp_meetings(company_id,meeting_date,meeting_type,notes) VALUES(%s,%s,'conference',%s) RETURNING id",(company['id'],options['meetingDate'],notes))
        mid=cur.fetchone()['id']
        for i,d in enumerate(docs):
            cur.execute('INSERT INTO mp_documents(meeting_id,filename,file_data,doc_type,extracted_text,upload_order,file_size,token_estimate) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id',
                        (mid,d['filename'],d['fileData'],d['docType'],d['extractedText'],i,len(base64.b64decode(d['fileData'])),len(d['extractedText'])//4))
            d['id']=cur.fetchone()['id'];del d['fileData']
        inp={'companyMemoryRequested':True,'commandId':cid,'commandRevision':bridge['revision'],'meetingId':mid,'ticker':command['ticker'],
             'companyName':company['name'],'sector':company['sector'],'docs':docs,'pastQuestions':past,
             'unresolvedQuestions':[q for q in past if q.get('status') not in ('resolved','answered')],
             'timeframe':obj(command['input'])['payload']['since']+' through '+obj(command['input'])['payload']['until']+'. Meeting assignment: '+bridge['instructions'],
             'meetingProfile':{k:options.get(k,default) for k,default in [('format','conference'),('audience','specialist')]}, 'recoveryAttempts':0}
        cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,'command_meeting',%s,'queued',%s::jsonb)",(jid,command['ticker'],json.dumps(inp,default=str)))
    return jid


def drain(get_db,run):
    if not _WORKER.acquire(blocking=False):return
    try:
        from amendment_ownership import worker_session
        # One shared conference worker across server processes; each pack's document
        # analysis has its own bounded concurrency in the existing meeting pipeline.
        with worker_session(get_db,'conference-meeting-worker') as owned:
            if not owned:return
            with get_db(commit=True) as (_,cur):
                cur.execute("SELECT * FROM mp_jobs WHERE stage='command_meeting' AND status IN ('queued','running') ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED")
                row=cur.fetchone()
                if not row:return
                inp=obj(row['input']);attempts=inp.get('recoveryAttempts',0)
                if row['status']=='running':
                    if attempts>=2:
                        cur.execute("UPDATE mp_jobs SET status='failed',error='Automatic recovery limit reached; retry this meeting pack.',updated_at=NOW() WHERE id=%s",(row['id'],));return
                    inp['recoveryAttempts']=attempts+1
                inp['workerToken']=str(uuid.uuid4())
                cur.execute("UPDATE mp_jobs SET status='running',error=NULL,input=%s::jsonb,updated_at=NOW() WHERE id=%s",(json.dumps(inp),row['id']))
            try:
                with get_db() as (_,cur):
                    cur.execute('SELECT id,filename,file_data FROM mp_documents WHERE meeting_id=%s',(inp['meetingId'],))
                    actual={r['id']:dict(r) for r in cur.fetchall()}
                for d in inp['docs']:
                    if d['id'] not in actual or file_hash(actual[d['id']])!=d['sha256']:
                        raise ValueError('Meeting source removed or changed. Existing checkpoints were retained; restore the original before retrying.')
                run(row['id'],inp,obj(row['result']))
            except Exception as exc:
                with get_db(commit=True) as (_,cur):
                    cur.execute("UPDATE mp_jobs SET status='failed',error=%s,updated_at=NOW() WHERE id=%s AND status='running' AND input->>'workerToken'=%s",(str(exc)[:1500],row['id'],inp['workerToken']))
    finally:_WORKER.release()


def create_blueprint(get_db,run,has_key):
    from flask import Blueprint,jsonify,request
    bp=Blueprint('meeting_commands',__name__)
    @bp.route('/api/research/meeting-commands',methods=['GET','POST'])
    def commands():
        if request.method=='POST':
            try:bid,commands=batch(request.get_json(silent=True))
            except (ValueError,TypeError,AttributeError) as exc:return jsonify(error=str(exc)),400
            if not has_key():return jsonify(error='A server research key is required to prepare meeting packs.'),400
            with get_db(commit=True) as (_,cur):
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('meeting-batch:'+bid,))
                # Entire selection validates before any command is inserted.
                for command in commands:
                    cur.execute('SELECT id FROM analysts WHERE %s=ANY(coverage_tickers) LIMIT 1',(command['payload']['ticker'],))
                    if not cur.fetchone():return jsonify(error='Assign a covering analyst to '+command['payload']['ticker']+' first.'),400
                cur.execute("SELECT id,input FROM mp_jobs WHERE input->>'meetingBatchId'=%s",(bid,))
                old={r['id']:obj(r['input']) for r in cur.fetchall()}
                expected={c['id']:{'action':'research_task','meetingBatchId':bid,'payload':c['payload']} for c in commands}
                if old and old!=expected:return jsonify(error='This request ID belongs to a different meeting selection.'),409
                if not old:
                    for command in commands:
                        cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,'collection_control',%s,'queued',%s::jsonb)",
                                    (command['id'],command['payload']['ticker'],json.dumps(expected[command['id']])))
            return jsonify(batchId=bid,commands=[{'id':c['id'],'ticker':c['payload']['ticker']} for c in commands]),202
        with get_db() as (_,cur):
            cur.execute('SELECT coverage_tickers FROM analysts')
            tickers=sorted({t for r in cur.fetchall() for t in (r['coverage_tickers'] or [])})
            cur.execute("""SELECT c.id,c.ticker,c.status,c.error,c.input->'payload' AS options,c.input->>'meetingBatchId' AS batch_id,
                c.result->>'topic' AS topic,m.id AS job_id,m.status AS prep_status,m.error AS prep_error,
                m.input->>'meetingId' AS meeting_id,m.result->>'stage' AS prep_step,
                m.result->>'completed' AS completed,m.result->>'total' AS total,
                c.result->>'meetingIssue' AS meeting_issue
                FROM mp_jobs c LEFT JOIN mp_jobs m ON m.stage='command_meeting' AND m.input->>'commandId'=c.id
                WHERE c.stage='collection_control' AND c.input->'payload'->'meetingPrep' IS NOT NULL ORDER BY c.created_at DESC LIMIT 100""")
            rows=[dict(r) for r in cur.fetchall()]
        return jsonify(tickers=tickers,jobs=rows)

    @bp.route('/api/research/meeting-commands/<jid>/reuse',methods=['POST'])
    def reuse(jid):
        try:
            uuid.UUID(jid)
            if not has_key():return jsonify(error='Server research key is missing.'),400
            from meeting_reuse import create
            result=create(get_db,jid,request.get_json() or {})
        except (ValueError,TypeError,AttributeError,KeyError) as exc:return jsonify(error=str(exc)),400
        threading.Thread(target=drain,args=(get_db,run),daemon=True).start()
        return jsonify(result),202

    @bp.route('/api/research/meeting-commands/<jid>/revision',methods=['POST'])
    def revise(jid):
        try:
            uuid.UUID(jid)
            data=request.get_json() or {}
            reason=data.get('reason','').strip()
            if not reason or len(reason)>1000:raise ValueError('A concise revision reason is required.')
            with get_db(commit=True) as (_,cur):
                cur.execute("SELECT input,result,status FROM mp_jobs WHERE id=%s AND stage='command_meeting' FOR UPDATE",(jid,))
                row=cur.fetchone()
                if not row or row['status']!='done':return jsonify(error='Only completed meeting packs can be revised.'),409
                inp=obj(row['input']);result=obj(row['result']);mid=inp['meetingId']
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('meeting-save:'+str(mid),))
                cur.execute('SELECT id,version FROM mp_question_sets WHERE meeting_id=%s ORDER BY version DESC LIMIT 1',(mid,))
                current=cur.fetchone()
                if not current or current['id']!=data.get('expectedQuestionSetId'):return jsonify(error='Meeting version changed. Reload before revising.'),409
                cur.execute('SELECT filename FROM mp_documents WHERE meeting_id=%s',(mid,))
                topics=validate_pack(data.get('topics'),list(cur.fetchall()))
                cur.execute("INSERT INTO mp_question_sets(meeting_id,version,status,topics_json,synthesis_json,generation_model,generation_tokens) VALUES(%s,%s,'ready',%s,NULL,%s,0) RETURNING id,version",
                            (mid,current['version']+1,json.dumps(topics),'reviewed revision'))
                saved=dict(cur.fetchone())
                result.update(topics=topics,questionSetId=saved['id'],version=saved['version'],revisionReason=reason,previousQuestionSetId=current['id'])
                # Superseded synthesis remains in its immutable prior version, not in the current pack.
                result.pop('synthesis',None)
                cur.execute("UPDATE mp_jobs SET result=%s::jsonb,updated_at=NOW() WHERE id=%s",(json.dumps(result),jid))
                cur.execute("UPDATE mp_past_questions SET status='superseded' WHERE meeting_id=%s AND status='planned'",(mid,))
                cur.execute('SELECT company_id FROM mp_meetings WHERE id=%s',(mid,));company=cur.fetchone()
                for topic in topics:
                    for q in topic['questions']:
                        cur.execute("INSERT INTO mp_past_questions(company_id,meeting_id,question,topic,status) VALUES(%s,%s,%s,%s,'planned')",(company['company_id'],mid,q['question'],topic['topic']))
            return jsonify(questionSetId=saved['id'],version=saved['version'])
        except (ValueError,TypeError,AttributeError) as exc:return jsonify(error=str(exc)),400

    @bp.route('/api/research/meeting-commands/<jid>/retry',methods=['POST'])
    def retry(jid):
        try:uuid.UUID(jid)
        except ValueError:return jsonify(error='Invalid meeting job'),400
        with get_db(commit=True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET status='queued',error=NULL,updated_at=NOW() WHERE id=%s AND stage='command_meeting' AND status='failed' RETURNING id",(jid,))
            if not cur.fetchone():return jsonify(error='Only failed meeting packs can be retried.'),409
        return jsonify(queued=True)

    @bp.route('/api/agent/advance-meeting-commands',methods=['POST'])
    def advance():
        if not has_key():return jsonify(error='Server research key is missing.'),400
        with get_db() as (_,cur):
            cur.execute("""SELECT c.id,c.ticker,c.input FROM mp_jobs c WHERE c.stage='collection_control' AND c.status='applied'
                AND c.input->'payload'->'meetingPrep' IS NOT NULL
                AND NOT EXISTS(SELECT 1 FROM mp_jobs m WHERE m.stage='command_meeting' AND m.input->>'commandId'=c.id)
                ORDER BY c.result->>'meetingCheckedAt' ASC NULLS FIRST,c.created_at ASC LIMIT 3""")
            commands=list(cur.fetchall())
        outcomes=[]
        for command in commands:
            issue=None
            try:jid=prepare(get_db,command)
            except (ValueError,TypeError,KeyError) as exc:issue=str(exc)
            with get_db(commit=True) as (_,cur):
                cur.execute("UPDATE mp_jobs SET result=COALESCE(result,'{}'::jsonb)||jsonb_build_object('meetingIssue',%s::text,'meetingCheckedAt',NOW()::text) WHERE id=%s",(issue,command['id']))
            outcomes.append({'commandId':command['id'],'issue':issue})
        threading.Thread(target=drain,args=(get_db,run),daemon=True).start()
        return jsonify(outcomes=outcomes)
    return bp


def start_server_worker(get_db,run,has_key):
    """Resume durable meeting jobs without requiring the user's Mac heartbeat."""
    import time
    def loop():
        while True:
            try:
                if has_key():drain(get_db,run)
            except Exception as exc:
                print('[meeting worker] Queue check failed: '+type(exc).__name__)
            time.sleep(30)
    threading.Thread(target=loop,daemon=True,name='managed-meeting-queue').start()
