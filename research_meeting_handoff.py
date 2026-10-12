"""Explicit, idempotent transfer of saved diligence questions; no model calls."""
import hashlib
import json
import re
from datetime import date
from flask import Blueprint, jsonify, request
from stock_evidence import missing_value, valid_citations


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode()).hexdigest()


def preview(run):
    if run.get('status')!='complete':raise ValueError('Choose a completed stock report.')
    report=run.get('state',{}).get('report',{})
    rows=report.get('diligence_questions',[])
    if not isinstance(rows,list) or len(rows)>10:raise ValueError('Saved diligence questions have an unsupported format.')
    valid=valid_citations(run)
    questions=[]
    for index,value in enumerate(rows):
        if missing_value(value):continue
        if not isinstance(value,str) or len(value)>1600:raise ValueError('A saved question has an unsupported format.')
        path=f'/diligence_questions/{index}'
        questions.append({'index':index,'question':value,'path':path,'sourceMatched':path in valid})
    return {'runId':run['id'],'ticker':run['ticker'],'createdAt':str(run.get('created_at') or ''),
            'questions':questions,'fingerprint':digest({'id':run['id'],'ticker':run['ticker'],'questions':questions})}


def prepare(run,data):
    if not isinstance(data,dict):raise ValueError('Select questions and a meeting date.')
    pack=preview(run)
    if data.get('fingerprint')!=pack['fingerprint']:raise ValueError('The saved questions changed. Reload before copying them.')
    if 'expectedSources' in data:
        expected=data['expectedSources']
        actual=[{k:s.get(k) for k in ('filename','originalHash')} for s in run.get('sources',[])]
        if not isinstance(expected,list) or not expected or expected!=actual:
            raise ValueError('The shared research source pack changed. Inspect the saved assignment.')
    indices=data.get('indices')
    if not isinstance(indices,list) or not 1<=len(indices)<=10 or any(type(i) is not int for i in indices) or len(set(indices))!=len(indices):
        raise ValueError('Select 1–10 distinct saved questions.')
    available={q['index']:q for q in pack['questions']}
    if any(i not in available for i in indices):raise ValueError('A selected question is unavailable. Reload the report.')
    when=data.get('meetingDate')
    try:
        if not isinstance(when,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',when):raise ValueError()
        date.fromisoformat(when)
    except ValueError:raise ValueError('Choose a valid meeting date.')
    kind=data.get('meetingType')
    if kind not in ('management','expert','conference','other'):raise ValueError('Choose a meeting type.')
    selected=[available[i] for i in sorted(indices)]
    topics=[{'topic':'Saved research · diligence questions','questions':[{
        'question':q['question'],'priority':'high',
        'context':'Copied unchanged from saved stock research. Review factual premises against the original documents before the meeting.',
        'source':f"{pack['ticker']} stock report {pack['runId']} · "+('source matched and model reviewed; not independently verified' if q['sourceMatched'] else 'premise needs source review'),
        'source_filenames':[], 'supporting_quotes':[],
        'follow_up_angle':'Record the actual answer, supporting evidence and any unresolved follow-up.',
        'research_origin':{'runId':pack['runId'],'ticker':pack['ticker'],'path':q['path'],'createdAt':pack['createdAt']}
    } for q in selected]}]
    identity={'runId':pack['runId'],'fingerprint':pack['fingerprint'],'indices':sorted(indices),'meetingDate':when,'meetingType':kind, **({'expectedSources':data['expectedSources']} if 'expectedSources' in data else {})}
    return digest(identity),identity,topics


def save(cur,run,data):
    identity,body,topics=prepare(run,data)
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('research-meeting:'+identity,))
    cur.execute('SELECT meeting_id,question_set_id FROM research_meeting_handoffs WHERE id=%s',(identity,))
    existing=cur.fetchone()
    if existing:
        if existing['meeting_id'] is None:raise ValueError('This copied meeting was deleted. Change the date or question selection to create another.')
        return {'meetingId':existing['meeting_id'],'questionSetId':existing['question_set_id'],'reused':True}
    # Do not replace the established company name with a ticker or generated name.
    cur.execute('INSERT INTO mp_companies(ticker,name) VALUES(%s,%s) ON CONFLICT(ticker) DO UPDATE SET ticker=EXCLUDED.ticker RETURNING id',(run['ticker'],run['ticker']))
    company=cur.fetchone()['id']
    note=f"Questions copied from {run['ticker']} saved stock report {run['id']}. No additional research or source verification was performed. Selected questions are must-ask; record actual answers in meeting mode."
    cur.execute("INSERT INTO mp_meetings(company_id,meeting_date,meeting_type,notes) VALUES(%s,%s,%s,%s) RETURNING id",(company,body['meetingDate'],body['meetingType'],note))
    mid=cur.fetchone()['id']
    cur.execute("INSERT INTO mp_question_sets(meeting_id,version,status,topics_json,generation_model,generation_tokens) VALUES(%s,1,'ready',%s,'saved research transfer',0) RETURNING id",(mid,json.dumps(topics)))
    qid=cur.fetchone()['id']
    for q in topics[0]['questions']:
        cur.execute("INSERT INTO mp_past_questions(company_id,meeting_id,question,topic,status) VALUES(%s,%s,%s,%s,'planned')",(company,mid,q['question'],topics[0]['topic']))
    cur.execute('INSERT INTO research_meeting_handoffs(id,run_id,meeting_id,question_set_id,input) VALUES(%s,%s,%s,%s,%s::jsonb)',(identity,run['id'],mid,qid,json.dumps(body)))
    return {'meetingId':mid,'questionSetId':qid,'reused':False}


def create_blueprint(get_db):
    bp=Blueprint('research_meeting_handoff',__name__)
    @bp.route('/api/research/stock-analysis-run/<rid>/meeting-prep',methods=['GET','POST'])
    def handoff(rid):
        import uuid
        try:uuid.UUID(rid)
        except (ValueError,TypeError):return jsonify(error='Invalid saved report.'),400
        try:
            with get_db(commit=request.method=='POST') as (_,cur):
                cur.execute('SELECT id,ticker,status,state,sources,created_at FROM stock_analysis_runs WHERE id=%s FOR SHARE',(rid,))
                row=cur.fetchone()
                if not row:return jsonify(error='Saved report not found.'),404
                run=dict(row)
                if request.method=='GET':return jsonify(preview(run))
                # Schema setup and all writes are in the same transaction. Reads never create work.
                body=request.get_json(silent=True)
                prepare(run,body)
                if 'expectedSources' in body:
                    from company_research import eligible
                    from research_originals import verified_original
                    for source in run['sources']:
                        cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE',(run['ticker'],source['filename']))
                        originals=cur.fetchall()
                        if len(originals)!=1 or not eligible(originals[0].get('metadata') or {}):
                            raise ValueError('A shared-pack original is missing or restricted. Meeting preparation was not saved.')
                        verified_original(source,dict(originals[0]),source['originalHash'])

                cur.execute("SELECT pg_advisory_xact_lock(hashtext('research-meeting-schema'))")
                cur.execute('''CREATE TABLE IF NOT EXISTS research_meeting_handoffs (
                    id TEXT PRIMARY KEY,run_id TEXT NOT NULL,meeting_id INTEGER REFERENCES mp_meetings(id) ON DELETE SET NULL,
                    question_set_id INTEGER,input JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')
                result=save(cur,run,request.get_json(silent=True))
            return jsonify(result)
        except (ValueError,TypeError,AttributeError) as exc:return jsonify(error=str(exc)),400
    return bp
