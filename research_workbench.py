"""Versioned analyst work records with source/answer snapshots and due-work inventory."""
import hashlib,json,re,uuid
from datetime import date
from flask import Blueprint,jsonify,request
from driver_bridge import calculate

def text(v,k,cap=6000):
    if not isinstance(v,str) or not v.strip() or len(v)>cap:raise ValueError('Enter '+k)
    return v.strip()

def validate(data):
    if not isinstance(data,dict):raise ValueError('Work record required.')
    rid=str(uuid.UUID(data.get('requestId','')));ident=str(uuid.UUID(data.get('id','')))
    if type(data.get('revision')) is not int or data['revision']<0:raise ValueError('Reload the work record.')
    b=data.get('body',{})
    if not isinstance(b,dict) or b.get('kind') not in ('model','followup','underweight'):raise ValueError('Choose a work type.')
    out={'kind':b['kind'],**{k:text(b.get(k),k,200 if k in ('title','owner') else 6000) for k in ('title','owner','rationale','nextAction')}}
    if b.get('status') not in ('open','reviewed','closed'):raise ValueError('Choose a status.')
    out['status']=b['status'];out['dueDate']=date.fromisoformat(b.get('dueDate','')).isoformat()
    if out['status']!='open':out['outcome']=text(b.get('outcome'),'review outcome')
    if type(b.get('caseRevision')) is not int or b['caseRevision']<1:raise ValueError('Save the investment case first.')
    out['caseRevision']=b['caseRevision'];out['assumptionId']=str(uuid.UUID(b.get('assumptionId','')))
    if b['kind']=='model':
        out['inputs']=b.get('inputs');out['calculation']=calculate(out['inputs'])
        out['inputs']={k:out['inputs'][k] for k in ('revenueMillions','sharesMillions','beforeMarginPct','afterMarginPct','taxPct','baselineEPS','multiple','referencePrice','currency','period','basis','asOf')}
        out['sourceId']=b.get('sourceId');out['sourceHash']=b.get('sourceHash')
        if type(out['sourceId']) is not int or out['sourceId']<1:raise ValueError('Choose an original source.')
        out['passage']=text(b.get('passage'),'exact supporting passage',12000)
        if not isinstance(out['sourceHash'],str) or not re.fullmatch('[a-f0-9]{64}',out['sourceHash']):raise ValueError('Reload sources.')
    elif b['kind']=='followup':
        if type(b.get('answerId')) is not int or b['answerId']<1:raise ValueError('Choose a recorded answer.')
        out['answerId']=b['answerId'];out['answerHash']=b.get('answerHash')
        if b.get('resolution') not in ('resolved','partial','unresolved'):raise ValueError('Choose how far the answer resolves the issue.')
        out['resolution']=b['resolution']
    else:
        out['mandate']=text(b.get('mandate'),'mandate',200);out['benchmark']=text(b.get('benchmark'),'benchmark',200)
        out['asOf']=date.fromisoformat(b.get('asOf','')).isoformat()
        if out['asOf']>date.today().isoformat():raise ValueError('Holdings date cannot be future dated.')
        if b.get('reason') not in ('valuation','quality','recovery','constraint','research_gap'):raise ValueError('Choose a nonownership reason.')
        out['reason']=b['reason'];out['valuationAssessment']=text(b.get('valuationAssessment'),'current valuation assessment')
        from decimal import Decimal,InvalidOperation
        for k in ('holdingPct','benchmarkPct'):
            try:n=Decimal(str(b.get(k)))
            except InvalidOperation:raise ValueError('Enter '+k)
            if not n.is_finite() or not 0<=n<=100:raise ValueError('Invalid '+k)
            out[k]=str(n)
        out['activeWeightPct']=str(Decimal(out['holdingPct'])-Decimal(out['benchmarkPct']))
        if Decimal(out['activeWeightPct']) > 0:
            raise ValueError('Portfolio weight exceeds benchmark weight. This is not an underweight review.')
        conditions = b.get('reviewConditions', [])
        if not isinstance(conditions, list) or len(conditions) > 12:
            raise ValueError('Use up to 12 reconsideration conditions.')
        out['reviewConditions'] = []
        seen = set()
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError('Invalid reconsideration condition.')
            ident = str(uuid.UUID(condition.get('id', '')))
            if ident in seen:
                raise ValueError('Condition IDs must be unique.')
            seen.add(ident)
            category = condition.get('category')
            state = condition.get('state')
            if category not in ('fundamental', 'valuation', 'constraint', 'research_gap') or state not in ('unassessed', 'not_met', 'partly_met', 'met'):
                raise ValueError('Choose a valid condition type and assessment.')
            item = dict(id=ident, category=category, state=state,
                        trigger=text(condition.get('trigger'), 'reconsideration condition'))
            for key in ('evidence', 'sourceReference'):
                value = condition.get(key, '')
                if not isinstance(value, str) or len(value) > 6000:
                    raise ValueError('Invalid condition evidence.')
                item[key] = value.strip()
            if state != 'unassessed' and not item['evidence']:
                raise ValueError('Explain the evidence behind each assessed condition.')
            out['reviewConditions'].append(item)
        decision = b.get('reviewDecision', 'pending')
        if decision not in ('pending', 'maintain', 'investigate', 'propose_change'):
            raise ValueError('Choose a review decision.')
        if conditions and out['status'] != 'open' and decision == 'pending':
            raise ValueError('Record your underweight review decision before completing it.')
        out['reviewDecision'] = decision

    return rid,ident,data['revision'],out

def answer_hash(row):
    return hashlib.sha256(json.dumps(dict(row),sort_keys=True,default=str).encode()).hexdigest()

def create_blueprint(get_db):
    bp=Blueprint('research_workbench',__name__)
    def ensure():
        with get_db(commit=True) as (_,c):
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('research-work-schema',))
            c.execute('''CREATE TABLE IF NOT EXISTS research_work_versions(id TEXT NOT NULL,ticker TEXT NOT NULL,revision INTEGER NOT NULL,
                request_id TEXT UNIQUE NOT NULL,fingerprint TEXT NOT NULL,body JSONB NOT NULL,created_at TIMESTAMPTZ DEFAULT NOW(),PRIMARY KEY(id,revision))''')
    @bp.route('/api/research/workbench/<ticker>',methods=['GET','POST'])
    def work(ticker):
        if not re.fullmatch('[A-Z0-9][A-Z0-9.-]{0,19}',ticker):return jsonify(error='Choose a ticker.'),400
        if request.method=='POST':
            try:rid,ident,revision,body=validate(request.get_json(silent=True))
            except (ValueError,TypeError,AttributeError) as e:return jsonify(error=str(e)),400
        ensure()
        if request.method=='GET':
            with get_db() as (_,c):
                c.execute('SELECT DISTINCT ON(id) id,revision,body,created_at FROM research_work_versions WHERE ticker=%s ORDER BY id,revision DESC',(ticker,));rows=[dict(r) for r in c.fetchall()]
                c.execute('''SELECT d.id,d.filename,encode(sha256(convert_to(d.extracted_text,'UTF8')),'hex') AS hash
                    FROM mp_documents d JOIN mp_meetings m ON m.id=d.meeting_id JOIN mp_companies p ON p.id=m.company_id
                    WHERE upper(p.ticker)=%s AND length(trim(coalesce(d.extracted_text,'')))>0 ORDER BY d.id DESC LIMIT 100''',(ticker,));sources=[dict(r) for r in c.fetchall()]
                c.execute('''SELECT q.id,q.question,q.response_notes,q.meeting_id,m.meeting_date FROM mp_past_questions q
                    JOIN mp_meetings m ON m.id=q.meeting_id AND m.company_id=q.company_id JOIN mp_companies p ON p.id=q.company_id
                    WHERE upper(p.ticker)=%s AND q.status IN ('answered','resolved') AND length(trim(coalesce(q.response_notes,'')))>0
                    AND m.meeting_date<=CURRENT_DATE ORDER BY m.meeting_date DESC,q.id DESC LIMIT 100''',(ticker,));answers=[dict(r) for r in c.fetchall()]
                for a in answers:a['hash']=answer_hash(a)
                for row in rows:
                    b=row['body']
                    if b['kind']=='model':
                        current=next((s for s in sources if s['id']==b['sourceId']),None)
                        row['evidenceState']='unchanged_saved_text' if current and current['hash']==b['sourceHash'] else 'changed' if current else 'not_in_current_choices'
                    elif b['kind']=='followup':
                        current=next((a for a in answers if a['id']==b['answerId']),None)
                        row['evidenceState']='unchanged_recorded_answer' if current and current['hash']==b['answerHash'] else 'changed' if current else 'not_in_current_choices'
            r=jsonify(records=rows,sources=sources,answers=answers,scope='Latest 100 source and answer choices. Records are analyst work, not executed portfolio changes.');r.headers['Cache-Control']='no-store';return r
        fp=hashlib.sha256(json.dumps([ticker,ident,revision,body],sort_keys=True).encode()).hexdigest()
        with get_db(commit=True) as (_,c):
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('work-request:'+rid,))
            c.execute('SELECT fingerprint,revision FROM research_work_versions WHERE request_id=%s',(rid,));old=c.fetchone()
            if old:
                if old['fingerprint']!=fp:return jsonify(error='Request ID reused for different work.'),409
                return jsonify(id=ident,revision=old['revision'],replayed=True)
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('work:'+ident,))
            c.execute('SELECT ticker,revision,body FROM research_work_versions WHERE id=%s ORDER BY revision DESC LIMIT 1',(ident,));old=c.fetchone()
            if (old and (old['ticker']!=ticker or old['revision']!=revision or old['body']['kind']!=body['kind'])) or (not old and revision):return jsonify(error='Work changed. Reload before saving.'),409
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('investment-case:'+ticker,))
            c.execute('SELECT revision,body FROM investment_case_versions WHERE ticker=%s ORDER BY revision DESC LIMIT 1',(ticker,));case=c.fetchone()
            if not case or case['revision']!=body['caseRevision']:return jsonify(error='Investment case changed. Rebase and review before saving.'),409
            assumption=next((a for a in case['body'].get('assumptions',[]) if a['id']==body['assumptionId']),None)
            if not assumption:return jsonify(error='Assumption no longer exists.'),409
            body['assumptionSnapshot']=assumption
            if body['kind']=='model':
                c.execute('''SELECT d.filename,d.extracted_text FROM mp_documents d JOIN mp_meetings m ON m.id=d.meeting_id
                    JOIN mp_companies p ON p.id=m.company_id WHERE d.id=%s AND upper(p.ticker)=%s FOR SHARE OF d''',(body['sourceId'],ticker));source=c.fetchone()
                if not source or hashlib.sha256((source['extracted_text'] or '').encode()).hexdigest()!=body['sourceHash'] or body['passage'] not in source['extracted_text']:return jsonify(error='Source changed or passage does not match the saved extraction.'),409
                body['sourceFilename']=source['filename']
            elif body['kind']=='followup':
                c.execute('''SELECT q.id,q.question,q.response_notes,q.meeting_id,m.meeting_date FROM mp_past_questions q JOIN mp_meetings m ON m.id=q.meeting_id AND m.company_id=q.company_id
                    JOIN mp_companies p ON p.id=q.company_id WHERE q.id=%s AND upper(p.ticker)=%s AND q.status IN ('answered','resolved') AND length(trim(coalesce(q.response_notes,'')))>0 AND m.meeting_date<=CURRENT_DATE FOR SHARE OF q''',(body['answerId'],ticker));answer=c.fetchone()
                if not answer or answer_hash(answer)!=body['answerHash']:return jsonify(error='Answer changed. Reload and review its wording.'),409
                body['answerSnapshot']={k:str(v) if k=='meeting_date' else v for k,v in answer.items()}
            c.execute('INSERT INTO research_work_versions(id,ticker,revision,request_id,fingerprint,body) VALUES(%s,%s,%s,%s,%s,%s::jsonb)',(ident,ticker,revision+1,rid,fp,json.dumps(body)))
        return jsonify(id=ident,revision=revision+1),201
    @bp.get('/api/research/workbench-queue')
    def queue():
        ensure()
        with get_db() as (_,c):
            c.execute('''SELECT id,ticker,revision,body FROM
                (SELECT DISTINCT ON(id) * FROM research_work_versions ORDER BY id,revision DESC) latest
                WHERE body->>'status'='open' ORDER BY body->>'dueDate',ticker LIMIT 200''')
            work=[dict(r) for r in c.fetchall()]
            c.execute("SELECT to_regclass('research_decisions') AS name")
            issues=[]
            if c.fetchone()['name']:
                c.execute('''SELECT d.id,d.ticker,d.revision,d.body FROM research_decisions d
                    WHERE coalesce(d.body->>'reviewDate','')<>'' AND NOT EXISTS
                    (SELECT 1 FROM research_decisions n WHERE n.ticker=d.ticker AND n.body->>'supersedes'=d.id)
                    ORDER BY d.body->>'reviewDate',d.ticker LIMIT 200''')
                issues=[dict(r) for r in c.fetchall()]
        today=date.today().isoformat()
        for row in work:row['dueDate']=row['body']['dueDate'];row['title']=row['body']['title'];row['kind']=row['body']['kind']
        for row in issues:row['dueDate']=row['body']['reviewDate'];row['title']=row['body'].get('issue') or row['body']['decision'];row['kind']='issue_review'
        rows=sorted(work+issues,key=lambda r:(r['dueDate'],r['ticker']))
        for row in rows:row['due']=row['dueDate']<=today
        r=jsonify(items=rows,asOf=today,scope='Live date-based queue, up to 200 work records and 200 issue reviews. No external reminders or semantic event monitoring.');r.headers['Cache-Control']='no-store';return r

    @bp.get('/api/research/workbench/<ticker>/<ident>/history')
    def history(ticker,ident):
        ensure()
        with get_db() as (_,c):
            c.execute('SELECT revision,body,created_at FROM research_work_versions WHERE ticker=%s AND id=%s ORDER BY revision DESC LIMIT 100',(ticker,ident))
            rows=[dict(r) for r in c.fetchall()]
        r=jsonify(versions=rows,scope='Latest 100 immutable revisions.');r.headers['Cache-Control']='no-store';return r
    return bp
