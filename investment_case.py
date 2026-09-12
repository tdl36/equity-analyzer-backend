"""User-authored investment cases with immutable revisions and deterministic scenarios."""
import hashlib
import copy
import json
import re
import threading
import uuid
from decimal import Decimal, InvalidOperation
from flask import Blueprint, jsonify, request


def case_context_hash(body):
    context={k:body.get(k,'') for k in ('thesis','variantView','marketBaseline','changeConditions','scenarios')}
    context['assumptions']=[{k:a.get(k,'') for k in ('id','claim','evidenceType')} for a in body.get('assumptions',[])]
    return hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()


def source_change(body, job, selection):
    """Bind a user-selected assumption field to an immutable reviewed excerpt snapshot.

    Passage matching proves provenance at generation, not currentness or truth.
    This function never changes the original thesis or the amendment's decision.
    """
    if not job or job['status'] not in ('awaiting_approval', 'applied'):
        raise ValueError('This source proposal is no longer available for review.')
    result=job['result'] or {}
    change=next((c for c in result.get('changes',[]) if c['id']==selection['changeId']),None)
    if not change or not change.get('passageMatched') or not change.get('reviewPassed'):
        raise ValueError('The selected change has unresolved evidence or review findings.')
    sources={s['id']:s for s in result.get('sources',[])}
    refs=[{**e,'source':sources[e['sourceId']]} for e in change.get('evidence',[])
          if e.get('status')=='passage_matched' and e.get('sourceId') in sources]
    if not refs:raise ValueError('No matching source snapshot is available.')
    merged=copy.deepcopy(body)
    assumption=next((a for a in merged.get('assumptions',[]) if a['id']==selection['assumptionId']),None)
    if not assumption:raise ValueError('The assumption no longer exists.')
    field=selection['field']
    if change.get('assumptionId'):
        if change['assumptionId']!=selection['assumptionId'] or change['field']!=field:
            raise ValueError('This proposal belongs to a different assumption or field.')
        if change.get('caseContextHash')!=case_context_hash(body) or assumption.get(field,'')!=change['before']:
            raise ValueError('The investment case changed after generation. Prepare a fresh proposal before accepting this edit.')
    before=assumption.get(field,'');after=text(change['after'],'Proposed wording')
    if before==after:raise ValueError('This wording is already in the selected field.')
    assumption[field]=after
    merged.setdefault('evidenceLinks',[]).append({
        'assumptionId':assumption['id'],'field':field,'before':before,'after':after,
        'jobId':str(job['id']),'changeId':change['id'],'reason':change.get('reason',''),
        'evidence':refs,'provenance':'Passage matched and model reviewed at proposal generation; analyst accepted.'})
    return merged


def text(value, label, maximum=12000):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f'{label} must be text of at most {maximum} characters.')
    return value.strip()


def validate(data):
    if not isinstance(data, dict):
        raise ValueError('An investment case is required.')
    result={k:text(data.get(k,''),k) for k in ('thesis','variantView','marketBaseline','changeConditions')}
    assumptions=data.get('assumptions',[])
    if not isinstance(assumptions,list) or len(assumptions)>30:
        raise ValueError('Use up to 30 explicit assumptions.')
    result['assumptions']=[]
    seen=set()
    for a in assumptions:
        if not isinstance(a,dict):raise ValueError('Invalid assumption.')
        ident=str(uuid.UUID(a.get('id','')))
        if ident in seen:raise ValueError('Assumption IDs must be unique.')
        seen.add(ident)
        item={'id':ident,**{k:text(a.get(k,''),k) for k in ('claim','support','contrary','nextTest','sourceReference')}}
        if not item['claim']:raise ValueError('Each assumption needs a claim.')
        kind=a.get('evidenceType','interpretation')
        if kind not in ('management_statement','reported_fact','broker_estimate','interpretation'):
            raise ValueError('Choose a recognized evidence type.')
        item['evidenceType']=kind
        result['assumptions'].append(item)
    if 'signals' in data:
        from case_signals import validate as validate_signals
        result['signals']=validate_signals(data['signals'],seen)
    result['scenarios']=data.get('scenarios',{})
    scenario_bridge(result['scenarios'])
    return result


def scenario_bridge(data):
    if not isinstance(data,dict):raise ValueError('Invalid scenario inputs.')
    if not data:return {}
    def number(key):
        value=data.get(key)
        if isinstance(value,bool) or not isinstance(value,(str,int,float)) or len(str(value))>40:
            raise ValueError(f'Enter {key}.')
        try:n=Decimal(str(value))
        except InvalidOperation:raise ValueError(f'Invalid {key}.')
        if not n.is_finite() or abs(n)>Decimal('1000000000'):raise ValueError(f'Invalid {key}.')
        return n
    price=number('referencePrice')
    if price<=0:raise ValueError('Reference price must be positive.')
    for key in ('period','currency','asOf'):
        if not text(data.get(key,''),key,100):raise ValueError(f'Enter {key}.')
    from datetime import date
    date.fromisoformat(data['asOf'])
    result={}
    for name in ('bear','base','bull'):
        eps=number(name+'EPS');multiple=number(name+'PE')
        if eps<=0 or multiple<=0:raise ValueError('EPS and P/E must be positive; this method is unsuitable for loss-making scenarios.')
        target=eps*multiple
        result[name]={'impliedPrice':str(target.quantize(Decimal('.01'))),
                      'priceReturnPct':str(((target/price-1)*100).quantize(Decimal('.01')))}
    return result


def create_blueprint(get_db):
    bp=Blueprint('investment_case',__name__)
    schema_lock=threading.Lock();ready=False
    def ensure():
        nonlocal ready
        with schema_lock:
            if ready:return
            with get_db(commit=True) as (_,cur):
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('investment-case-schema',))
                cur.execute('''CREATE TABLE IF NOT EXISTS investment_case_versions (
                  ticker TEXT NOT NULL, revision INTEGER NOT NULL, request_id TEXT UNIQUE NOT NULL,
                  payload_hash TEXT NOT NULL, body JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW(),
                  PRIMARY KEY(ticker,revision))''')
            ready=True
    @bp.get('/api/research/investment-case/<ticker>/saved-thesis-draft')
    def saved_thesis_draft(ticker):
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,19}',ticker):return jsonify(error='Invalid ticker'),400
        from thesis_baseline import draft
        with get_db() as (_,cur):
            cur.execute('SELECT analysis FROM portfolio_analyses WHERE ticker=%s',(ticker,))
            row=cur.fetchone()
        if not row:return jsonify(error='No saved thesis found for this company.'),404
        try:
            analysis=row['analysis']
            if isinstance(analysis,str):analysis=json.loads(analysis)
            return jsonify(**draft(ticker,analysis))
        except (ValueError,TypeError) as exc:return jsonify(error=str(exc)),409

    @bp.get('/api/research/investment-case/<ticker>/source-proposals')
    def source_proposals(ticker):
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker):return jsonify(error='Choose a valid ticker.'),400
        with get_db() as (_,cur):
            cur.execute("SELECT id,status,result,error,created_at,input->>'target' AS target FROM mp_jobs WHERE ticker=%s AND stage='evidence_amendment' AND (status IN ('awaiting_approval','applied') OR (input->>'target'='investment_case' AND status IN ('queued','running','failed','dismissed'))) ORDER BY created_at DESC LIMIT 30",(ticker,))
            rows=[dict(r) for r in cur.fetchall()]
            cur.execute('SELECT filename FROM document_files WHERE ticker=%s ORDER BY filename',(ticker,))
            documents=[r['filename'] for r in cur.fetchall()]
        response=jsonify(proposals=rows,documents=documents);response.headers['Cache-Control']='no-store';return response
    @bp.route('/api/research/investment-case/<ticker>',methods=['GET','POST'])
    def case(ticker):
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker):return jsonify(error='Choose a valid ticker.'),400
        ensure()
        if request.method=='GET':
            with get_db() as (_,cur):
                cur.execute('SELECT revision,body,created_at FROM investment_case_versions WHERE ticker=%s ORDER BY revision DESC LIMIT 100',(ticker,))
                rows=[dict(r) for r in cur.fetchall()]
            response=jsonify(ticker=ticker,revision=rows[0]['revision'] if rows else 0,
                body=rows[0]['body'] if rows else {},versions=rows,
                bridge=scenario_bridge(rows[0]['body'].get('scenarios',{})) if rows else {})
            response.headers['Cache-Control']='no-store';return response
        try:
            d=request.get_json(silent=True)
            if not isinstance(d,dict):raise ValueError('Invalid request.')
            ident=str(uuid.UUID(d.get('requestId','')))
            revision=d.get('revision')
            if type(revision)!=int or revision<0:raise ValueError('Load the current revision before saving.')
            mode=d.get('mode','save')
            if mode=='save':operation={'body':validate(d.get('body'))}
            elif mode=='restore':
                target=d.get('sourceRevision')
                if type(target)!=int or target<1:raise ValueError('Choose a saved revision.')
                operation={'sourceRevision':target}
            elif mode=='source_change':
                selection=d.get('sourceChange',{})
                if not isinstance(selection,dict):raise ValueError('Choose a source change.')
                selection={'jobId':str(uuid.UUID(selection.get('jobId',''))),
                           'assumptionId':str(uuid.UUID(selection.get('assumptionId',''))),
                           'changeId':text(selection.get('changeId'),'Change ID',100),
                           'field':selection.get('field')}
                if selection['field'] not in ('support','contrary','nextTest'):
                    raise ValueError('Choose supporting evidence, contrary evidence or next test.')
                operation={'sourceChange':selection}
            else:raise ValueError('Unknown save operation.')
        except (ValueError,TypeError,AttributeError) as e:return jsonify(error=str(e)),400
        digest=hashlib.sha256(json.dumps({'ticker':ticker,'revision':revision,**({} if mode=='save' else {'mode':mode}),**operation},sort_keys=True).encode()).hexdigest()
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('investment-case-request:'+ident,))
            cur.execute('SELECT ticker,revision,payload_hash,body FROM investment_case_versions WHERE request_id=%s',(ident,));receipt=cur.fetchone()
            if receipt:
                if receipt['payload_hash']!=digest:return jsonify(error='This request ID was already used for different edits.'),409
                return jsonify(ticker=ticker,revision=receipt['revision'],body=receipt['body'],bridge=scenario_bridge(receipt['body'].get('scenarios',{})),replayed=True)
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('investment-case:'+ticker,))
            cur.execute('SELECT revision,body FROM investment_case_versions WHERE ticker=%s ORDER BY revision DESC LIMIT 1',(ticker,));current=cur.fetchone()
            if (current['revision'] if current else 0)!=revision:
                return jsonify(error='A newer investment case was saved. Your unsaved edits are retained; reload and reconcile them before saving.'),409
            if mode=='save':
                body=operation['body']
                # Evidence metadata can only originate in a server-reviewed operation.
                body['evidenceLinks']=(current['body'].get('evidenceLinks',[]) if current else [])
            elif mode=='restore':
                cur.execute('SELECT body FROM investment_case_versions WHERE ticker=%s AND revision=%s',(ticker,operation['sourceRevision']))
                original=cur.fetchone()
                if not original:return jsonify(error='Saved revision not found.'),404
                body=copy.deepcopy(original['body'])
                body['restoredFromRevision']=operation['sourceRevision']
            else:
                selection=operation['sourceChange']
                cur.execute("SELECT id,status,result FROM mp_jobs WHERE id=%s AND ticker=%s AND stage='evidence_amendment' FOR SHARE",(selection['jobId'],ticker))
                try:body=source_change(current['body'] if current else {},cur.fetchone(),selection)
                except ValueError as e:return jsonify(error=str(e)),409
            cur.execute('INSERT INTO investment_case_versions(ticker,revision,request_id,payload_hash,body) VALUES(%s,%s,%s,%s,%s::jsonb)',(ticker,revision+1,ident,digest,json.dumps(body)))
        return jsonify(ticker=ticker,revision=revision+1,body=body,bridge=scenario_bridge(body.get('scenarios',{})))
    return bp
