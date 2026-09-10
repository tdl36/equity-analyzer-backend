"""User-authored investment cases with immutable revisions and deterministic scenarios."""
import hashlib
import json
import re
import threading
import uuid
from decimal import Decimal, InvalidOperation
from flask import Blueprint, jsonify, request


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
            body=validate(d.get('body'))
        except (ValueError,TypeError,AttributeError) as e:return jsonify(error=str(e)),400
        digest=hashlib.sha256(json.dumps({'ticker':ticker,'revision':revision,'body':body},sort_keys=True).encode()).hexdigest()
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('investment-case-request:'+ident,))
            cur.execute('SELECT ticker,revision,payload_hash FROM investment_case_versions WHERE request_id=%s',(ident,));receipt=cur.fetchone()
            if receipt:
                if receipt['payload_hash']!=digest:return jsonify(error='This request ID was already used for different edits.'),409
                return jsonify(ticker=ticker,revision=receipt['revision'],replayed=True)
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('investment-case:'+ticker,))
            cur.execute('SELECT revision FROM investment_case_versions WHERE ticker=%s ORDER BY revision DESC LIMIT 1',(ticker,));current=cur.fetchone()
            if (current['revision'] if current else 0)!=revision:
                return jsonify(error='A newer investment case was saved. Your unsaved edits are retained; reload and reconcile them before saving.'),409
            cur.execute('INSERT INTO investment_case_versions(ticker,revision,request_id,payload_hash,body) VALUES(%s,%s,%s,%s,%s::jsonb)',(ticker,revision+1,ident,digest,json.dumps(body)))
        return jsonify(ticker=ticker,revision=revision+1,bridge=scenario_bridge(body['scenarios']))
    return bp
