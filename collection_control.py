"""Authenticated cloud-to-Mac collection commands; local receipts prevent replay."""
import json
import uuid
from flask import Blueprint, jsonify, request

STAGE='collection_control'
KEY='collection_control_snapshot'


def decode(value):
    return json.loads(value) if isinstance(value,str) else value


def command(data):
    if not isinstance(data,dict):raise ValueError('A command is required.')
    try:uuid.UUID(data.get('requestId',''))
    except (ValueError,TypeError,AttributeError):raise ValueError('A valid request ID is required.')
    action=data.get('action');payload=data.get('payload')
    if action not in ('save','trigger','save_batch','cancel','retry','event_refresh','save_trigger') or not isinstance(payload,dict):raise ValueError('Unsupported collection command.')
    if action=='save_trigger':
        return {'action':action,'payload':command({'requestId':data['requestId'],'action':'save','payload':payload})['payload']}
    if action=='event_refresh':
        policy=command({'requestId':data['requestId'],'action':'save','payload':payload.get('policy')})['payload']
        event=payload.get('event')
        if not isinstance(event,dict) or any(not isinstance(event.get(k),str) or not event[k].strip() or len(event[k])>2000 for k in ('id','reason','url')):raise ValueError('A sourced catalyst event is required.')
        if policy['workflow']!='recap':raise ValueError('Catalyst events require a recap destination.')
        return {'action':action,'payload':{'policy':policy,'event':{k:event[k] for k in ('id','reason','url')}}}
    if action=='save_batch':
        policies=payload.get('policies')
        if not isinstance(policies,list) or not 1<=len(policies)<=100:raise ValueError('Choose 1–100 ticker policies.')
        values=[command({'requestId':data['requestId'],'action':'save','payload':p})['payload'] for p in policies]
        if len({p['ticker'] for p in values})!=len(values):raise ValueError('A ticker appears more than once.')
        return {'action':action,'payload':{'policies':values}}
    import re
    ticker=payload.get('ticker')
    if not isinstance(ticker,str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker):raise ValueError('A valid uppercase ticker is required.')
    if action=='save':
        if type(payload.get('hours'))!=int or not 0<=payload['hours']<=8760:raise ValueError('Invalid refresh frequency.')
        if type(payload.get('lookbackDays'))!=int or not 1<=payload['lookbackDays']<=365:raise ValueError('Lookback must be 1–365 days.')
        if type(payload.get('createFolder',False))!=bool:raise ValueError('createFolder must be true or false.')
        if not isinstance(payload.get('enabled'),bool):raise ValueError('enabled must be true or false.')
        if payload.get('workflow') not in ('thesis','note','recap'):raise ValueError('Invalid workflow.')
        if not isinstance(payload.get('instructions',''),str) or len(payload.get('instructions',''))>3000:raise ValueError('Instructions limit is 3,000 characters.')
        kinds=payload.get('kinds')
        if not isinstance(kinds,list) or not kinds or any(k not in ('transcript','broker-report','press-release') for k in kinds):raise ValueError('Choose source types.')
        topic=payload.get('topic','')
        if not isinstance(topic,str) or len(topic)>160 or '/' in topic or '\\' in topic or topic.startswith('.') or (payload['workflow']=='recap' and not topic.strip()):raise ValueError('A plain event folder name is required for recaps.')
        create_folder=payload.get('createFolder',False)
        payload={k:payload.get(k,'') for k in ('ticker','hours','lookbackDays','enabled','workflow','instructions','kinds','topic')}
        payload['createFolder']=create_folder
    elif action in ('cancel','retry'):
        try:uuid.UUID(payload.get('refreshRequestId',''))
        except (ValueError,TypeError,AttributeError):raise ValueError('Choose a valid browser refresh request.')
        payload={'ticker':ticker,'refreshRequestId':payload['refreshRequestId']}
    else:payload={'ticker':ticker}
    return {'action':action,'payload':payload}


def create_blueprint(get_db, is_agent):
    bp=Blueprint('collection_control',__name__)
    @bp.route('/api/collection/control',methods=['GET','POST'])
    def control():
        if request.method=='POST':
            data=request.get_json(silent=True)
            try:value=command(data)
            except ValueError as e:return jsonify(error=str(e)),400
            jid=data['requestId']
            with get_db(commit=True) as (_,cur):
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(STAGE,))
                cur.execute('SELECT stage,input,status FROM mp_jobs WHERE id=%s',(jid,));old=cur.fetchone()
                if old:
                    if old['stage']!=STAGE or decode(old['input'])!=value:return jsonify(error='Request ID is already used for a different command.'),409
                    return jsonify(id=jid,status=old['status'])
                cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,%s,%s,'queued',%s::jsonb)",(jid,STAGE,value['payload'].get('ticker') or value['payload'].get('policy',{}).get('ticker'),json.dumps(value)))
            return jsonify(id=jid,status='queued'),202
        with get_db() as (_,cur):
            cur.execute('SELECT value,updated_at FROM app_settings WHERE key=%s',(KEY,));row=cur.fetchone()
            cur.execute('SELECT id,ticker,status,input,result,error,created_at FROM mp_jobs WHERE stage=%s ORDER BY created_at DESC LIMIT 50',(STAGE,));commands=[dict(r) for r in cur.fetchall()]
        r=jsonify(snapshot=decode(row['value']) if row else None,updatedAt=str(row['updated_at']) if row else None,commands=commands);r.headers['Cache-Control']='no-store';return r

    @bp.route('/api/agent/collection-control',methods=['GET','POST'])
    def sync():
        if not is_agent():return jsonify(error='Local agent API key required.'),403
        with get_db(commit=request.method=='POST') as (_,cur):
            if request.method=='POST':
                data=request.get_json(silent=True)
                if not isinstance(data,dict) or not isinstance(data.get('snapshot'),dict) or not isinstance(data.get('receipts',[]),list) or len(json.dumps(data))>500000:return jsonify(error='Invalid collection sync payload.'),400
                cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(KEY,json.dumps(data['snapshot'])))
                for receipt in data.get('receipts',[])[:10]:
                    if not isinstance(receipt,dict) or receipt.get('status') not in ('applied','failed'):continue
                    cur.execute("UPDATE mp_jobs SET status=%s,result=%s::jsonb,error=%s,updated_at=NOW() WHERE id=%s AND stage=%s AND status='queued'",(receipt['status'],json.dumps(receipt.get('result',{})),str(receipt.get('error',''))[:2000] or None,receipt.get('id'),STAGE))
                return jsonify(ok=True)
            cur.execute("SELECT id,input FROM mp_jobs WHERE stage=%s AND status='queued' ORDER BY created_at LIMIT 10",(STAGE,))
            return jsonify(commands=[dict(r) for r in cur.fetchall()])
    return bp
