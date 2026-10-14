"""HTTP adapter for Lab import transactions and the authenticated Mac agent.

Normal user routes require the application's existing authentication middleware.
Agent mutation routes additionally require the injected Mac-agent credential check.
"""
import json
import re
from flask import Blueprint,jsonify,request
from pipeline_lab_icloud import PREFIX,load,job_key,enqueue,freeze,complete,cancel,attention,retry


def body(limit):
    if request.mimetype!='application/json':raise ValueError('Send a JSON import request.')
    if request.content_length is not None and request.content_length>limit:raise ValueError('Import request exceeds its size limit.')
    raw=request.stream.read(limit+1)
    if len(raw)>limit:raise ValueError('Import request exceeds its size limit.')
    try:value=json.loads(raw)
    except (ValueError,UnicodeError):raise ValueError('Invalid import request JSON.')
    if not isinstance(value,dict):raise ValueError('Expected an import request object.')
    return value


def create_blueprint(get_db,is_agent):
    bp=Blueprint('pipeline_lab_icloud',__name__)
    @bp.after_request
    def private(response):
        response.headers['Cache-Control']='private, no-store';return response

    @bp.get('/api/pipeline-lab/icloud-imports/<ticker>')
    def inventory(ticker):
        if not re.fullmatch('[A-Z0-9][A-Z0-9.-]{0,19}',ticker):return jsonify(error='Invalid ticker.'),400
        with get_db() as (_,cur):
            cur.execute("SELECT value FROM app_settings WHERE key LIKE %s AND value::jsonb->'input'->>'ticker'=%s ORDER BY updated_at DESC LIMIT 25",(PREFIX+'%',ticker))
            rows=[json.loads(r['value']) if isinstance(r['value'],str) else r['value'] for r in cur.fetchall()]
        return jsonify(imports=rows)

    @bp.post('/api/pipeline-lab/icloud-imports/<ticker>')
    def submit(ticker):
        try:
            data=body(4096)
            if set(data)!={'requestId','task','revision'}:raise ValueError('Request identity, saved task and revision required.')
            with get_db(commit=True) as (_,cur):result=enqueue(cur,ticker,data['task'],data['revision'],data['requestId'])
            return jsonify(result)
        except ValueError as exc:return jsonify(error=str(exc)),409

    @bp.get('/api/pipeline-lab/icloud-import/<request_id>')
    def status(request_id):
        try:
            with get_db() as (_,cur):result=load(cur,job_key(request_id))
            return (jsonify(result),200) if result else (jsonify(error='Import request not found.'),404)
        except ValueError as exc:return jsonify(error=str(exc)),400

    @bp.post('/api/pipeline-lab/icloud-import/<request_id>/cancel')
    def stop(request_id):
        try:
            if body(1024)!={}:raise ValueError('Cancel requires an empty object.')
            with get_db(commit=True) as (_,cur):result=cancel(cur,request_id)
            return jsonify(result)
        except ValueError as exc:return jsonify(error=str(exc)),409

    @bp.post('/api/pipeline-lab/icloud-import/<request_id>/retry')
    def resume(request_id):
        try:
            if body(1024)!={}:raise ValueError('Retry requires an empty object.')
            with get_db(commit=True) as (_,cur):result=retry(cur,request_id)
            return jsonify(result)
        except ValueError as exc:return jsonify(error=str(exc)),409

    @bp.get('/api/agent/pipeline-lab-imports')
    def pending():
        if not is_agent():return jsonify(error='Mac agent authentication required.'),403
        with get_db() as (_,cur):
            cur.execute("SELECT value FROM app_settings WHERE key LIKE %s AND value::jsonb->>'status' IN ('queued','frozen') ORDER BY updated_at ASC LIMIT 10",(PREFIX+'%',))
            rows=[json.loads(r['value']) if isinstance(r['value'],str) else r['value'] for r in cur.fetchall()]
        return jsonify(imports=rows)

    @bp.post('/api/agent/pipeline-lab-import/<request_id>/<action>')
    def receive(request_id,action):
        if not is_agent():return jsonify(error='Mac agent authentication required.'),403
        if action not in ('freeze','complete','issue'):return jsonify(error='Unknown import action.'),404
        try:
            data=body(80_100_000 if action=='complete' else 16000)
            if action=='issue':
                if set(data)!={'code'}:raise ValueError('Bounded import issue code required.')
                with get_db(commit=True) as (_,cur):result=attention(cur,request_id,data['code'])
                return jsonify(result)
            field='manifest' if action=='freeze' else 'originals'
            if set(data)!={field}:raise ValueError('Exact import manifest or originals required.')
            with get_db(commit=True) as (_,cur):result=(freeze if action=='freeze' else complete)(cur,request_id,data[field])
            return jsonify(result)
        except ValueError as exc:return jsonify(error=str(exc)),409
    return bp
