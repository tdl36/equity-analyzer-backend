"""Pending authenticated full-note submission boundary; not registered.

The server-owned builder must resolve saved selections, freeze context/playbook and
validate model/input budgets. Clients cannot provide prompts, keys or source bytes.
"""
import hashlib
import json
import re
from flask import Blueprint,jsonify,request
from pipeline_lab_notes import _note_identity
from pipeline_lab_note_jobs import enqueue,status,recover


def schema(cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS pipeline_lab_note_submissions (
        note_id UUID PRIMARY KEY REFERENCES pipeline_lab_note_plans(note_id),
        request_hash CHAR(64) NOT NULL)''')


def submit(cur,ticker,data,build_plan):
    if not isinstance(ticker,str) or not re.fullmatch('[A-Z0-9][A-Z0-9.-]{0,19}',ticker):raise ValueError('Confirmed ticker required.')
    if not isinstance(data,dict) or set(data)!={'requestId','revision','digest','allowPdfText'}:
        raise ValueError('Request identity, saved revision, checked digest and representation choice required.')
    note_id=_note_identity(data['requestId'])
    if type(data['revision']) is not int or data['revision']<1:raise ValueError('Saved note-selection revision required.')
    if not isinstance(data['digest'],str) or not re.fullmatch('[0-9a-f]{64}',data['digest']):raise ValueError('Checked source digest required.')
    if type(data['allowPdfText']) is not bool:raise ValueError('Explicit representation choice required.')
    signature={**data,'requestId':note_id,'ticker':ticker}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('lab-note-submit:'+note_id,))
    cur.execute('SELECT request_hash FROM pipeline_lab_note_submissions WHERE note_id=%s',(note_id,))
    prior=cur.fetchone()
    if prior:
        if prior['request_hash']!=digest:raise ValueError('Request identity was already used for different inputs.')
        return status(cur,note_id)
    plan=build_plan(cur,signature)
    if plan['noteId']!=note_id or plan['ticker']!=ticker or plan['allowPdfText']!=data['allowPdfText']:
        raise ValueError('Server plan does not match the submitted request.')
    result=enqueue(cur,plan)
    cur.execute('INSERT INTO pipeline_lab_note_submissions(note_id,request_hash) VALUES(%s,%s)',(note_id,digest))
    return result


def create_blueprint(get_db,authenticated,build_plan):
    bp=Blueprint('pipeline_lab_note_submit',__name__)
    @bp.before_request
    def auth():
        if not authenticated():return jsonify(error='Sign in to manage Lab note jobs.'),401
    @bp.after_request
    def private(response):
        response.headers['Cache-Control']='private, no-store';return response
    def body():
        if request.args or request.mimetype!='application/json':raise ValueError('Send a JSON request without query options.')
        if request.content_length is not None and request.content_length>4096:raise ValueError('Request exceeds its size limit.')
        raw=request.stream.read(4097)
        if len(raw)>4096:raise ValueError('Request exceeds its size limit.')
        try:return json.loads(raw)
        except (ValueError,UnicodeError):raise ValueError('Invalid request JSON.')
    @bp.post('/api/pipeline-lab/note-jobs/<ticker>')
    def start(ticker):
        try:
            data=body()
            with get_db(commit=True) as (_,cur):result=submit(cur,ticker,data,build_plan)
            return jsonify(result)
        except ValueError as exc:return jsonify(error=str(exc)),409
    @bp.get('/api/pipeline-lab/note-job/<note_id>')
    def progress(note_id):
        try:
            with get_db() as (_,cur):result=status(cur,note_id)
            return jsonify(result) if result else (jsonify(error='Note job not found.'),404)
        except ValueError as exc:return jsonify(error=str(exc)),400
    @bp.get('/api/pipeline-lab/note-job/<note_id>/inspection')
    def inspection(note_id):
        from pipeline_lab_note_inspection import inspect
        try:
            if set(request.args)!={'ticker'} or len(request.args.getlist('ticker'))!=1:
                raise ValueError('An exact company ticker is required.')
            with get_db() as (_,cur):result=inspect(cur,note_id,request.args.get('ticker'))
            return jsonify(result) if result else (jsonify(error='Note job not found for this company.'),404)
        except ValueError as exc:return jsonify(error=str(exc)),400
    @bp.post('/api/pipeline-lab/note-job/<note_id>/recover')
    def resume(note_id):
        try:
            if body()!={}:raise ValueError('Recovery requires an empty object.')
            with get_db(commit=True) as (_,cur):result=recover(cur,note_id)
            return jsonify(result)
        except ValueError as exc:return jsonify(error=str(exc)),409
    return bp
