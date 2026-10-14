"""Pending Lab note call journal. Not registered with production.

Commit intent before calling a provider. An interrupted intent is ambiguous and
never dispatches again. Completed replies can be reused without another call.
"""
import hashlib
import json
import re
from pipeline_lab_notes import _note_identity, normalized, obj


def schema(cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS pipeline_lab_note_calls (
        note_id UUID NOT NULL, batch INTEGER NOT NULL,
        request_hash CHAR(64) NOT NULL, request JSONB NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('dispatched','uncertain','complete')),
        response JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(note_id,batch))''')


def request_value(value):
    keys = {'noteId','batch','sourcePack','contextSha256','promptSha256','modelKey','maxTokens'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('A complete frozen call identity is required.')
    result = dict(value)
    result['noteId'] = _note_identity(value['noteId'])
    if type(value['batch']) is not int or not 1 <= value['batch'] <= 8:
        raise ValueError('A bounded batch number is required.')
    for field in ('contextSha256', 'promptSha256'):
        if not isinstance(value[field], str) or not re.fullmatch('[0-9a-f]{64}', value[field]):
            raise ValueError('Frozen context and prompt hashes are required.')
    if not isinstance(value['modelKey'], str) or not re.fullmatch('[A-Za-z0-9_.:/-]{1,120}', value['modelKey']):
        raise ValueError('A pinned model key is required.')
    if type(value['maxTokens']) is not int or not 1 <= value['maxTokens'] <= 32000:
        raise ValueError('A bounded output cap is required.')
    result['sourcePack'] = normalized({'title':'Call manifest','markdown':'Manifest',
                                      'sourcePack':value['sourcePack'],'qualityIssues':[]})['sourcePack']
    return result


def reserve(cur, request):
    value = request_value(request)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    identity = (value['noteId'], value['batch'])
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('lab-note-call:'+str(identity),))
    cur.execute('SELECT request_hash,status,response FROM pipeline_lab_note_calls WHERE note_id=%s AND batch=%s', identity)
    old = cur.fetchone()
    if old:
        if old['request_hash'] != digest:
            raise ValueError('This batch already has a different frozen call identity.')
        return {'dispatch':False,'status':old['status'],'requestHash':digest,
                'response':obj(old['response']) if old['response'] is not None else None}
    cur.execute('''INSERT INTO pipeline_lab_note_calls(note_id,batch,request_hash,request,status)
                   VALUES(%s,%s,%s,%s::jsonb,'dispatched')''', (*identity,digest,json.dumps(value)))
    return {'dispatch':True,'status':'dispatched','requestHash':digest,'response':None}


def response_value(response):
    if not isinstance(response, dict) or set(response) != {'text','provider','model','stopReason'}:
        raise ValueError('A bounded provider reply is required.')
    for field, bound in (('text',3_000_000),('provider',120),('model',120),('stopReason',100)):
        if not isinstance(response[field],str) or len(response[field].encode('utf-8')) > bound:
            raise ValueError('Provider reply exceeds its bound.')
    if not response['text'].strip():
        raise ValueError('Provider returned an empty note.')
    return dict(response)


def finish(cur, note_id, batch, request_hash, response):
    value = response_value(response)
    cur.execute('SELECT request_hash,status,response FROM pipeline_lab_note_calls WHERE note_id=%s AND batch=%s FOR UPDATE', (note_id,batch))
    row = cur.fetchone()
    if not row or row['request_hash'] != request_hash:
        raise ValueError('Call receipt does not match the reserved intent.')
    if row['status'] == 'complete':
        if obj(row['response']) != value:
            raise ValueError('A different reply was already saved for this call.')
        return value
    cur.execute("UPDATE pipeline_lab_note_calls SET status='complete',response=%s::jsonb,updated_at=NOW() WHERE note_id=%s AND batch=%s", (json.dumps(value),note_id,batch))
    return value


def run_once(get_db, request, invoke):
    """Injected provider invocation; no automatic retry under any exception.

    A commit failure after response receipt also leaves an ambiguous intent.
    The next caller reuses a committed reply or returns attention without spending.
    """
    value = request_value(request)
    with get_db(commit=True) as (_,cur):
        intent = reserve(cur,value)
    if not intent['dispatch']:
        return {'status':'complete','response':intent['response']} if intent['status']=='complete' else {'status':'attention','reason':'provider_outcome_unknown'}
    try:
        response = invoke()
        with get_db(commit=True) as (_,cur):
            saved = finish(cur,value['noteId'],value['batch'],intent['requestHash'],response)
        return {'status':'complete','response':saved}
    except Exception:
        # No exception text: it may contain private provider request material.
        with get_db(commit=True) as (_,cur):
            cur.execute("UPDATE pipeline_lab_note_calls SET status='uncertain',updated_at=NOW() WHERE note_id=%s AND batch=%s AND status='dispatched'", (value['noteId'],value['batch']))
        return {'status':'attention','reason':'provider_outcome_unknown'}
