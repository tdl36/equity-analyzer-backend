"""Atomic, replay-safe copies from Research storage. Never moves originals or starts research."""
import base64
import hashlib
import json
import mimetypes
import re
import uuid
from flask import jsonify,request
from company_research import eligible
from pipeline_source_metadata import fields


def digest(encoded):
    if not isinstance(encoded,str) or not 0<len(encoded)<=20_000_000:
        raise ValueError('An original is empty or exceeds the 15 MB import limit.')
    try:raw=base64.b64decode(encoded,validate=True)
    except ValueError:raise ValueError('An original has invalid stored bytes.')
    if not 0<len(raw)<=15_000_000:raise ValueError('An original is empty or exceeds the 15 MB import limit.')
    return hashlib.sha256(raw).hexdigest(),len(raw)


def import_selected(cur,ticker,body):
    from pipeline_lab import config
    signature={'ticker':ticker,'task':body['task'],'revision':body['revision']}
    receipt_key='pipeline_lab_import:'+body['requestId']
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(receipt_key,))
    cur.execute('SELECT value FROM app_settings WHERE key=%s',(receipt_key,));old=cur.fetchone()
    if old:
        receipt=json.loads(old['value']) if isinstance(old['value'],str) else old['value']
        if receipt['input']!=signature:raise ValueError('Import request identity already used with different settings.')
        return {**receipt,'replayed':True}
    key='pipeline_lab_v1:'+ticker
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(key,))
    cur.execute('SELECT value FROM app_settings WHERE key=%s FOR UPDATE',(key,));row=cur.fetchone()
    current=json.loads(row['value']) if row and isinstance(row['value'],str) else (row['value'] if row else None)
    if not current or current['revision']!=body['revision']:raise ValueError('Lab settings changed. Reload before importing.')
    selection=config(current['selection']);refs=selection[body['task']]
    ids=[r[len('charlie:research_'):] for r in refs if r.startswith('charlie:research_')]
    if not ids or len(refs)>1000:raise ValueError('Select up to 1000 sources, including a Research-storage original, and save first.')
    cur.execute('''SELECT rdf.id::text AS id,rdf.filename,rdf.file_type,
        CASE WHEN octet_length(rdf.file_data)<=20000000 THEN rdf.file_data ELSE NULL END AS file_data,
        rd.published_date FROM research_document_files rdf
        JOIN research_documents rd ON rdf.document_id=rd.id
        JOIN research_categories rc ON rd.category_id=rc.id
        WHERE UPPER(rc.name)=%s AND rdf.id::text=ANY(%s) FOR SHARE OF rdf,rd,rc''',(ticker,ids))
    originals=list(cur.fetchall())
    if len(originals)!=len(ids):raise ValueError('A selected Research original is missing from this stock. Reload inventory.')
    mapping={};total=0;receipts=[]
    for doc in originals:
        name=doc['filename']
        if not isinstance(name,str) or not 1<=len(name)<=255:raise ValueError('An original filename cannot be used for research. Review it in Research storage.')
        sha,size=digest(doc['file_data']);total+=size
        if total>60_000_000:raise ValueError('The imported originals exceed the 60 MB pack limit.')
        metadata={'source':'pipeline_lab_research_import','researchFileId':doc['id'],'sha256':sha,'usage':'research',
                  'published':fields({'published':str(doc.get('published_date') or '')})['publishedDate']}
        cur.execute('''INSERT INTO document_files(ticker,filename,file_data,file_type,mime_type,file_size,metadata)
            VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT(ticker,filename) DO NOTHING''',
            (ticker,name,doc['file_data'],doc.get('file_type') or '',mimetypes.guess_type(name)[0] or 'application/octet-stream',size,json.dumps(metadata)))
        cur.execute('SELECT id,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE',(ticker,name));stored=cur.fetchone()
        if not stored or digest(stored['file_data'])[0]!=sha:
            raise ValueError('A different original already uses this filename in Charlie. Nothing was overwritten; resolve the source conflict before importing.')
        if not eligible(stored.get('metadata') or {}):raise ValueError('The matching Charlie original is excluded or reference-only. Review its source policy first.')
        source='charlie:research_'+doc['id'];target='charlie:thesis_'+str(stored['id']);mapping[source]=target
        receipts.append({'source':source,'target':target,'sha256':sha,'bytes':size})
    selection[body['task']]=sorted(set(mapping.get(r,r) for r in refs))
    updated={'revision':current['revision']+1,'selection':selection}
    cur.execute('UPDATE app_settings SET value=%s,updated_at=NOW() WHERE key=%s',(json.dumps(updated),key))
    receipt={'requestId':body['requestId'],'input':signature,'revision':updated['revision'],'receipts':receipts,'researchStarted':False}
    cur.execute('INSERT INTO app_settings(key,value,updated_at) VALUES(%s,%s,NOW())',(receipt_key,json.dumps(receipt)))
    return receipt


def register(bp,get_db):
    @bp.post('/api/pipeline-lab/import-research/<ticker>')
    def import_research(ticker):
        body=request.get_json(silent=True)
        try:
            if (not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker) or not isinstance(body,dict)
                    or set(body)!={'task','revision','requestId'} or body['task'] not in ('thesis','note')
                    or type(body['revision']) is not int or body['revision']<1):raise ValueError()
            body={**body,'requestId':str(uuid.UUID(body['requestId']))}
        except (ValueError,TypeError,AttributeError):return jsonify(error='A saved task selection and import identity are required.'),400
        try:
            with get_db(commit=True) as (_,cur):result=import_selected(cur,ticker,body)
            return jsonify(result)
        except ValueError as exc:return jsonify(error=str(exc)),409
