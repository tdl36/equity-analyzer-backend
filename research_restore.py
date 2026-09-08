"""Preview and restore saved research as new versions, preserving every original."""
import copy
import hashlib
import json
import re
import uuid
from datetime import datetime,timezone
from flask import Blueprint,jsonify,request

TABLES={'note':'research_notes','review':'investment_reviews'}
FIELDS={'note':['note_markdown','sources_markdown','changelog_markdown','note_docx','charts','metadata','version','status'],
        'review':['state','review_markdown','review_html','review_pdf','qc','changelog','metadata','mode']}


def obj(value):
    if isinstance(value,str):
        try:value=json.loads(value)
        except ValueError:return {}
    return value if isinstance(value,dict) else {}


def canonical(value):
    if isinstance(value,(bytes,memoryview)):return {'binarySha256':hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,datetime):return value.isoformat()
    return value


def fingerprint(kind,row):
    return hashlib.sha256(json.dumps(canonical({k:row.get(k) for k in ['id','ticker','created_at']+FIELDS[kind]}),sort_keys=True,default=str).encode()).hexdigest()


def metadata(kind,row,new_id):
    meta=copy.deepcopy(obj(row.get('metadata')))
    meta['restoration']={'sourceId':row['id'],'historicalAsOf':str(row.get('created_at') or ''),'restoredAt':datetime.now(timezone.utc).isoformat(),'createdId':new_id}
    if kind=='review':
        meta['historicalEvidence']=meta.pop('evidence',{})
        meta['evidence']={}
        meta['readiness']={'status':'needs_review','issues':['Restored historical snapshot. Recheck source freshness, prices, estimates and assumptions before relying on it.']}
    return meta


def read(cur,kind,ident,lock=False):
    cur.execute(f'SELECT * FROM {TABLES[kind]} WHERE id=%s'+(' FOR UPDATE' if lock else ''),(ident,));return cur.fetchone()


def latest(cur,kind,ticker):
    cur.execute(f'SELECT * FROM {TABLES[kind]} WHERE ticker=%s ORDER BY created_at DESC,id DESC LIMIT 1',(ticker,));return cur.fetchone()


def summary(kind,row):
    markdown=row.get('note_markdown' if kind=='note' else 'review_markdown') or ''
    return {'id':row['id'],'createdAt':str(row.get('created_at') or ''),'status':row.get('status','saved'),'hash':fingerprint(kind,row),'markdown':markdown[:120000],'previewTruncated':len(markdown)>120000}


def create_blueprint(get_db,render_review):
    bp=Blueprint('research_restore',__name__)
    @bp.route('/api/research/restoration-preview/<kind>/<ident>')
    def preview(kind,ident):
        if kind not in TABLES:return jsonify(error='Choose a note or investment review'),400
        with get_db() as (_,cur):
            source=read(cur,kind,ident)
            if not source:return jsonify(error='Saved version not found'),404
            current=latest(cur,kind,source['ticker'])
        result=jsonify(kind=kind,ticker=source['ticker'],source=summary(kind,source),latest=summary(kind,current))
        result.headers['Cache-Control']='no-store';return result

    @bp.route('/api/research/restorations',methods=['POST'])
    def restore():
        d=request.get_json(silent=True)
        if not isinstance(d,dict) or d.get('kind') not in TABLES:return jsonify(error='Choose a saved version'),400
        try:jid=str(uuid.UUID(d.get('requestId')))
        except (ValueError,TypeError,AttributeError):return jsonify(error='A valid restoration request ID is required'),400
        kind=d['kind'];identity={k:d.get(k) for k in ('kind','sourceId','sourceHash','latestId','latestHash')}
        if any(not isinstance(v,str) or not v for v in identity.values()):return jsonify(error='Preview this restoration first'),400
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('restore:'+jid,))
            cur.execute('SELECT stage,input,result FROM mp_jobs WHERE id=%s',(jid,));receipt=cur.fetchone()
            if receipt:
                if receipt['stage']!='research_restore' or obj(receipt['input'])!=identity:return jsonify(error='Request ID belongs to another restoration'),409
                return jsonify(obj(receipt['result']))
            source=read(cur,kind,d['sourceId'])
            if not source:return jsonify(error='Original version is no longer available'),404
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('research-edit:'+source['ticker'],))
            source=read(cur,kind,d['sourceId'],True)
            if not source:return jsonify(error='Original version is no longer available'),404
            current=latest(cur,kind,source['ticker'])
            if fingerprint(kind,source)!=d['sourceHash'] or current['id']!=d['latestId'] or fingerprint(kind,current)!=d['latestHash']:
                return jsonify(error='Saved research changed after preview. Reload the comparison before restoring.'),409
            if source['id']==current['id']:return jsonify(error='This is already the latest saved version'),400
            new_id=str(uuid.uuid4());meta=metadata(kind,source,new_id)
            if kind=='note':
                cur.execute("INSERT INTO research_notes(id,ticker,version,note_markdown,sources_markdown,changelog_markdown,note_docx,charts,metadata,status) VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,'draft')",(new_id,source['ticker'],source.get('version') or '1.0',source.get('note_markdown') or '',source.get('sources_markdown') or '',f"Restored historical note from {source['id']}; original saved {source.get('created_at')}. Review before publishing.",source.get('note_docx') or '',json.dumps(source.get('charts') or []),json.dumps(meta)))
            else:
                state=copy.deepcopy(obj(source['state']))
                state['as_of']=state.get('as_of') or str(source.get('created_at') or '')[:10]
                state['historical_as_of']=state['as_of']
                md,html,pdf=render_review(source['ticker'],state,source.get('mode') or 'review')
                qc={'verdict':'revise','findings':[{'severity':'medium','issue':'Historical restoration; prices, sources and assumptions require a fresh review.'}]}
                cur.execute('INSERT INTO investment_reviews(id,ticker,mode,state,review_markdown,review_html,review_pdf,qc,changelog,metadata) VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)',(new_id,source['ticker'],source.get('mode') or 'review',json.dumps(state),md,html,pdf,json.dumps(qc),json.dumps([{'restoredFrom':source['id']}]),json.dumps(meta)))
            result={'createdId':new_id,'kind':kind,'ticker':source['ticker'],'status':'draft' if kind=='note' else 'historical_review'}
            cur.execute("INSERT INTO mp_jobs(id,ticker,stage,status,input,result) VALUES(%s,%s,'research_restore','applied',%s::jsonb,%s::jsonb)",(jid,source['ticker'],json.dumps(identity),json.dumps(result)))
        return jsonify(result),201
    return bp
