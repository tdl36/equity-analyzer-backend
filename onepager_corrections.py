"""Immutable pending editorial corrections. No model calls or research acceptance."""
import hashlib
import json
import uuid
from flask import Blueprint, jsonify, request
from onepager_editorial_fields import editable_fields
from company_research import eligible
from research_originals import verified_original


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode()).hexdigest()


def preparation(run):
    if run.get('status')!='complete':raise ValueError('Choose a completed saved report.')
    fields=editable_fields(run.get('state',{}).get('report',{}))
    baseline=digest({k:run.get(k) for k in ('id','ticker','input','baseline','sources','state')})
    return {'runId':str(run['id']),'ticker':run['ticker'],'fingerprint':baseline,
            'fields':[{'path':p,'before':v} for p,v in fields.items()],
            'sources':[{'id':s['id'],'filename':s['filename'],'sha256':s.get('originalHash')} for s in run.get('sources',[])]}


def evidence_items(proposal):
    """Primary plus explicitly supplied additional originals, preserving old records."""
    return [proposal] + proposal.get('additionalEvidence', [])


def submission_evidence(body):
    extra=body.get('additionalEvidence', [])
    if not isinstance(extra,list) or ('additionalEvidence' in body and not 1<=len(extra)<=7):
        raise ValueError('Add between one and seven additional originals.')
    items=[{k:body.get(k) for k in ('sourceId','excerpt','locator')}]
    for item in extra:
        if not isinstance(item,dict) or set(item)!={'sourceId','excerpt','locator'}:
            raise ValueError('Additional evidence must contain only source ID, passage and locator.')
        items.append(item)
    ids=[]
    for item in items:
        if not isinstance(item['sourceId'],str) or not item['sourceId'] or item['sourceId'] in ids:
            raise ValueError('Choose distinct frozen originals for each evidence passage.')
        ids.append(item['sourceId'])
        for key,limit in [('excerpt',4000),('locator',200)]:
            if not isinstance(item[key],str) or not item[key].strip() or len(item[key])>limit:
                raise ValueError('Evidence text is missing or exceeds its limit.')
    return items


def prepare(run,body,document):
    if not isinstance(body,dict) or set(body)-{'additionalEvidence'}!={'requestId','fingerprint','path','statement','basis','reason','sourceId','excerpt','locator'}:
        raise ValueError('Use the correction form; unknown review or approval fields are not accepted.')
    try:identity=str(uuid.UUID(body['requestId']))
    except (ValueError,TypeError,AttributeError):raise ValueError('Invalid correction identity.') from None
    pack=preparation(run)
    if body['fingerprint']!=pack['fingerprint']:raise ValueError('The saved report or sources changed. Reload before preparing a correction.')
    fields={f['path']:f['before'] for f in pack['fields']}
    if not isinstance(body['path'],str) or body['path'] not in fields:raise ValueError('Choose a saved claim or permitted scenario field.')
    for key,limit in [('statement',2000),('reason',2000),('excerpt',4000),('locator',200)]:
        if not isinstance(body[key],str) or not body[key].strip() or len(body[key])>limit:raise ValueError('Correction text is missing or exceeds its limit.')
    if body['statement']==fields[body['path']]:raise ValueError('The proposed wording is unchanged.')
    if body['basis'] not in ('reported_fact','management_guidance','broker_estimate','interpretation','hypothesis'):raise ValueError('Choose a claim type.')
    checked=[]
    for item in submission_evidence(body):
        source=next((s for s in run.get('sources',[]) if s['id']==item['sourceId']),None)
        if not source:raise ValueError('Choose a source from this frozen report.')
        doc=document.get(source['filename']) if isinstance(document,dict) and 'file_data' not in document else document
        if not doc or not eligible(doc.get('metadata') or {}):raise ValueError('The source is unavailable or restricted. Restore eligible original evidence before submitting.')
        verified_original(source,doc,(run.get('input',{}).get('hashes') or {}).get(source['filename']))
        excerpt=' '.join(item['excerpt'].split())
        if len(excerpt)<30 or excerpt not in ' '.join(str(source.get('text') or '').split()):
            raise ValueError('The passage was not matched in this report’s frozen extracted source. Inspect the original; do not substitute another source.')
        checked.append({**item,'filename':source['filename'],'sha256':source['originalHash']})
    proposal={k:body[k] for k in ('path','statement','basis','reason','sourceId','excerpt','locator')}
    proposal.update(before=fields[body['path']],filename=checked[0]['filename'],sha256=checked[0]['sha256'],
                    passageMatched=True,review='pending',
                    caveat='Passage matches frozen extracted text and original bytes. Claim meaning, completeness and locator still require investor review. No research or thesis was changed.')
    if 'additionalEvidence' in body:proposal['additionalEvidence']=checked[1:]
    return identity,digest(body),proposal


def create_blueprint(get_db):
    bp=Blueprint('onepager_corrections',__name__)
    @bp.after_request
    def private_response(response):
        response.headers['Cache-Control']='private, no-store'
        return response
    @bp.route('/api/research/stock-analysis-run/<rid>/corrections',methods=['GET','POST'])
    def corrections(rid):
        try:
            uuid.UUID(rid)
            body=request.get_json(silent=True) if request.method=='POST' else None
            if request.method=='POST' and (not isinstance(body,dict) or len(json.dumps(body))>50000):raise ValueError('Invalid or oversized correction.')
            with get_db(commit=request.method=='POST') as (_,cur):
                cur.execute('SELECT id,ticker,status,input,baseline,sources,state FROM stock_analysis_runs WHERE id=%s'+(' FOR SHARE' if request.method=='POST' else ''),(rid,))
                row=cur.fetchone()
                if not row:return jsonify(error='Saved report not found.'),404
                run=dict(row);pack=preparation(run)
                if request.method=='GET':
                    # Policy metadata preflight only; writes/export still verify exact bytes.
                    for source in pack['sources']:
                        cur.execute('SELECT metadata FROM document_files WHERE ticker=%s AND filename=%s', (run['ticker'], source['filename']))
                        document=cur.fetchone()
                        source['policyStatus']='missing' if document is None else 'eligible' if eligible(document.get('metadata') or {}) else 'restricted'
                    pack['sourcePolicyReady']=bool(pack['sources']) and all(s['policyStatus']=='eligible' for s in pack['sources'])
                    cur.execute("SELECT to_regclass('onepager_corrections') AS name")
                    exists=cur.fetchone()['name'];rows=[]
                    if exists:
                        cur.execute('SELECT id,fingerprint,proposal,created_at FROM onepager_corrections WHERE run_id=%s ORDER BY created_at DESC LIMIT 150',(rid,))
                        rows=[dict(r) for r in cur.fetchall()]
                    for item in rows:item['stale']=item['fingerprint']!=pack['fingerprint']
                    return jsonify(**pack,corrections=rows,limit=150)
                # Schema and submission are one transaction; reads never create work.
                cur.execute("SELECT pg_advisory_xact_lock(hashtext('onepager-corrections-schema'))")
                cur.execute('''CREATE TABLE IF NOT EXISTS onepager_corrections (
                    id UUID PRIMARY KEY,run_id TEXT NOT NULL,fingerprint TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,proposal JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')
                cur.execute('CREATE INDEX IF NOT EXISTS onepager_corrections_run_idx ON onepager_corrections(run_id,created_at DESC)')
                try:ident=str(uuid.UUID(body.get('requestId','')))
                except (ValueError,TypeError,AttributeError):raise ValueError('Invalid correction identity.') from None
                cur.execute('SELECT run_id,payload_hash FROM onepager_corrections WHERE id=%s',(ident,))
                prior=cur.fetchone()
                if prior:
                    if str(prior['run_id'])!=rid or prior['payload_hash']!=digest(body):return jsonify(error='This submission identity belongs to different wording. Reload saved corrections.'),409
                    return jsonify(id=ident,reused=True,status='pending')
                documents={}
                for item in submission_evidence(body):
                    source=next((s for s in run.get('sources',[]) if s['id']==item['sourceId']),None)
                    if not source:raise ValueError('Choose a source from this frozen report.')
                    cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE',(run['ticker'],source['filename']))
                    documents[source['filename']]=cur.fetchone()
                ident,signature,proposal=prepare(run,body,documents)
                cur.execute('INSERT INTO onepager_corrections(id,run_id,fingerprint,payload_hash,proposal) VALUES(%s,%s,%s,%s,%s::jsonb)',(ident,rid,pack['fingerprint'],signature,json.dumps(proposal)))
                return jsonify(id=ident,reused=False,status='pending'),201
        except (ValueError,TypeError,AttributeError) as exc:return jsonify(error=str(exc)),400
    return bp
