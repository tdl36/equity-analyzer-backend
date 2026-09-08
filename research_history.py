"""Read-only chronology of saved research records, with explicit provenance links."""
import json
import re
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request


def timestamp(value):
    try:
        if isinstance(value,(float,int)) and not isinstance(value,bool):d=datetime.fromtimestamp(value,timezone.utc)
        elif isinstance(value,datetime):d=value
        else:d=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if d.tzinfo is None:d=d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).isoformat()
    except (ValueError,TypeError,OverflowError,OSError):return None


def obj(v):
    if isinstance(v,str):
        try:v=json.loads(v)
        except ValueError:return {}
    return v if isinstance(v,dict) else {}


def records(groups):
    out=[]
    for kind,rows in groups.items():
        for r in rows:
            title=r.get('title') or kind.replace('_',' ').capitalize()
            record={'id':f"{kind}:{r['id']}",'recordId':str(r['id']),'kind':kind,
                'ticker':r.get('ticker'),'title':str(title)[:300],'status':r.get('status') or 'saved',
                'createdAt':timestamp(r.get('created_at')),'updatedAt':timestamp(r.get('updated_at')),
                'parentId':r.get('parent_id'),'proposalId':r.get('proposal_id')}
            out.append(record)
    # A source import's creation time is not its publication date. Activity updates
    # are current snapshots, not an invented sequence of status transition events.
    return sorted(out,key=lambda r:(r['createdAt'] or '',r['id']),reverse=True)[:100]


def create_blueprint(get_db):
    bp=Blueprint('research_history',__name__)
    @bp.route('/api/research/workspace-history')
    def history():
        ticker=request.args.get('ticker','').strip().upper()
        if ticker and not re.fullmatch(r'[A-Z0-9.^-]{1,20}',ticker):return jsonify(error='Enter a valid ticker.'),400
        where=' WHERE ticker=%s' if ticker else ''
        args=(ticker,) if ticker else ()
        groups={}
        with get_db() as (_,cur):
            queries={
                'source':"SELECT id,ticker,filename AS title,created_at FROM document_files",
                'note':"SELECT id,ticker,'Research note ' || version AS title,status,created_at,updated_at,COALESCE(metadata->'restoration'->>'sourceId',metadata->'revision'->>'parentId') AS parent_id,metadata->'revision'->>'proposalId' AS proposal_id FROM research_notes",
                'review':"SELECT id,ticker,'Investment review' AS title,created_at,COALESCE(metadata->'restoration'->>'sourceId',metadata->'revision'->>'parentId') AS parent_id,metadata->'revision'->>'proposalId' AS proposal_id FROM investment_reviews",
                'activity':"SELECT id,ticker,activity_type || COALESCE(' · ' || (input->>'topic'),'') AS title,status,created_at,updated_at FROM analyst_activities"}
            for kind,sql in queries.items():
                cur.execute(sql+where+' ORDER BY created_at DESC LIMIT 100',args)
                groups[kind]=[dict(r) for r in cur.fetchall()]
            cur.execute("SELECT id,ticker,stage AS title,status,created_at,updated_at,input->>'targetId' AS parent_id FROM mp_jobs WHERE stage IN ('research_edit','research_chat','evidence_amendment','collection_control','catalyst_signal','research_restore')"+(' AND ticker=%s' if ticker else '')+' ORDER BY created_at DESC LIMIT 100',args)
            groups['job']=[dict(r) for r in cur.fetchall()]
            cur.execute("SELECT value,updated_at FROM app_settings WHERE key='collection_control_snapshot'")
            snapshot=cur.fetchone()
        state=obj(snapshot['value']) if snapshot else {}
        groups['collection']=[{'id':r['id'],'ticker':r.get('ticker'),'title':'AlphaSense browser refresh','status':r.get('status'),'created_at':r.get('created')} for r in state.get('requests',[]) if isinstance(r,dict) and r.get('id') and (not ticker or r.get('ticker')==ticker)]
        response=jsonify(records=records(groups),collectionReportedAt=timestamp(snapshot['updated_at']) if snapshot else None,
            scope='Latest 100 records by creation time. Status is the latest saved state; this is not a complete status-transition audit. Imports are dated by ingestion, not publication. Records sharing a ticker are not automatically linked as cause and effect.')
        response.headers['Cache-Control']='no-store';return response
    return bp
