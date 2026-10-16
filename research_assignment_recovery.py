"""Deliver a separately authored pending review package without approving research."""
import json
from datetime import datetime, timezone
from onepager_edition_contract import _uuid
from onepager_corrections import digest
from onepager_decisions import read_lineage
from onepager_edition_evidence import pending_evidence
from research_originals import verified_original
from company_research import eligible
from thesis_imports import current, row_hash


def deliver(get_db, aid, body):
    aid = _uuid(aid)
    if not isinstance(body, dict) or set(body) not in ({'requestId','thesisDraftId','editionId'}, {'requestId','thesisDraftId','editionId','previousRequestId'}):
        raise ValueError('Choose a saved pending thesis and edition; approval flags are forbidden.')
    ident, tid, eid = (_uuid(body[k]) for k in ('requestId','thesisDraftId','editionId'))
    with get_db(commit=True) as (_, cur):
        cur.execute("SELECT * FROM mp_jobs WHERE id=%s AND stage='research_assignment' FOR UPDATE", (aid,))
        row = cur.fetchone()
        if not row: raise ValueError('Saved assignment not found.')
        assignment = dict(row); state = dict(assignment.get('result') or {})
        previous = state.get('reviewPackage')
        # Replays remain idempotent even after a later revision becomes current.
        for saved_package in [previous] + state.get('reviewPackageHistory', []):
            if saved_package and saved_package['requestId'] == ident:
                if saved_package['requestDigest'] != digest(body):
                    raise ValueError('Delivery identity was reused with different content.')
                return {'package':saved_package,'reused':True}
        if previous:
            if body.get('previousRequestId') != previous['requestId']:
                raise ValueError('The review package changed. Refresh before revising it.')
            if assignment['status'] != 'complete':
                raise ValueError('Only a delivered package can be revised.')
        elif 'previousRequestId' in body or assignment['status'] != 'attention':
            raise ValueError('Use a blocked summary, thesis and visual assignment.')
        if set(assignment['input'].get('outputs',[])) != {'stock_summary','thesis','visual'}:
            raise ValueError('Use a summary, thesis and visual assignment.')
        if (state.get('collection') or {}).get('status') != 'complete' or len(state.get('artifacts') or {}) < 2:
            raise ValueError('Collection and retained outputs must be complete before manual delivery.')
        cur.execute('SELECT * FROM external_thesis_drafts WHERE id=%s FOR SHARE',(tid,))
        draft = cur.fetchone()
        # Fence legacy thesis writers and approval while checking the live baseline.
        cur.execute('LOCK TABLE portfolio_analyses IN SHARE MODE')
        if not draft or draft['ticker'] != assignment['ticker'] or draft['status'] != 'pending':
            raise ValueError('Use a pending thesis for this company.')
        if row_hash(current(cur,assignment['ticker'])) != draft['baseline_hash']:
            raise ValueError('The live thesis changed. Reconcile a fresh proposal.')
        if draft['package'].get('baseline',{}).get('hash') != assignment['input'].get('thesisBaseline',{}).get('baseline',{}).get('hash'):
            raise ValueError('Thesis and assignment baselines differ.')
        expected = {s.get('sha256') for s in state.get('sources',[]) if s.get('sha256')}
        supplied = {s.get('sha256') for s in draft['package'].get('sourceRegister',[]) if s.get('sha256')}
        if not expected or expected != supplied:
            raise ValueError('Recovery thesis must use the exact frozen original pack.')
        parent, decisions = read_lineage(cur,aid)
        if eid in decisions: raise ValueError('Use a pending edition without an investor decision.')
        cur.execute('SELECT record FROM onepager_editions WHERE id=%s AND assignment_id=%s FOR SHARE',(eid,aid))
        saved = cur.fetchone()
        if not saved: raise ValueError('Pending edition is missing or belongs to another assignment.')
        record = saved['record']; rid = state.get('reportId')
        cur.execute('SELECT id,ticker,status,input,baseline,sources,state FROM stock_analysis_runs WHERE id=%s FOR SHARE',(rid,))
        run = cur.fetchone()
        if not run: raise ValueError('Saved report missing.')
        run = dict(run)
        if {s.get('originalHash') for s in run.get('sources',[])} != expected:
            raise ValueError('Report and assignment original packs differ.')
        cur.execute('SELECT id,run_id,fingerprint,payload_hash,proposal FROM onepager_corrections WHERE id=ANY(%s::uuid[]) FOR SHARE',(record['correctionIds'],))
        corrections = [dict(r) for r in cur.fetchall()]
        documents = {}
        for s in run['sources']:
            cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE',(run['ticker'],s['filename']))
            matches = cur.fetchall()
            if len(matches)!=1 or not eligible(matches[0].get('metadata') or {}):
                raise ValueError('Frozen original missing, ambiguous or restricted.')
            documents[s['filename']] = dict(matches[0])
            verified_original(s,documents[s['filename']],(run.get('input',{}).get('hashes') or {}).get(s['filename']))
        pending_evidence(run,assignment,record,corrections,documents,state.get('visualFundamentals'),parent_id=parent)
        package = {'requestId':ident,'requestDigest':digest(body),'thesisDraftId':tid,'editionId':eid,
                   'thesisFingerprint':draft['fingerprint'],'editionDigest':digest(record),
                   'sourceHashes':sorted(expected),'deliveredAt':datetime.now(timezone.utc).isoformat(),
                   'status':'pending_investor_review','originalFailure':previous.get('originalFailure') if previous else assignment.get('error'),
                   'previousThesisProposal':state.get('separateThesisProposal'),
                   'caveat':'Assistant-authored recovery drafts. Exact source identity and saved baselines checked; meaning requires investor review. Original automated research remains preserved and is not certified.'}
        if previous:
            state.setdefault('reviewPackageHistory', []).append(previous)
        state['reviewPackage']=package
        state['step']='Recovered drafts ready for investor review'
        # This is delivery completion, not thesis/edition acceptance. Keep original
        # errors and output pointers in the retained state and receipt.
        cur.execute("UPDATE mp_jobs SET result=%s::jsonb,status='complete',error=NULL,updated_at=NOW() WHERE id=%s",(json.dumps(state),aid))
        return {'package':package,'reused':False}


def summary_html(draft, package):
    """Readable saved draft; caller revalidates source edition and thesis first."""
    from html import escape
    a=draft['package']['analysis'];esc=lambda x:escape(str(x or ''),quote=True)
    def refs(item):
        return '<small>Sources: '+esc(', '.join(r.get('sourceId','')+' pp '+','.join(map(str,r.get('pdfPages',[]))) for r in item.get('sources',[])))+'</small>'
    parts=['<h2>Investment view</h2><p>'+esc(a['thesis']['summary'])+'</p>']
    for item in a['thesis']['pillars']:
        parts.append('<section><h2>'+esc(item['title'])+'</h2><p>'+esc(item['description'])+'</p>'+refs(item)+'</section>')
    parts.append('<h2>Monitoring plan</h2>')
    thresholds = draft['package'].get('provenance', {}).get('monitoringThresholds')
    if thresholds: parts.append('<p>'+esc(thresholds)+'</p>')
    for item in a['signposts']:
        parts.append('<section><h3>'+esc(item['metric'])+'</h3><p>'+esc(item['target'])+'</p><p>'+esc(item['timeframe'])+'</p>'+refs(item)+'</section>')
    parts.append('<h2>Risks and reconsideration triggers</h2>')
    for item in a['threats']:
        parts.append('<section><h3>'+esc(item['threat'])+'</h3><p>'+esc(item['triggerPoints'])+'</p>'+refs(item)+'</section>')
    parts.append('<h2>Conclusion and scope</h2><p>'+esc(a['conclusion'])+'</p>')
    parts+=['<p>'+esc(x)+'</p>' for x in draft['package'].get('provenance',{}).get('limitations',[])]
    parts.append('<h2>Source register</h2><ol>')
    for s in draft['package']['sourceRegister']:
        parts.append('<li><b>'+esc(s['id'])+'</b> '+esc(s['filename'])+'<br>Reviewed pages: '+esc(', '.join(map(str,s.get('reviewedPdfPages',[]))))+'<br><small>SHA-256 '+esc(s['sha256'])+'</small></li>')
    parts.append('</ol>')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>*{box-sizing:border-box;color:black;font-family:Calibri,sans-serif}body{max-width:850px;margin:24px auto;padding:0 22px;font-size:15px;line-height:1.4}h1{font-size:28px}h2{font-size:19px;border-bottom:1px solid #bbb;padding-bottom:5px}h3{font-size:16px}small{font-size:11px}section,li{break-inside:avoid}li{overflow-wrap:anywhere;margin-bottom:12px}.notice{background:#eef2ef;padding:12px}@media print{@page{size:A4;margin:16mm}body{font-size:10pt;margin:0;padding:0}h1{font-size:20pt}h2{font-size:13pt}h3{font-size:11pt}small{font-size:8pt}}</style></head><body><h1>'+esc(draft['ticker'])+' · Reconciled stock summary</h1><p class="notice">Pending investor review. Assistant-authored recovery; no thesis changes approved. Source window: '+esc(draft['package'].get('provenance',{}).get('sourceWindow','See frozen assignment'))+'. Dated prices are not live quotes.</p>'+''.join(parts)+'<p><small>Thesis '+esc(package['thesisDraftId'])+' · edition '+esc(package['editionId'])+'</small></p></body></html>'
