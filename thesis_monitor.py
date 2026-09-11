"""Prospective saved-source monitoring with durable submission receipts."""
import hashlib
import json
import re
import uuid
from flask import jsonify, request


def ensure(get_db):
    with get_db(commit=True) as (_, c):
        c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('thesis-monitor-schema',))
        c.execute('''CREATE TABLE IF NOT EXISTS thesis_monitors (
            ticker TEXT PRIMARY KEY, enabled BOOLEAN NOT NULL DEFAULT FALSE,
            revision INTEGER NOT NULL DEFAULT 0, seen JSONB NOT NULL DEFAULT '{}',
            pending JSONB, last_result JSONB NOT NULL DEFAULT '{}',
            checked_at TIMESTAMPTZ, updated_at TIMESTAMPTZ DEFAULT NOW())''')


def inventory(c, ticker):
    c.execute("SELECT filename,encode(sha256(convert_to(file_data,'UTF8')),'hex') AS digest FROM document_files WHERE ticker=%s ORDER BY filename", (ticker,))
    rows=list(c.fetchall())
    if len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source filenames must be resolved before monitoring.')
    return {r['filename']:r['digest'] for r in rows}


def candidate(current, seen):
    # A renamed byte-identical source is not new information.
    prior=set(seen.values())
    unique={}
    for name,digest in sorted(current.items()):
        if seen.get(name)!=digest and digest not in prior:
            unique.setdefault(digest,name)
    return [name for name in unique.values()][:10]


def register(bp,get_db,submit):
    @bp.route('/api/research/investment-case/<ticker>/monitor',methods=['GET','POST'])
    def policy(ticker):
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,19}',ticker):return jsonify(error='Invalid ticker'),400
        ensure(get_db)
        try:
            with get_db(commit=True) as (_,c):
                c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('thesis-monitor:'+ticker,))
                c.execute('SELECT * FROM thesis_monitors WHERE ticker=%s',(ticker,));row=c.fetchone()
                if request.method=='POST':
                    data=request.get_json(silent=True) or {}
                    if type(data.get('enabled')) is not bool or type(data.get('revision')) is not int:
                        return jsonify(error='Choose monitoring state and reload its revision.'),400
                    if data['revision']!=(row['revision'] if row else 0):return jsonify(error='Monitor settings changed. Reload.'),409
                    if data.get('resetPending'):
                        pending=(row or {}).get('pending')
                        if pending:
                            c.execute('SELECT id FROM mp_jobs WHERE id=%s',(pending['requestId'],))
                            if c.fetchone():return jsonify(error='This reservation already has a proposal. Review it before resetting.'),409
                            c.execute('UPDATE thesis_monitors SET pending=NULL WHERE ticker=%s',(ticker,))
                    if data['enabled']:
                        c.execute('SELECT body FROM investment_case_versions WHERE ticker=%s ORDER BY revision DESC LIMIT 1',(ticker,));case=c.fetchone()
                        if not case or not case['body'].get('assumptions'):return jsonify(error='Save your baseline investment case and assumptions first.'),409
                    # Initial enrollment establishes a baseline without backfilling old documents.
                    seen=row['seen'] if row else inventory(c,ticker)
                    c.execute('''INSERT INTO thesis_monitors(ticker,enabled,revision,seen) VALUES(%s,%s,1,%s::jsonb)
                        ON CONFLICT(ticker) DO UPDATE SET enabled=EXCLUDED.enabled,revision=thesis_monitors.revision+1,updated_at=NOW()
                        RETURNING *''',(ticker,data['enabled'],json.dumps(seen)));row=c.fetchone()
                if not row:return jsonify(enabled=False,revision=0,initial=True)
                result={k:row[k] for k in ('enabled','revision','checked_at','last_result','updated_at')}
                result['pending']=row['pending'];result['baselineSources']=len(row['seen'])
                return jsonify(**result)
        except ValueError as exc:return jsonify(error=str(exc)),409

    @bp.post('/api/agent/advance-thesis-monitors')
    def advance():
        ensure(get_db)
        with get_db() as (_,c):
            c.execute('SELECT ticker FROM thesis_monitors WHERE enabled ORDER BY checked_at NULLS FIRST,ticker LIMIT 3')
            tickers=[r['ticker'] for r in c.fetchall()]
        results=[]
        for ticker in tickers:
            # Reserve under a per-company lock, then commit before submission.
            # Duplicate ticks reuse the immutable reservation and stable job ID.
            with get_db(commit=True) as (connection,c):
                c.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS locked',('thesis-monitor:'+ticker,))
                if not c.fetchone()['locked']:continue
                c.execute('SELECT * FROM thesis_monitors WHERE ticker=%s',(ticker,));row=c.fetchone()
                if not row['enabled']:continue
                try:
                    current=inventory(c,ticker)
                    names=candidate(current,row['seen'])
                    pending=row['pending']
                    if not pending and names:
                        c.execute('SELECT revision FROM investment_case_versions WHERE ticker=%s ORDER BY revision DESC LIMIT 1',(ticker,));case=c.fetchone()
                        if not case:raise ValueError('No saved investment case baseline.')
                        fingerprints={n:current[n] for n in names}
                        identity=json.dumps([ticker,case['revision'],fingerprints],sort_keys=True)
                        pending=dict(requestId=str(uuid.uuid5(uuid.NAMESPACE_URL,'charlie:source-monitor:'+identity)),
                                     revision=case['revision'],filenames=names,expectedStoredHashes=fingerprints,
                                     instructions='Assess newly imported evidence against the saved thesis assumptions and underweight reconsideration conditions. Explain materiality, contrary evidence, uncertainty and what remains unresolved. Do not force a thesis edit when no material change is supported.')
                    if pending:
                        c.execute('UPDATE thesis_monitors SET pending=%s::jsonb WHERE ticker=%s',(json.dumps(pending),ticker))
                        connection.commit()
                        # Stable reservation survives process loss; concurrent ticks replay the same job ID.
                        response=submit(ticker,pending,target='investment_case')
                        body,status=response if isinstance(response,tuple) else (response,response.status_code)
                        receipt=body.get_json()
                        if status not in (200,202):
                            # A waiting approval or changed case is re-evaluated next tick;
                            # no source is marked processed without a confirmed job.
                            result=dict(state='blocked',message=receipt.get('error','Submission blocked'))
                            # Keep a reservation across transient blockers; stale inputs are explicit.
                        else:
                            if receipt.get('jobId')!=pending['requestId']:raise ValueError('Unexpected proposal receipt.')
                            seen={**row['seen'],**pending['expectedStoredHashes']}
                            c.execute('UPDATE thesis_monitors SET seen=%s::jsonb WHERE ticker=%s',(json.dumps(seen),ticker))
                            result=dict(state='submitted',proposalId=receipt['jobId'],filenames=pending['filenames'],message='Evidence comparison queued. Approval is required for case changes.')
                            pending=None
                    else:
                        result=dict(state='watching',message='No unassessed source content detected.')
                    c.execute('UPDATE thesis_monitors SET pending=%s::jsonb,last_result=%s::jsonb,checked_at=NOW() WHERE ticker=%s',
                              (json.dumps(pending),json.dumps(result),ticker))
                except ValueError as exc:
                    result=dict(state='blocked',message=str(exc))
                    c.execute('UPDATE thesis_monitors SET last_result=%s::jsonb,checked_at=NOW() WHERE ticker=%s',(json.dumps(result),ticker))
                results.append(dict(ticker=ticker,**result))
        return jsonify(outcomes=results)

# Existing agent heartbeats wake this worker; no local-agent restart is needed.
import threading
import time
import logging
_wake_lock=threading.Lock()
_last_wake=0.0

def wake(app):
    global _last_wake
    if not _wake_lock.acquire(blocking=False):return
    if time.monotonic()-_last_wake<60:
        _wake_lock.release();return
    _last_wake=time.monotonic()
    def run():
        try:
            with app.app_context():
                handler=app.view_functions.get('research_amendments.advance')
                if handler:handler()
        except Exception:
            logging.getLogger(__name__).exception('Thesis monitor scan failed; next heartbeat can retry')
        finally:_wake_lock.release()
    threading.Thread(target=run,daemon=True,name='thesis-monitor-scan').start()
