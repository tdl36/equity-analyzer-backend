"""Persisted new-document intake. Automatic drafts, never automatic thesis writes."""
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request

KEY='research_auto_intake_v1'
STAGE='auto_evidence_intake'
EXTENSIONS=('.pdf','.txt','.md','.csv','.tsv','.docx','.xlsx','.xlsm')


def eligible(ticker,name):
    import re
    return isinstance(ticker,str) and isinstance(name,str) and not name.startswith(('.', '~$')) and len(name)<=255 and '/' not in name and '\\' not in name and name.lower().endswith(EXTENSIONS) and bool(re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker))


def identity(ticker,name):
    return hashlib.sha256((ticker+'\0'+name).encode()).hexdigest()


def new_entries(seen, entries):
    seen=set(seen);fresh=[]
    for tk,name in entries:
        if eligible(tk,name) and identity(tk,name) not in seen:
            fresh.append((tk,name));seen.add(identity(tk,name))
    return sorted(seen),fresh


class ResearchAutomation:
    def __init__(self,app,get_db,submit,fetch_files,get_manifest,has_key,budget_block):
        self.app=app;self.db=get_db;self.submit=submit;self.fetch_files=fetch_files
        self.manifest=get_manifest;self.has_key=has_key;self.budget_block=budget_block
        self.running=threading.Lock()
        bp=Blueprint('research_automation',__name__);self.blueprint=bp
        bp.add_url_rule('/api/research/automation','status',self.status,methods=['GET','POST'])

    def load(self,cur):
        cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(KEY,))
        cur.execute('SELECT value FROM app_settings WHERE key=%s',(KEY,));row=cur.fetchone()
        return json.loads(row['value']) if row else {'enabled':False,'dailyLimit':2,'seen':[],'day':'','used':0,'manifestInitialized':False}

    def save(self,cur,state):
        cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(KEY,json.dumps(state)))

    def entries(self,manifest):
        return [(tk,f.get('filename')) for tk,files in (manifest if isinstance(manifest,dict) else {}).items() if isinstance(files,list) for f in files if isinstance(f,dict) and f.get('folder')=='main']

    def status(self):
        with self.db(commit=True) as (_,cur):
            state=self.load(cur)
            if request.method=='POST':
                data=request.get_json(silent=True)
                if not isinstance(data,dict) or not isinstance(data.get('enabled'),bool): return jsonify(error='enabled must be true or false'),400
                limit=data.get('dailyLimit',state['dailyLimit'])
                if type(limit)!=int or not 1<=limit<=5:return jsonify(error='Daily comparison limit must be 1–5'),400
                if data['enabled'] and not state['enabled']:
                    # Re-enabling starts prospectively. Previously queued intake is retained.
                    cur.execute('SELECT ticker,filename FROM document_files')
                    entries=[(r['ticker'],r['filename']) for r in cur.fetchall() or []]
                    manifest=self.manifest()
                    state['seen'],_=new_entries(state['seen'],entries+self.entries(manifest.get('manifest',{})))
                    state['manifestInitialized']=bool(manifest.get('timestamp'))
                    state['enabledAt']=datetime.now(timezone.utc).isoformat()
                state.update(enabled=data['enabled'],dailyLimit=limit)
                self.save(cur,state)
            cur.execute('SELECT id,ticker,status,input,result,error,created_at FROM mp_jobs WHERE stage=%s ORDER BY created_at DESC LIMIT 30',(STAGE,))
            events=[dict(r) for r in cur.fetchall() or []]
        response=jsonify(enabled=state['enabled'],dailyLimit=state['dailyLimit'],usedToday=state['used'] if state['day']==datetime.now(timezone.utc).date().isoformat() else 0,manifestInitialized=state['manifestInitialized'],hasServerKey=bool(self.has_key()),lastIssue=state.get('lastIssue'),events=events)
        response.headers['Cache-Control']='no-store'
        if request.method=='POST':self.wake()
        return response

    def observe(self,entries,initial_manifest=False):
        try:
            with self.db(commit=True) as (_,cur):
                state=self.load(cur)
                if not state['enabled']:return
                seen,fresh=new_entries(state['seen'],entries)
                if initial_manifest and not state['manifestInitialized']:
                    fresh=[];state['manifestInitialized']=True
                state['seen']=seen
                for tk,name in fresh:
                    event_id=str(uuid.uuid5(uuid.NAMESPACE_URL,'charlie-intake:'+identity(tk,name)))
                    cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,%s,%s,'waiting',%s::jsonb) ON CONFLICT(id) DO NOTHING",(event_id,STAGE,tk,json.dumps({'filename':name,'folder':'main'})))
                self.save(cur,state)
            self.wake()
        except Exception:self.app.logger.exception('Automatic intake observation failed')

    def wake(self):
        if not self.running.acquire(False):return
        def work():
            try:
                with self.app.app_context():self.tick()
            except Exception:self.app.logger.exception('Automatic intake worker failed')
            finally:self.running.release()
        threading.Thread(target=work,daemon=True).start()

    def tick(self):
        with self.db(commit=True) as (_,cur):
            state=self.load(cur)
            if not state['enabled']:return
            day=datetime.now(timezone.utc).date().isoformat()
            if state['day']!=day:state.update(day=day,used=0)
            issue= None if self.has_key() else 'A server-side model key is required for unattended comparisons.'
            if not issue:issue=self.budget_block()
            if not issue and state['used']>=state['dailyLimit']:issue='Daily comparison limit reached; pending sources remain queued.'
            state['lastIssue']=issue;self.save(cur,state)
            if issue:return
            # Interrupted imports are visible for manual investigation; never silently replay a model call.
            cur.execute("UPDATE mp_jobs SET status='attention',error='Intake interrupted; inspect the proposal list before retrying manually.' WHERE stage=%s AND status='processing' AND updated_at<NOW()-INTERVAL '30 minutes'",(STAGE,))
            cur.execute("""SELECT j.ticker FROM mp_jobs j WHERE j.stage=%s AND j.status='waiting'
              AND EXISTS(SELECT 1 FROM portfolio_analyses p WHERE p.ticker=j.ticker)
              AND NOT EXISTS(SELECT 1 FROM mp_jobs p WHERE p.ticker=j.ticker AND
                ((p.stage='evidence_amendment' AND p.status IN ('queued','running','awaiting_approval')) OR
                 (p.stage=%s AND p.status='processing')))
              AND j.created_at<NOW()-INTERVAL '2 minutes' ORDER BY j.created_at LIMIT 1""",(STAGE,STAGE))
            row=cur.fetchone()
            if not row:return
            tk=row['ticker']
            cur.execute("SELECT id,input FROM mp_jobs WHERE stage=%s AND ticker=%s AND status='waiting' AND created_at<NOW()-INTERVAL '2 minutes' ORDER BY created_at LIMIT 10 FOR UPDATE",(STAGE,tk))
            events=list(cur.fetchall() or []);ids=[r['id'] for r in events]
            files=[json.loads(r['input']) if isinstance(r['input'],str) else r['input'] for r in events]
            cur.execute("UPDATE mp_jobs SET status='processing',updated_at=NOW() WHERE id=ANY(%s)",(ids,))
            state['used']+=1;self.save(cur,state)
        error=None;job_id=None
        try:
            _,missing=self.fetch_files(tk,files)
            if missing:raise ValueError('Some files could not be imported from iCloud. No comparison was started.')
            # Pause takes effect before starting a model job, even if import was in progress.
            with self.db(commit=True) as (_,cur):
                if not self.load(cur)['enabled']:raise ValueError('Automation was paused during import. Compare these sources manually when ready.')
            job_id=str(uuid.uuid5(uuid.NAMESPACE_URL,'auto-proposal:'+','.join(sorted(ids))))
            response=self.submit(tk,{'requestId':job_id,'filenames':[f['filename'] for f in files]})
            reply,code=response if isinstance(response,tuple) else (response,200)
            if code not in (200,202):raise ValueError(reply.get_json().get('error','Comparison could not be queued.'))
        except ValueError as e:error=str(e)
        except Exception:error='Intake could not complete. Inspect source availability and proposal status.'
        with self.db(commit=True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET status=%s,result=%s::jsonb,error=%s,updated_at=NOW() WHERE id=ANY(%s)",('attention' if error else 'submitted',json.dumps({'proposalId':job_id}),error,ids))
