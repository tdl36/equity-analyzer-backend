"""Ticker refresh policies and resumable work for the scheduled browser worker."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
import uuid
from charlie_collector import Collector, DEFAULT_STATE, DEFAULT_STOCKS, ticker_name, now

KINDS = ('transcript', 'broker-report', 'press-release', 'presentation')
CADENCES = (0, 1, 4, 12, 24, 168)
TERMINAL = ('complete', 'cancelled')


class RefreshManager:
    def __init__(self, collector, clock=time.time):
        self.c = collector
        self.db = collector.db
        self.clock = clock
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS refresh_policies (
                ticker TEXT PRIMARY KEY, config TEXT NOT NULL, next_due REAL,
                last_success TEXT, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS refresh_requests (
                id TEXT PRIMARY KEY, ticker TEXT, run TEXT UNIQUE, config TEXT,
                status TEXT, created REAL, lease_until REAL, owner TEXT,
                issue TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS refresh_worker (
                id INTEGER PRIMARY KEY CHECK(id=1), checked REAL, status TEXT);
        ''')

    def validate(self, config):
        ticker = ticker_name(config.get('ticker', ''))
        cadence = config.get('hours', 0)
        days = config.get('lookbackDays', 30)
        if type(cadence) is not int or not 0 <= cadence <= 8760:
            raise ValueError('Choose manual (0) or an interval of 1–8760 hours')
        if type(days) is not int or not 1 <= days <= 365:
            raise ValueError('Lookback must be 1–365 days')
        if not isinstance(config.get('enabled', True), bool):
            raise ValueError('Enabled must be true or false')
        instructions = config.get('instructions', '')
        if not isinstance(instructions, str) or len(instructions) > 3000:
            raise ValueError('Instructions must be at most 3,000 characters')
        kinds = config.get('kinds', [k for k in KINDS if k != 'presentation'])
        if not isinstance(kinds, list) or not kinds or any(k not in KINDS for k in kinds):
            raise ValueError('Choose supported source types: transcripts, broker reports, press releases or presentations')
        workflow = config.get('workflow', 'thesis')
        if workflow not in ('thesis', 'note', 'recap'):
            raise ValueError('Choose thesis intake, research note or event recap')
        topic = config.get('topic', '').strip()
        if workflow == 'recap':
            if not topic or len(topic) > 160 or Path(topic).name != topic or topic.startswith('.') or '\\' in topic:
                raise ValueError('Event recap needs a plain event folder name')
            ticker_folder = self.c.catalysts / ticker
        else:
            topic = ''
            ticker_folder = self.c.stocks / ticker
        create_folder=config.get('createFolder',False)
        if type(create_folder)!=bool:raise ValueError('createFolder must be true or false')
        if ticker_folder.is_symlink() or (not ticker_folder.is_dir() and not (create_folder and not ticker_folder.exists() and ticker_folder.parent.is_dir())):
            raise ValueError(f'Existing iCloud ticker folder required: {ticker}')
        return dict(ticker=ticker, hours=cadence, lookbackDays=days,
                    enabled=config.get('enabled', True), createFolder=create_folder, instructions=instructions.strip(),
                    kinds=list(dict.fromkeys(kinds)), workflow=workflow, topic=topic)

    def save(self, config):
        with self.c.lock():
            return self._save(config)

    def _save(self, config):
        cfg = self.validate(config)
        if cfg['createFolder']:
            root=self.c.catalysts if cfg['workflow']=='recap' else self.c.stocks
            (root/cfg['ticker']).mkdir(exist_ok=True)
        old = self.db.execute('SELECT * FROM refresh_policies WHERE ticker=?', (cfg['ticker'],)).fetchone()
        due = self.clock() + cfg['hours'] * 3600 if cfg['enabled'] and cfg['hours'] else None
        if old and json.loads(old['config']).get('hours') == cfg['hours'] and json.loads(old['config']).get('enabled') == cfg['enabled']:
            due = old['next_due']
        self.db.execute('''INSERT INTO refresh_policies(ticker,config,next_due,updated) VALUES(?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET config=excluded.config,next_due=excluded.next_due,updated=excluded.updated''',
            (cfg['ticker'], json.dumps(cfg), due, now()))
        return cfg

    def apply_cloud_command(self, command_id, value):
        # Receipt and mutation commit together. Even after a successful collection
        # finishes, redelivery cannot create another refresh request.
        with self.c.lock():
            self.db.execute('CREATE TABLE IF NOT EXISTS cloud_control_receipts(id TEXT PRIMARY KEY, input TEXT, result TEXT)')
            encoded=json.dumps(value,sort_keys=True)
            old=self.db.execute('SELECT input,result FROM cloud_control_receipts WHERE id=?',(command_id,)).fetchone()
            if old:
                if old['input']!=encoded:raise ValueError('Cloud command ID conflict')
                return json.loads(old['result'])
            if value['action']=='research_task':
                from research_commands import plan
                raw=value['payload'];p=plan({**raw,'date':raw['until'],'days':(datetime.fromisoformat(raw['until'])-datetime.fromisoformat(raw['since'])).days+1})
                tk=p['ticker'];topic=f"{tk} {p['until']} {p['kind']} {command_id[:8]}"
                cfg=dict(ticker=tk,hours=0,enabled=True,createFolder=True,lookbackDays=7,workflow='recap',topic=topic,kinds=[k for k in KINDS if k!='presentation' or p.get('meetingPrep')],instructions=p['instruction'])
                self.validate(cfg);(self.c.catalysts/tk).mkdir(exist_ok=True)
                if not self.db.execute('SELECT ticker FROM refresh_policies WHERE ticker=?',(tk,)).fetchone():self._save(cfg)
                rid=self._enqueue({'ticker':tk,'config':json.dumps(cfg),'last_success':None},manual=True,event={'id':command_id,'reason':p['instruction'],'url':'https://www.sec.gov/edgar/search/'},command=p)
                result={'refreshRequestId':rid,'topic':topic,'plan':p}
            elif value['action']=='save':result={'policy':self._save(value['payload'])}
            elif value['action']=='save_trigger':
                cfg=self._save(value['payload'])
                row=self.db.execute('SELECT * FROM refresh_policies WHERE ticker=?',(cfg['ticker'],)).fetchone()
                result={'policy':cfg,'refreshRequestId':self._enqueue(row,manual=True)}
            elif value['action']=='save_batch':
                configs=[self.validate(p) for p in value['payload']['policies']]
                result={'policies':[self._save(p) for p in configs]}
            elif value['action'] in ('cancel','retry'):
                rid=value['payload']['refreshRequestId']
                row=self.db.execute('SELECT ticker FROM refresh_requests WHERE id=?',(rid,)).fetchone()
                if not row or row['ticker']!=value['payload']['ticker']:raise ValueError('Refresh request does not match the selected ticker')
                if value['action']=='cancel':self._cancel(rid)
                else:self._retry(rid)
                result={'refreshRequestId':rid,'action':value['action']}
            elif value['action']=='event_refresh':
                cfg=value['payload']['policy'];event=value['payload']['event']
                existing=self.db.execute("SELECT id FROM refresh_requests WHERE json_extract(config,'$.eventId')=?",(event['id'],)).fetchone()
                if existing:result={'refreshRequestId':existing['id']}
                else:
                    checked=self.validate(cfg)
                    if checked.get('createFolder'):(self.c.catalysts/checked['ticker']).mkdir(exist_ok=True)
                    if not self.db.execute('SELECT ticker FROM refresh_policies WHERE ticker=?',(cfg['ticker'],)).fetchone():raise ValueError('Save a coverage policy before event-triggered collection')
                    result={'refreshRequestId':self._enqueue({'config':json.dumps(cfg),'last_success':None,'ticker':cfg['ticker']},event=event)}
            elif value['action']=='trigger':
                ticker=ticker_name(value['payload']['ticker'])
                row=self.db.execute('SELECT * FROM refresh_policies WHERE ticker=?',(ticker,)).fetchone()
                if not row:raise ValueError('Save this ticker policy first')
                result={'refreshRequestId':self._enqueue(row,manual=True)}
            else:raise ValueError('Unsupported cloud command')
            self.db.execute('INSERT INTO cloud_control_receipts VALUES(?,?,?)',(command_id,encoded,json.dumps(result)))
            return result

    def _enqueue(self, row, manual=False, event=None, command=None):
        # Caller holds the collector transaction/operation lock.
        existing = self.db.execute("SELECT id FROM refresh_requests WHERE ticker=? AND status NOT IN ('complete','cancelled') ORDER BY created LIMIT 1", (row['ticker'],)).fetchone()
        if existing and not event:
            if manual:
                prior=self.db.execute('SELECT config FROM refresh_requests WHERE id=?',(existing['id'],)).fetchone()
                content=json.loads(prior['config']);content['manual']=True
                self.db.execute('UPDATE refresh_requests SET config=? WHERE id=?',(json.dumps(content),existing['id']))
            return existing['id']
        cfg = self.validate(json.loads(row['config']))
        if manual:cfg['manual']=True
        if event:cfg.update(eventId=event['id'],eventReason=event['reason'],eventUrl=event['url'])
        if command:cfg['researchCommand']=command
        today = datetime.fromtimestamp(self.clock(), timezone.utc).date()
        since = today - timedelta(days=cfg['lookbackDays'] - 1)
        if row['last_success']:
            since = min(today, datetime.fromisoformat(row['last_success']).date() - timedelta(days=2))
        if command:
            since=datetime.fromisoformat(command['since']).date();today=datetime.fromisoformat(command['until']).date()
        topic = cfg['topic'] or None
        if topic:
            folder = self.c.catalysts / cfg['ticker'] / topic
            if folder.is_symlink() or not folder.resolve().is_relative_to(self.c.catalysts.resolve()):
                raise ValueError('Event folder must stay inside CATALYSTS')
            folder.mkdir(exist_ok=True)
            if command:(folder/'.charlie-collection-pending').write_text(event['id'])
        run, request_id = uuid.uuid4().hex[:12], str(uuid.uuid4())
        self.db.execute('INSERT INTO runs(id,created,since,until_date,topic) VALUES(?,?,?,?,?)',
                        (run, now(), since.isoformat(), today.isoformat(), topic))
        self.db.executemany('INSERT INTO tasks(run,ticker,kind) VALUES(?,?,?)', [(run,cfg['ticker'],k) for k in cfg['kinds']])
        self.db.execute("INSERT INTO refresh_requests(id,ticker,run,config,status,created) VALUES(?,?,?,?,'queued',?)",
                        (request_id,cfg['ticker'],run,json.dumps(cfg),self.clock()))
        if not manual and not event:
            self.db.execute('UPDATE refresh_policies SET next_due=? WHERE ticker=?',
                            (self.clock()+cfg['hours']*3600 if cfg['hours'] else None,cfg['ticker']))
        self.c.event(run, 'refresh_queued', requestId=request_id, workflow=cfg['workflow'])
        return request_id

    def trigger(self, ticker):
        with self.c.lock():
            row = self.db.execute('SELECT * FROM refresh_policies WHERE ticker=?', (ticker_name(ticker),)).fetchone()
            if not row:
                raise ValueError('Save ticker settings before refreshing')
            return self._enqueue(row,manual=True)

    def due(self):
        with self.c.lock():
            rows = self.db.execute('SELECT * FROM refresh_policies WHERE next_due IS NOT NULL AND next_due<=?', (self.clock(),)).fetchall()
            return [self._enqueue(r) for r in rows if json.loads(r['config'])['enabled']]

    def status(self):
        policies = []
        for r in self.db.execute('SELECT * FROM refresh_policies ORDER BY ticker'):
            policies.append(dict(json.loads(r['config']), nextDue=r['next_due'], lastSuccess=r['last_success']))
        requests = [dict(r) for r in self.db.execute('SELECT id,ticker,run,status,created,issue,result FROM refresh_requests ORDER BY created DESC LIMIT 50')]
        for r in requests:
            r['result'] = json.loads(r['result']) if r['result'] else None
        worker = self.db.execute('SELECT checked,status FROM refresh_worker WHERE id=1').fetchone()
        folders={}
        for name,root in [('stocks',self.c.stocks),('catalysts',self.c.catalysts)]:
            values=[]
            if root.is_dir():
                for path in root.iterdir():
                    if not path.is_dir() or path.is_symlink():continue
                    try:
                        if ticker_name(path.name)==path.name:values.append(path.name)
                    except ValueError:pass
            folders[name]=sorted(values)
        return {'policies':policies,'requests':requests,'worker':dict(worker) if worker else None,'folders':folders}

    def claim(self):
        self.due()
        with self.c.lock():
            stamp = self.clock()
            self.db.execute("INSERT INTO refresh_worker VALUES(1,?,'awake') ON CONFLICT(id) DO UPDATE SET checked=excluded.checked,status=excluded.status", (stamp,))
            if self.db.execute("SELECT id FROM refresh_requests WHERE lease_until>? AND status IN ('collecting','verifying')", (stamp,)).fetchone():
                return None
            row = self.db.execute("""SELECT q.* FROM refresh_requests q JOIN refresh_policies p ON p.ticker=q.ticker
                WHERE q.status IN ('queued','collecting','verifying') AND (q.lease_until IS NULL OR q.lease_until<=?)
                AND (json_extract(p.config,'$.enabled')=1 OR json_extract(q.config,'$.manual')=1) ORDER BY q.created LIMIT 1""", (stamp,)).fetchone()
            if not row:
                return None
            owner = str(uuid.uuid4())
            self.db.execute("UPDATE refresh_requests SET status='collecting',owner=?,lease_until=?,issue=NULL WHERE id=?", (owner,stamp+1800,row['id']))
            result = dict(row)
            result.update(owner=owner, lease_until=stamp+1800, status='collecting', config=json.loads(row['config']), collection=self.c.status(row['run']))
            return result

    def include_meeting_presentations(self, request_id, owner):
        """Repair a pre-presentation meeting request without changing its dates or destination."""
        with self.c.lock():
            row=self.db.execute("SELECT * FROM refresh_requests WHERE id=? AND owner=? AND lease_until>? AND status='collecting'",(request_id,owner,self.clock())).fetchone()
            if not row:raise ValueError('An active owned collection is required')
            cfg=json.loads(row['config'])
            if not (cfg.get('researchCommand') or {}).get('meetingPrep'):raise ValueError('Only guided meeting assignments request this repair')
            if 'presentation' not in cfg['kinds']:
                cfg['kinds'].append('presentation')
                self.db.execute('UPDATE refresh_requests SET config=? WHERE id=?',(json.dumps(cfg),request_id))
                self.db.execute('INSERT OR IGNORE INTO tasks(run,ticker,kind) VALUES(?,?,?)',(row['run'],row['ticker'],'presentation'))
                self.c.event(row['run'],'meeting_presentation_support_added',requestId=request_id,
                    reason='Honor the existing meeting assignment request for earnings presentations; preserve all original searches and source windows.')
            return {'requestId':request_id,'kinds':cfg['kinds']}

    def mark(self, request_id, owner, status, issue=''):
        if status not in ('collecting','needs_auth','attention','queued'):
            raise ValueError('Invalid worker status')
        with self.c.lock():
            row = self.db.execute('SELECT * FROM refresh_requests WHERE id=? AND owner=? AND lease_until>?', (request_id,owner,self.clock())).fetchone()
            if not row:
                raise ValueError('Worker lease expired or belongs to another worker')
            self.db.execute('UPDATE refresh_requests SET status=?,issue=?,lease_until=? WHERE id=?',
                            (status,str(issue)[:1500] or None,self.clock()+1800 if status=='collecting' else None,request_id))

    def cancel(self, request_id):
        with self.c.lock():return self._cancel(request_id)

    def _cancel(self, request_id):
        row = self.db.execute('SELECT status FROM refresh_requests WHERE id=?', (request_id,)).fetchone()
        if not row or row['status'] == 'complete':
            raise ValueError('A completed refresh cannot be cancelled')
        self.db.execute("UPDATE refresh_requests SET status='cancelled',lease_until=NULL,owner=NULL,issue='Cancelled; validated originals retained' WHERE id=?", (request_id,))

    def retry(self, request_id):
        with self.c.lock():return self._retry(request_id)

    def _retry(self, request_id):
        row = self.db.execute('SELECT run,status FROM refresh_requests WHERE id=?', (request_id,)).fetchone()
        if not row or row['status'] not in ('attention','needs_auth'):
            raise ValueError('Only a request needing attention can be resumed')
        # Restore authentication-paused tasks without discarding validated downloads.
        self.db.execute("UPDATE tasks SET status=COALESCE(paused_status,'pending'),paused_status=NULL,evidence_after=? WHERE run=? AND status='needs_auth'", (now(),row['run']))
        self.db.execute("UPDATE refresh_requests SET status='queued',lease_until=NULL,owner=NULL,issue=NULL WHERE id=?", (request_id,))

    def complete(self, request_id, owner, fetcher=None):
        row = self.db.execute('SELECT * FROM refresh_requests WHERE id=? AND owner=? AND lease_until>?', (request_id,owner,self.clock())).fetchone()
        if not row:
            raise ValueError('Active worker lease required')
        collection = self.c.status(row['run'])
        if any(t['status'] not in ('complete','complete_with_exceptions','no_results') for t in collection['tasks']):
            raise ValueError('All requested searches must be reviewed before completing a refresh')
        verification = self.c.verify(row['run'], fetcher=fetcher)
        if any(v.get('error') or v.get('missing') for v in verification['verifications']):
            raise ValueError('iCloud handoff verification has not passed')
        delivered = [d for d in collection['documents'] if d['usage']=='research' and d['status']=='handed_off']
        result = {'newDocuments':len(delivered),'heldDocuments':sum(d['usage']=='reference_only' for d in collection['documents']),
                  'verification':verification, 'research':'existing Charlie intake' if delivered else 'no new eligible documents'}
        cfg = json.loads(row['config'])
        public_count=0
        if cfg.get('researchCommand'):
            from catalyst_sources import inventory, fingerprint, record_dispatch
            folder = self.c.catalysts/cfg['ticker']/cfg['topic']
            source_inventory, source_issues = inventory(folder, allow_pending=True)
            if source_issues: raise ValueError('Source inventory needs attention before dispatch')
            dispatch_revision = fingerprint(source_inventory)
            from research_task_sources import verify_public_sources
            public_count=verify_public_sources(self,row,fetcher)
            result['publicDocuments']=public_count
            from command_source_import import import_sources
            result['importedOriginals']=import_sources(self,row)
            if not delivered and not public_count:
                result['research']='No eligible sources found in the verified searches; no recap generated'
        if (delivered or public_count) and cfg['workflow'] in ('note','recap'):
            # Reserve dispatch before the network call. Uncertain POSTs are never replayed blindly.
            with self.c.lock():
                reserved = self.db.execute('UPDATE refresh_requests SET result=? WHERE id=? AND owner=? AND lease_until>? AND result IS NULL',
                    (json.dumps({'research':'dispatch reserved'}),request_id,owner,self.clock()))
                if not reserved.rowcount:
                    raise ValueError('Research dispatch already attempted or worker lease revoked; inspect existing job before retrying')
            try:
                result['research'] = self.dispatch(cfg, delivered)
            except Exception as exc:
                self.mark(request_id, owner, 'attention', 'Research dispatch needs inspection before retry: ' + type(exc).__name__)
                raise ValueError('Research dispatch did not confirm completion; inspect existing research jobs before retrying') from exc
        with self.c.lock():
            updated = self.db.execute("UPDATE refresh_requests SET status='complete',lease_until=NULL,result=?,issue=NULL WHERE id=? AND owner=? AND lease_until>?", (json.dumps(result),request_id,owner,self.clock()))
            if not updated.rowcount:
                raise ValueError('Refresh was cancelled or reassigned during verification')
            if cfg.get('researchCommand'):
                from catalyst_sources import inventory, fingerprint, record_dispatch
                folder = self.c.catalysts/cfg['ticker']/cfg['topic']
                sources, issues = inventory(folder, allow_pending=True)
                if issues or fingerprint(sources) != dispatch_revision:
                    raise ValueError('Source inventory changed before dispatch acknowledgement; inspect the existing job before retrying')
                record_dispatch(folder, dispatch_revision, request_id)
                (folder/'.charlie-collection-pending').unlink(missing_ok=True)
            if not cfg.get('eventId'):
                self.db.execute('UPDATE refresh_policies SET last_success=? WHERE ticker=?', (collection['until_date'],row['ticker']))
        return result

    def dispatch(self, cfg, documents):
        import requests
        from charlie_local_agent import CHARLIE_API, _agent_headers, get_secret, load_api_key_from_config
        headers = _agent_headers()
        if cfg['workflow']=='note':
            key = get_secret('ANTHROPIC_API_KEY') or load_api_key_from_config()
            if not key:
                raise ValueError('Local research API key unavailable; no note dispatched')
            body = {'ticker':cfg['ticker'],'apiKey':key,'runOn':'agent','mode':'update',
                    'fileSelection':[{'filename':Path(d['destination']).name,'folder':'main'} for d in documents]}
            r = requests.post(CHARLIE_API+'/api/notes/generate',headers=headers,json=body,timeout=30)
        else:
            from catalyst_sources import inventory, fingerprint
            sources, issues = inventory(self.c.catalysts / cfg['ticker'] / cfg['topic'], allow_pending=bool(cfg.get('researchCommand')))
            if issues: raise ValueError('Event sources need attention before recap generation')
            revision = fingerprint(sources)
            r = requests.post(CHARLIE_API+'/api/analysts/queue-catalyst-activity',headers=headers,
                json={'ticker':cfg['ticker'],'topic':cfg['topic'],'fingerprint':revision,'fileCount':len(sources),'customInstructions':cfg['instructions'],'deferAutomaticRun':bool(cfg.get('researchCommand')),'coordinated':(cfg.get('researchCommand') or {}).get('coordinated',False)},timeout=30)
            r.raise_for_status()
            created = r.json().get('created',[])
            pending = requests.get(CHARLIE_API+'/api/analyst-activities/pending',headers=headers,timeout=30)
            pending.raise_for_status()
            candidates = [a for a in pending.json().get('activities',[]) if a.get('ticker')==cfg['ticker'] and (a.get('input') or {}).get('topic')==cfg['topic'] and (a.get('input') or {}).get('fingerprint')==revision]
            for a in candidates:
                if not any(x['activityId']==a['id'] for x in created):
                    created.append({'activityId':a['id'], 'autoRan':a.get('status')=='running' or bool((a.get('output') or {}).get('synthesisMarkdown'))})
            for activity in created:
                if not activity.get('autoRan'):
                    resp = requests.post(CHARLIE_API+'/api/analyst-activities/'+activity['activityId']+'/run',headers=headers,json={'customInstructions':cfg['instructions']},timeout=30)
                    resp.raise_for_status()
            if not created:
                if r.json().get('count',0)>0: return {'status':'Existing activity already routed; inspect Analyst Inbox'}
                raise ValueError('No covering analyst; configure coverage before generating a recap')
            return {'activities':created}
        r.raise_for_status()
        return r.json()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,default=DEFAULT_STATE)
    parser.add_argument('--stocks',type=Path,default=DEFAULT_STOCKS)
    parser.add_argument('command',choices=['status','due','claim','trigger','mark','complete','retry','cancel'])
    parser.add_argument('--ticker');parser.add_argument('--request');parser.add_argument('--owner')
    parser.add_argument('--status');parser.add_argument('--issue',default='')
    args=parser.parse_args();c=Collector(args.state,args.stocks);m=RefreshManager(c)
    try:
        if args.command=='trigger':result=m.trigger(args.ticker)
        elif args.command=='mark':result=m.mark(args.request,args.owner,args.status,args.issue)
        elif args.command=='complete':result=m.complete(args.request,args.owner)
        elif args.command=='retry':result=m.retry(args.request)
        elif args.command=='cancel':result=m.cancel(args.request)
        else:result=getattr(m,args.command)()
        print(json.dumps(result,indent=2))
    finally:c.db.close()

if __name__=='__main__':main()
