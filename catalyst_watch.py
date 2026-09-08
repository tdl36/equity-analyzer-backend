"""Prospective company-news signals. Keyword matches trigger research, not conclusions."""
import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from flask import Blueprint, jsonify, request
import requests
from collection_control import command

KEY='catalyst_watch_v1'
STAGE='catalyst_signal'
RULES={
 'clinical':r'\b(phase [123i]+|clinical trial|primary endpoint|topline|top-line|readout|read-out)\b',
 'regulatory':r'\b(fda approv|fda reject|complete response letter|clinical hold|recall)',
 'guidance':r'\b(raises? guidance|cuts? guidance|lowers? guidance|withdraws? guidance|earnings results|quarterly results)\b',
 'corporate':r'\b(acquires?|acquisition|merger agreement|definitive agreement|divestiture)\b'}


def candidate(ticker,article,after,now):
    if not isinstance(article,dict):return None
    title=article.get('headline');url=article.get('url');ts=article.get('datetime')
    if not isinstance(title,str) or not isinstance(url,str) or type(ts) not in (int,float) or not after<=ts<=now+300:return None
    parsed=urlparse(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:return None
    # Scheduled presentations/rumours alone are not result announcements.
    if re.search(r'\b(will present|to present|will announce|to announce|preview|rumou?r|could acquire|may acquire)\b',title,re.I):return None
    category=next((k for k,p in RULES.items() if re.search(p,title,re.I)),None)
    if not category:return None
    if category=='clinical' and not re.search(r'\b(results?|met|meets|failed|fails|positive|negative|readout|read-out|topline|top-line)\b',title,re.I):return None
    related=article.get('related')
    if isinstance(related,str) and related.strip() and ticker not in [x.strip().upper() for x in related.split(',')]:return None
    # Same ticker, normalized headline and publication day deduplicate syndication.
    identity=hashlib.sha256((ticker+'|'+datetime.fromtimestamp(ts,timezone.utc).date().isoformat()+'|'+re.sub(r'\W+',' ',title.lower()).strip()).encode()).hexdigest()
    third_party_purchase=category=='corporate' and bool(re.search(r'\b(acquires?|purchases?|buys?)\b.*\b(gpus?|compute cluster|servers?|hardware|equipment)\b',title,re.I))
    issuer_explicit=bool(re.match(r'^'+re.escape(ticker)+r'\b',title.strip(),re.I))
    requires_review=third_party_purchase or (category=='corporate' and not issuer_explicit)
    triage='Hardware purchase may be a customer event, not a material issuer transaction.' if third_party_purchase else 'Verify the transaction parties and issuer relevance before automatic collection.' if requires_review else 'Potential catalyst; original-source verification still required.'
    return {'id':identity,'ticker':ticker,'title':title[:500],'url':url[:2000],'publishedAt':ts,'category':category,'requiresReview':requires_review,'triageReason':triage,
            'reason':f'Potential {category} catalyst: {title[:500]}. Rule-based news signal; confirm company, event and materiality in primary sources.'}


def decode(v):
    if isinstance(v,str):
        try:return json.loads(v)
        except ValueError:return {}
    return v if isinstance(v,dict) else {}


class CatalystWatch:
    def __init__(self,app,get_db,is_agent):
        self.app=app;self.db=get_db;self.lock=threading.Lock()
        bp=Blueprint('catalyst_watch',__name__);self.blueprint=bp
        bp.add_url_rule('/api/research/catalyst-watch','status',self.status,methods=['GET','POST'])
        @bp.route('/api/agent/catalyst-scan',methods=['POST'])
        def scan():
            if not is_agent():return jsonify(error='Local agent API key required.'),403
            if not self.lock.acquire(False):return jsonify(status='busy')
            def work():
                try:
                    with self.app.app_context():self.tick()
                except Exception:self.app.logger.warning('Catalyst scan failed; next agent cycle can retry.')
                finally:self.lock.release()
            threading.Thread(target=work,daemon=True).start();return jsonify(status='scan_requested'),202

    def load(self,cur):
        cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(KEY,))
        cur.execute('SELECT value FROM app_settings WHERE key=%s',(KEY,));r=cur.fetchone()
        return decode(r['value']) if r else {'enabled':False,'automatic':False,'tickers':[],'dailyLimit':10,'cursor':0,'used':0,'day':'','lastCheck':0}

    def save(self,cur,state):
        cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(KEY,json.dumps(state)))

    def status(self):
        with self.db(commit=True) as (_,cur):
            state=self.load(cur)
            if request.method=='POST':
                d=request.get_json(silent=True)
                if not isinstance(d,dict) or type(d.get('enabled'))!=bool or type(d.get('automatic'))!=bool:return jsonify(error='Choose enabled and automatic settings.'),400
                tickers=d.get('tickers');limit=d.get('dailyLimit')
                if not isinstance(tickers,list) or not 1<=len(tickers)<=100 or any(not isinstance(t,str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',t) for t in tickers):return jsonify(error='Choose 1–100 valid tickers.'),400
                if type(limit)!=int or not 1<=limit<=30:return jsonify(error='Daily automatic event limit must be 1–30.'),400
                if d['enabled'] and not state['enabled']:state['enabledAt']=datetime.now(timezone.utc).timestamp()
                state.update(enabled=d['enabled'],automatic=d['automatic'],tickers=sorted(set(tickers)),dailyLimit=limit)
                self.save(cur,state)
            cur.execute('SELECT id,ticker,status,input,result,created_at FROM mp_jobs WHERE stage=%s ORDER BY created_at DESC LIMIT 50',(STAGE,));events=[dict(r) for r in cur.fetchall()]
            cur.execute("SELECT value FROM app_settings WHERE key='finnhub_api_key'");key=cur.fetchone()
        response=jsonify(config=state,events=events,hasNewsKey=bool(key and key['value']),scope='One company-news lookup per minute while the Mac agent is connected; a full sweep takes roughly one minute per ticker. New signals only after enabling. Keyword detection is incomplete and can misclassify materiality. Automatic event runs create recap drafts; thesis changes require review.')
        response.headers['Cache-Control']='no-store';return response

    def tick(self):
        now=datetime.now(timezone.utc);stamp=now.timestamp()
        with self.db(commit=True) as (_,cur):
            state=self.load(cur)
            if not state['enabled'] or not state['tickers'] or stamp-state.get('lastCheck',0)<55:return
            cur.execute("SELECT value FROM app_settings WHERE key='finnhub_api_key'");key=cur.fetchone()
            if not key or not key['value']:
                state.update(lastCheck=stamp,lastIssue='Save a Finnhub news API key in Settings to enable detection.');self.save(cur,state);return
            ticker=state['tickers'][state.get('cursor',0)%len(state['tickers'])]
            state.update(cursor=state.get('cursor',0)+1,lastCheck=stamp,lastTicker=ticker,lastIssue=None);self.save(cur,state)
            api_key=str(key['value']).strip();after=max(state.get('enabledAt',stamp),stamp-2*86400)
        try:
            r=requests.get('https://finnhub.io/api/v1/company-news',params={'symbol':ticker,'from':(now-timedelta(days=2)).date().isoformat(),'to':now.date().isoformat()},headers={'X-Finnhub-Token':api_key},timeout=15)
            r.raise_for_status();articles=r.json()
            if not isinstance(articles,list):raise ValueError('Invalid news response')
        except Exception:
            with self.db(commit=True) as (_,cur):
                state=self.load(cur);state['lastIssue']='Company-news lookup failed. Check news access/key; this ticker will be retried on a later sweep.';self.save(cur,state)
            return
        signals=[c for a in articles[:300] if (c:=candidate(ticker,a,after,stamp))]
        with self.db(commit=True) as (_,cur):
            state=self.load(cur)
            if not state['enabled'] or ticker not in state['tickers']:return
            day=now.date().isoformat()
            if state.get('day')!=day:state.update(day=day,used=0)
            cur.execute("SELECT value FROM app_settings WHERE key='collection_control_snapshot'");row=cur.fetchone();snapshot=decode(row['value']) if row else {}
            policy=next((p for p in snapshot.get('policies',[]) if p.get('ticker')==ticker and p.get('enabled')),None)
            cur.execute('SELECT id FROM analysts WHERE %s=ANY(coverage_tickers) LIMIT 1',(ticker,));analyst=cur.fetchone()
            for signal in signals:
                jid=str(uuid.uuid5(uuid.NAMESPACE_URL,'charlie-catalyst:'+signal['id']))
                cur.execute('SELECT id FROM mp_jobs WHERE id=%s',(jid,))
                if cur.fetchone():continue
                status='detected';result={}
                if not signal.get('requiresReview') and state['automatic'] and policy and analyst and state['used']<state['dailyLimit']:
                    cfg={**policy,'createFolder':True,'workflow':'recap','topic':f"{ticker} {now.date().isoformat()} {signal['category']} {signal['id'][:8]}",'lookbackDays':7,'kinds':['press-release','broker-report','transcript'],'instructions':signal['reason']+' Source URL: '+signal['url']}
                    cid=str(uuid.uuid5(uuid.NAMESPACE_URL,'charlie-event-refresh:'+signal['id']))
                    value=command({'requestId':cid,'action':'event_refresh','payload':{'policy':cfg,'event':{'id':signal['id'],'reason':signal['reason'],'url':signal['url']}}})
                    cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,'collection_control',%s,'queued',%s::jsonb) ON CONFLICT(id) DO NOTHING",(cid,ticker,json.dumps(value)))
                    status='collection_queued';result={'commandId':cid};state['used']+=1
                elif signal.get('requiresReview'):
                    status='needs_review';result={'reason':signal['triageReason']}
                else:result={'reason':'Automatic collection off, daily limit reached, no enabled policy, or no covering analyst. Use the collection controls to research this signal.'}
                cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result) VALUES(%s,%s,%s,%s,%s::jsonb,%s::jsonb)",(jid,STAGE,ticker,status,json.dumps(signal),json.dumps(result)))
            state['lastSuccessAt']=stamp
            if len(articles)>300:state['lastIssue']='News response exceeded 300 articles; this scan was limited.'
            self.save(cur,state)
