"""Versioned source preferences and explicit pre-export broker selection."""
import json
import re
from research_commands import revision, obj
KEY='source_preferences_v1'
DEFAULT={'mode':'auto','rules':[]}

def policy(value):
    if not isinstance(value,dict) or value.get('mode') not in ('auto','preferred','review'):raise ValueError('Choose a source selection mode.')
    rules=value.get('rules',[])
    if not isinstance(rules,list) or len(rules)>80:raise ValueError('Keep at most 80 source rules.')
    out=[];seen=set()
    for r in rules:
        if not isinstance(r,dict):raise ValueError('Invalid source rule.')
        name=r.get('name','');ticker=r.get('ticker','').strip().upper();task=r.get('task','')
        if not isinstance(name,str) or not 1<=len(name.strip())<=120:raise ValueError('Enter a broker or source name (up to 120 characters).')
        if ticker and not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker):raise ValueError('Invalid rule ticker.')
        if task not in ('','meeting','filing','earnings','event'):raise ValueError('Invalid task scope.')
        if r.get('disposition') not in ('preferred','standard','excluded','reference_only'):raise ValueError('Invalid source preference.')
        k=(name.strip().casefold(),ticker,task)
        if k in seen:raise ValueError('Keep one preference per source and scope.')
        seen.add(k);out.append(dict(name=name.strip(),ticker=ticker,task=task,disposition=r['disposition']))
    return {'mode':value['mode'],'rules':out}

def decision(p,publisher,ticker,task):
    p=policy(p);name=(publisher or '').strip().casefold()
    matches=[r for r in p['rules'] if r['name'].casefold()==name and (not r['ticker'] or r['ticker']==ticker) and (not r['task'] or r['task']==task)]
    if matches:
        score=max(bool(r['ticker'])+bool(r['task']) for r in matches)
        # Conflicting equally specific rules fail closed; explicit restrictions win.
        rank={'excluded':4,'reference_only':3,'preferred':2,'standard':1}
        disposition=max((r for r in matches if bool(r['ticker'])+bool(r['task'])==score),key=lambda r:rank[r['disposition']])['disposition']
    else:disposition='standard'
    if disposition in ('excluded','reference_only'):return disposition
    if p['mode']=='review':return 'review'
    if p['mode']=='preferred' and disposition!='preferred':return 'review'
    return 'include'

def instructions(p):
    return ('Source selection policy (frozen for this assignment): '+json.dumps(policy(p))+'. Match verified publisher names exactly; do not infer prestige from a brand or silently treat an alias as approved. Prioritize analyst expertise, original evidence, relevance, freshness and depth, not report counts. Primary disclosures anchor facts. Record relevant dissent and request review for a valuable non-preferred source. Excluded/reference-only preferences never enter AI synthesis. Use the source shortlist review API before exporting any source requiring review; user decisions never override provider GenAI restrictions.')

def create_routes(bp,get_db):
    from flask import request,jsonify
    @bp.route('/api/research/source-preferences',methods=['GET','PUT'])
    def preferences():
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(KEY,))
            cur.execute('SELECT value FROM app_settings WHERE key=%s',(KEY,));r=cur.fetchone();value=obj(r['value']) if r else DEFAULT
            if request.method=='PUT':
                data=request.get_json(silent=True) or {}
                if data.get('revision')!=revision(value):return jsonify(error='Preferences changed elsewhere. Reload before saving.'),409
                try:value=policy(data.get('policy'))
                except (ValueError,AttributeError,TypeError) as e:return jsonify(error=str(e)),400
                cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(KEY,json.dumps(value)))
        return jsonify(policy=value,revision=revision(value))

    @bp.route('/api/research/source-shortlists',methods=['GET'])
    def lists():
        with get_db() as (_,cur):
            cur.execute("SELECT value FROM app_settings WHERE key LIKE 'source_shortlist:%' ORDER BY updated_at DESC LIMIT 30")
            return jsonify(shortlists=[dict(obj(r['value']),revision=revision(obj(r['value']))) for r in cur.fetchall()])

    @bp.route('/api/research/commands/<cid>/source-shortlist',methods=['GET','POST','PUT'])
    def shortlist(cid):
        import uuid
        try:uuid.UUID(cid)
        except ValueError:return jsonify(error='Invalid command'),400
        key='source_shortlist:'+cid
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(key,))
            cur.execute("SELECT input FROM mp_jobs WHERE id=%s AND stage='collection_control'",(cid,));command=cur.fetchone()
            if not command:return jsonify(error='Command not found'),404
            p=obj(command['input'])['payload']
            cur.execute('SELECT value FROM app_settings WHERE key=%s',(key,));r=cur.fetchone()
            value=obj(r['value']) if r else {'commandId':cid,'ticker':p['ticker'],'candidates':[]}
            if request.method!='GET':
                data=request.get_json(silent=True) or {}
                if request.method=='POST':
                    rows=data.get('candidates')
                    if not isinstance(rows,list) or not 1<=len(rows)<=100:return jsonify(error='Submit 1–100 observed candidates.'),400
                    from urllib.parse import urlsplit
                    for item in rows:
                        if not isinstance(item,dict) or any(not isinstance(item.get(k),str) or not 1<=len(item[k])<=2000 for k in ('url','title','publisher','reason')):return jsonify(error='Candidate needs observed URL, title, publisher and selection reason.'),400
                        url=urlsplit(item['url'])
                        if url.scheme!='https' or url.hostname!='research.alpha-sense.com' or url.username or url.password:return jsonify(error='Use an observed AlphaSense document URL.'),400
                        if not any(x['url']==item['url'] for x in value['candidates']):
                            value['candidates'].append({k:item[k] for k in ('url','title','publisher','reason')}|{'decision':'pending'})
                    if len(value['candidates'])>200:return jsonify(error='Narrow this shortlist to 200 sources.'),400
                else:
                    if data.get('revision')!=revision(value):return jsonify(error='Shortlist changed. Reload before choosing.'),409
                    choices=data.get('choices')
                    if not isinstance(choices,dict) or any(v not in ('include','exclude') for v in choices.values()) or set(choices)-{x['url'] for x in value['candidates']}:return jsonify(error='Invalid source choices.'),400
                    for item in value['candidates']:
                        if item['url'] in choices:item['decision']=choices[item['url']]
                cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(key,json.dumps(value)))
        return jsonify(**value,revision=revision(value))
