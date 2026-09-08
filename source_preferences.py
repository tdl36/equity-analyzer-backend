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
        subsector=r.get('subsector','').strip();analyst=r.get('analyst','').strip()
        if len(subsector)>100 or len(analyst)>120:raise ValueError('Subsector or analyst name is too long.')
        if ticker and subsector:raise ValueError('Choose a stock or subsector scope, not both.')
        k=(name.strip().casefold(),ticker,task,subsector.casefold(),analyst.casefold())
        if k in seen:raise ValueError('Keep one preference per source and scope.')
        seen.add(k);out.append(dict(name=name.strip(),ticker=ticker,task=task,disposition=r['disposition'],**({'subsector':subsector} if subsector else {}),**({'analyst':analyst} if analyst else {})))
    coverage=value.get('subsectors',{})
    if not isinstance(coverage,dict) or len(coverage)>500:raise ValueError('Keep at most 500 company classifications.')
    normalized={}
    for tk,sector in coverage.items():
        if not isinstance(tk,str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',tk.strip().upper()) or not isinstance(sector,str) or not 1<=len(sector.strip())<=100:raise ValueError('Each classification needs a ticker and subsector.')
        if tk.strip().upper() in normalized:raise ValueError('Duplicate company classification.')
        normalized[tk.strip().upper()]=sector.strip()
    return {'mode':value['mode'],'rules':out,**({'subsectors':normalized} if normalized else {})}

def resolve(p,publisher,ticker,task,analyst=None,author_evidence=None):
    p=policy(p);name=(publisher or '').strip().casefold();ticker=ticker.strip().upper()
    sector=p.get('subsectors',{}).get(ticker,'')
    eligible=[r for r in p['rules'] if r['name'].casefold()==name and (not r['ticker'] or r['ticker']==ticker) and (not r.get('subsector') or r['subsector'].casefold()==sector.casefold()) and (not r['task'] or r['task']==task)]
    verified=bool(analyst and author_evidence and str(author_evidence).strip())
    matches=[r for r in eligible if not r.get('analyst') or (verified and r['analyst'].casefold()==analyst.strip().casefold())]
    score=lambda r:(4 if r['ticker'] else 2 if r.get('subsector') else 0, bool(r.get('analyst')),bool(r['task']))
    rank={'excluded':4,'reference_only':3,'preferred':2,'standard':1}
    rule=max(matches,key=lambda r:(score(r),rank[r['disposition']])) if matches else None
    disposition=rule['disposition'] if rule else 'standard'
    uncertain=not verified and any(r.get('analyst') and (not rule or score(r)>=score(rule)) for r in eligible)
    if disposition in ('excluded','reference_only'):outcome=disposition
    elif uncertain or p['mode']=='review' or (p['mode']=='preferred' and disposition!='preferred'):outcome='review'
    else:outcome='include'
    return dict(decision=outcome,disposition=disposition,rule=rule,subsector=sector,authorVerified=verified,
                reason='Verify the report author or request document-level review.' if uncertain else 'Matched explicit source preference.' if rule else 'No matching rule; using assignment selection mode.')


def decision(p,publisher,ticker,task,analyst=None,author_evidence=None):
    return resolve(p,publisher,ticker,task,analyst,author_evidence)['decision']


def instructions(p):
    return ('Source selection policy (frozen for this assignment): '+json.dumps(policy(p))+'. Match verified publisher names exactly; do not infer prestige from a brand or silently treat an alias as approved. Prioritize analyst expertise, original evidence, relevance, freshness and depth, not report counts. Primary disclosures anchor facts. Record relevant dissent and request review for a valuable non-preferred source. Excluded/reference-only preferences never enter AI synthesis. Use the source shortlist review API before exporting any source requiring review; user decisions never override provider GenAI restrictions.')

def create_routes(bp,get_db):
    from flask import request,jsonify
    @bp.route('/api/research/source-preferences/resolve',methods=['POST'])
    def preview():
        try:
            d=request.get_json() or {}
            return jsonify(resolve(d['policy'],d.get('publisher',''),d.get('ticker',''),d.get('task','event'),d.get('analyst'),d.get('authorEvidence')))
        except (ValueError,TypeError,AttributeError,KeyError) as e:return jsonify(error=str(e)),400

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
                        if any(k in item and (not isinstance(item[k],str) or len(item[k])>2000) for k in ('analyst','authorEvidence')):return jsonify(error='Invalid authorship metadata.'),400
                        url=urlsplit(item['url'])
                        if url.scheme!='https' or url.hostname!='research.alpha-sense.com' or url.username or url.password:return jsonify(error='Use an observed AlphaSense document URL.'),400
                        if not any(x['url']==item['url'] for x in value['candidates']):
                            value['candidates'].append({k:item[k] for k in ('url','title','publisher','reason','analyst','authorEvidence') if k in item}|{'decision':'pending'})
                    if len(value['candidates'])>200:return jsonify(error='Narrow this shortlist to 200 sources.'),400
                else:
                    if data.get('revision')!=revision(value):return jsonify(error='Shortlist changed. Reload before choosing.'),409
                    choices=data.get('choices')
                    if not isinstance(choices,dict) or any(v not in ('include','exclude') for v in choices.values()) or set(choices)-{x['url'] for x in value['candidates']}:return jsonify(error='Invalid source choices.'),400
                    for item in value['candidates']:
                        if item['url'] in choices:item['decision']=choices[item['url']]
                cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(key,json.dumps(value)))
        return jsonify(**value,revision=revision(value))
