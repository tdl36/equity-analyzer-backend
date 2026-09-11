"""Append-only analyst decision journal; never executes portfolio actions."""
import hashlib
import json
import re
import uuid
from datetime import date
from flask import Blueprint, jsonify, request


def validate(data):
    if not isinstance(data,dict):raise ValueError('A decision is required.')
    try: rid=str(uuid.UUID(data.get('requestId','')))
    except (ValueError,TypeError,AttributeError):raise ValueError('A valid request ID is required.')
    revision=data.get('revision')
    if type(revision) is not int or revision<0:raise ValueError('Reload the decision log before saving.')
    body={}
    for key,cap in [('decision',2000),('rationale',12000),('revisitWhen',6000)]:
        value=data.get(key)
        if not isinstance(value,str) or not value.strip() or len(value)>cap:raise ValueError(f'Enter {key} (up to {cap:,} characters).')
        body[key]=value.strip()
    try: when=date.fromisoformat(data.get('decisionDate',''))
    except (ValueError,TypeError):raise ValueError('Enter the date of the decision.')
    if when>date.today():raise ValueError('A recorded decision cannot be dated in the future.')
    body['decisionDate']=when.isoformat()
    prior=data.get('supersedes') or None
    if prior:
        try: prior=str(uuid.UUID(prior))
        except (ValueError,TypeError,AttributeError):raise ValueError('Choose a valid prior decision.')
    body['supersedes']=prior
    issue=data.get('issue','')
    if not isinstance(issue,str) or len(issue)>200:raise ValueError('Issue must be at most 200 characters.')
    if issue.strip():
        disposition=data.get('disposition','unresolved')
        if disposition not in ('unchanged','review_needed','accepted_change','unresolved'):
            raise ValueError('Choose a valid issue disposition.')
        body['issue']=issue.strip()
        body['issueId']=rid  # Server identity; a superseding review inherits below.
        body['disposition']=disposition
        due=data.get('reviewDate','')
        if due:
            try:body['reviewDate']=date.fromisoformat(due).isoformat()
            except (ValueError,TypeError):raise ValueError('Choose a valid review date.')
    elif data.get('reviewDate'):
        raise ValueError('Name the issue before setting a review date.')
    return rid,revision,body


def query_options(args):
    result={}
    query=args.get('q','').strip()
    if len(query)>200:raise ValueError('Search text must be 200 characters or fewer.')
    result['q']=query
    for key in ('from','to'):
        value=args.get(key,'')
        if value:
            try:result[key]=date.fromisoformat(value).isoformat()
            except (ValueError,TypeError):raise ValueError('Use valid decision dates.')
    if result.get('from','')>result.get('to','9999-12-31'):raise ValueError('Start date must not follow end date.')
    if args.get('before'):
        try:result['before']=int(args['before'])
        except ValueError:raise ValueError('Invalid history cursor.')
        if not 0<result['before']<=2147483647:raise ValueError('Invalid history cursor.')
    if args.get('id'):
        try:result['id']=str(uuid.UUID(args['id']))
        except ValueError:raise ValueError('Invalid decision identifier.')
    return result


def read(cur,ticker,limit=50,options=None):
    options=options or {}
    clauses=['d.ticker=%s'];params=[ticker]
    if options.get('q'):
        clauses.append("strpos(lower(concat_ws(' ',d.body->>'decision',d.body->>'rationale',d.body->>'revisitWhen',d.body->>'issue')),lower(%s))>0")
        params.append(options['q'])
    for key,op in [('from','>='),('to','<=')]:
        if options.get(key):
            clauses.append("d.body->>'decisionDate' "+op+" %s");params.append(options[key])
    if options.get('before'):
        clauses.append('d.revision<%s');params.append(options['before'])
    if options.get('id'):
        clauses.append('d.id=%s');params.append(options['id'])
    cur.execute('''SELECT d.id,d.revision,d.body,d.created_at,
        EXISTS(SELECT 1 FROM research_decisions n WHERE n.ticker=d.ticker AND n.body->>'supersedes'=d.id) AS superseded
        FROM research_decisions d WHERE '''+' AND '.join(clauses)+' ORDER BY d.revision DESC LIMIT %s',tuple(params+[limit+1]))
    rows=[dict(r) for r in cur.fetchall()]
    return rows[:limit],len(rows)>limit


def create_blueprint(get_db):
    bp=Blueprint('research_decisions',__name__)
    def ensure():
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('research-decisions-schema',))
            cur.execute('''CREATE TABLE IF NOT EXISTS research_decisions(
                id TEXT PRIMARY KEY,ticker TEXT NOT NULL,revision INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,body JSONB NOT NULL,created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(ticker,revision))''')
    @bp.route('/api/research/decisions/<ticker>',methods=['GET','POST'])
    def decisions(ticker):
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker):return jsonify(error='Choose a valid ticker.'),400
        if request.method=='POST':
            try:rid,revision,body=validate(request.get_json(silent=True))
            except ValueError as exc:return jsonify(error=str(exc)),400
        if request.method=='GET':
            try: options=query_options(request.args)
            except ValueError as exc:return jsonify(error=str(exc)),400
        ensure()
        if request.method=='GET':
            with get_db() as (_,cur):
                rows,more=read(cur,ticker,options=options)
                cur.execute('SELECT COALESCE(MAX(revision),0) AS revision FROM research_decisions WHERE ticker=%s',(ticker,))
                revision=cur.fetchone()['revision']
            response=jsonify(decisions=rows,hasMore=more,revision=revision,nextBefore=rows[-1]['revision'] if more else None)
            response.headers['Cache-Control']='no-store';return response
        fp=hashlib.sha256(json.dumps([ticker,revision,body],sort_keys=True).encode()).hexdigest()
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('decision-request:'+rid,))
            cur.execute('SELECT fingerprint,revision FROM research_decisions WHERE id=%s',(rid,));old=cur.fetchone()
            if old:
                if old['fingerprint']!=fp:return jsonify(error='Request ID belongs to another decision.'),409
                return jsonify(id=rid,revision=old['revision'],replayed=True)
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('decision-ticker:'+ticker,))
            cur.execute('SELECT COALESCE(MAX(revision),0) AS revision FROM research_decisions WHERE ticker=%s',(ticker,));current=cur.fetchone()['revision']
            if revision!=current:return jsonify(error='Decision log changed. Reload before saving; your draft is retained.'),409
            if body['supersedes']:
                cur.execute("SELECT id,body FROM research_decisions WHERE ticker=%s AND id=%s AND NOT EXISTS(SELECT 1 FROM research_decisions n WHERE n.ticker=%s AND n.body->>'supersedes'=%s)",(ticker,body['supersedes'],ticker,body['supersedes']))
                previous=cur.fetchone()
                if not previous:return jsonify(error='The prior decision is missing or already superseded. Reload the log.'),409
                prior_body=previous['body']
                if isinstance(prior_body,str):prior_body=json.loads(prior_body)
                if prior_body.get('issueId'):
                    if not body.get('issue'):return jsonify(error='Keep the issue name when superseding an issue review.'),400
                    body['issueId']=prior_body['issueId']
            cur.execute('INSERT INTO research_decisions(id,ticker,revision,fingerprint,body) VALUES(%s,%s,%s,%s,%s::jsonb)',(rid,ticker,current+1,fp,json.dumps(body)))
        return jsonify(id=rid,revision=current+1),201
    return bp
