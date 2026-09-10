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
    return rid,revision,body


def read(cur,ticker,limit=50):
    cur.execute('''SELECT d.id,d.revision,d.body,d.created_at,
        EXISTS(SELECT 1 FROM research_decisions n WHERE n.ticker=d.ticker AND n.body->>'supersedes'=d.id) AS superseded
        FROM research_decisions d WHERE ticker=%s ORDER BY revision DESC LIMIT %s''',(ticker,limit+1))
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
        ensure()
        if request.method=='GET':
            with get_db() as (_,cur): rows,more=read(cur,ticker)
            response=jsonify(decisions=rows,hasMore=more,revision=rows[0]['revision'] if rows else 0)
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
                cur.execute("SELECT id FROM research_decisions WHERE ticker=%s AND id=%s AND NOT EXISTS(SELECT 1 FROM research_decisions n WHERE n.ticker=%s AND n.body->>'supersedes'=%s)",(ticker,body['supersedes'],ticker,body['supersedes']))
                if not cur.fetchone():return jsonify(error='The prior decision is missing or already superseded. Reload the log.'),409
            cur.execute('INSERT INTO research_decisions(id,ticker,revision,fingerprint,body) VALUES(%s,%s,%s,%s,%s::jsonb)',(rid,ticker,current+1,fp,json.dumps(body)))
        return jsonify(id=rid,revision=current+1),201
    return bp
