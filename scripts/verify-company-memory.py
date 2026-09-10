#!/usr/bin/env python3
"""Read-only memory verification against an isolated local PostgreSQL schema."""
import sys, uuid, json
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask
import company_memory as cm
schema='charlie_memory_qa_'+uuid.uuid4().hex
admin=psycopg2.connect(dbname='postgres',host='/tmp');admin.autocommit=True
@contextmanager
def db(commit=False):
    c=psycopg2.connect(dbname='postgres',host='/tmp')
    try:
        cur=c.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        yield c,cur
        if commit:c.commit()
    finally:c.close()
try:
    with admin.cursor() as c:c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    with db(True) as (_,c):
        c.execute('CREATE TABLE portfolio_analyses(ticker TEXT,analysis JSONB,updated_at TIMESTAMPTZ DEFAULT NOW())')
        c.execute("INSERT INTO portfolio_analyses(ticker,analysis) VALUES ('ABT',%s),('MDT',%s)",(json.dumps({'thesis':{'summary':'ABT legacy'}}),json.dumps({'thesis':{'summary':'MDT private'}})))
    assert cm.load(db,'ABT')['entries'][0]['kind']=='legacy_thesis'
    with db(True) as (_,c):
        c.execute('CREATE TABLE investment_case_versions(ticker TEXT,revision INTEGER,body JSONB,created_at TIMESTAMPTZ DEFAULT NOW())')
        c.execute("INSERT INTO investment_case_versions(ticker,revision,body) VALUES('ABT',1,%s),('ABT',2,%s)",(json.dumps({'thesis':'superseded'}),json.dumps({'thesis':'current belief','evidenceLinks':[{'publisher':'Wells Fargo'}]})))
    app=Flask(__name__);app.register_blueprint(cm.create_blueprint(db))
    r=app.test_client().get('/api/research/company-memory/ABT')
    assert r.status_code==200 and r.headers['Cache-Control']=='no-store'
    assert r.json['entries'][0]['revision']==2
    assert 'MDT private' not in r.get_data(as_text=True)
    assert 'superseded' not in r.get_data(as_text=True)
    assert r.json==app.test_client().get('/api/research/company-memory/ABT').json
    assert cm.load(db,'UNKNOWN')['entries']==[]
    import meeting_memory
    with db(True) as (_,c):
        c.execute('CREATE TABLE mp_jobs(id TEXT PRIMARY KEY,ticker TEXT,input JSONB,status TEXT,updated_at TIMESTAMPTZ DEFAULT NOW())')
        c.execute("INSERT INTO mp_jobs VALUES('meeting','ABT',%s,'running',NOW())",(json.dumps({'companyMemoryRequested':True,'workerToken':'qa'}),))
    frozen=meeting_memory.freeze(db,'meeting','ABT','qa')
    assert frozen['entries'][0]['revision']==2
    with db(True) as (_,c):
        c.execute("INSERT INTO investment_case_versions(ticker,revision,body) VALUES('ABT',3,%s)",(json.dumps({'thesis':'new belief after dispatch'}),))
    assert meeting_memory.freeze(db,'meeting','ABT','qa')==frozen
    assert cm.load(db,'ABT')['entries'][0]['revision']==3
    try: meeting_memory.freeze(db,'meeting','ABT','wrong-owner')
    except ValueError: pass
    else: raise AssertionError('Wrong owner accepted')
    print('PASS: meeting snapshot persists across later revisions and rejects wrong worker.')
    print('PASS: latest revision, provenance, ticker isolation, stable snapshots, missing case table and no-store route.')
finally:
    with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    admin.close()
