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
    import research_decisions
    journal=Flask('journal');journal.register_blueprint(research_decisions.create_blueprint(db))
    client=journal.test_client();url='/api/research/decisions/ABT'
    assert client.get(url).json['revision']==0
    payload={'requestId':str(uuid.uuid4()),'revision':0,'decisionDate':'2026-01-01',
        'decision':'Keep researching','rationale':'Cash quality unresolved','revisitWhen':'Two quarters of conversion above hurdle'}
    assert client.post(url,json=payload).status_code==201
    assert client.post(url,json=payload).json['replayed']
    assert client.post('/api/research/decisions/MDT',json=payload).status_code==409
    assert client.post(url,json={**payload,'requestId':str(uuid.uuid4())}).status_code==409
    replacement={**payload,'requestId':str(uuid.uuid4()),'revision':1,'supersedes':payload['requestId'],'decision':'Reopen the debate'}
    assert client.post(url,json=replacement).status_code==201
    records=client.get(url).json['decisions']
    assert records[0]['superseded'] is False and records[1]['superseded'] is True
    assert records[1]['body']['decision']=='Keep researching'
    assert client.post(url,json={**replacement,'requestId':str(uuid.uuid4()),'revision':2}).status_code==409
    memory=cm.load(db,'ABT');decisions=[e for e in memory['entries'] if e['kind']=='analyst_decision']
    assert decisions[0]['body']['decision']=='Reopen the debate' and decisions[1]['status']=='superseded'
    assert not [e for e in cm.load(db,'MDT')['entries'] if e['kind']=='analyst_decision']
    assert meeting_memory.freeze(db,'meeting','ABT','qa')==frozen
    with db(True) as (_,c):
        for revision in range(3,65):
            body={'decision':'Literal 10% hurdle' if revision==3 else 'Historical review',
                  'rationale':'Cash conversion','revisitWhen':'New filing','decisionDate':'2026-02-01','supersedes':None}
            c.execute("INSERT INTO research_decisions(id,ticker,revision,fingerprint,body) VALUES(%s,'ABT',%s,'fixture',%s)",(str(uuid.uuid4()),revision,json.dumps(body)))
    first=client.get(url).json
    assert len(first['decisions'])==50 and first['hasMore'] and first['revision']==64
    second=client.get(url+'?before='+str(first['nextBefore'])).json
    assert len(second['decisions'])==14 and not second['hasMore'] and second['revision']==64
    assert not set(r['id'] for r in first['decisions']) & set(r['id'] for r in second['decisions'])
    match=client.get(url,query_string={'q':'10%'}).json
    assert len(match['decisions'])==1 and match['decisions'][0]['revision']==3 and match['revision']==64
    assert len(client.get(url,query_string={'to':'2026-01-31'}).json['decisions'])==2
    assert client.get(url,query_string={'q':'missing'}).json['revision']==64
    assert client.get(url,query_string={'id':payload['requestId']}).json['decisions'][0]['superseded'] is True
    assert client.get('/api/research/decisions/MDT',query_string={'id':payload['requestId']}).json['decisions']==[]
    assert client.get(url,query_string={'from':'bad'}).status_code==400
    assert 'older history is omitted' in cm.render(cm.load(db,'ABT'))
    import investor_framework
    framework_app=Flask('framework');framework_app.register_blueprint(investor_framework.create_blueprint(db))
    fc=framework_app.test_client();fu='/api/research/investor-framework'
    assert fc.get(fu).json['revision']==0
    fbody={'philosophy':'Test normalized cash earnings','evidenceDiscipline':'Retain management qualifiers'}
    fsave={'requestId':str(uuid.uuid4()),'revision':0,'body':fbody}
    assert fc.post(fu,json=fsave).status_code==201
    assert fc.post(fu,json=fsave).json['replayed']
    assert fc.post(fu,json={**fsave,'body':{'philosophy':'Changed request'}}).status_code==409
    assert fc.post(fu,json={**fsave,'requestId':str(uuid.uuid4())}).status_code==409
    context=cm.load(db,'ABT')
    assert context['framework']['revision']==1
    assert 'Test normalized cash earnings' in cm.render(context)
    assert 'cannot override original-source verification' in cm.render(context)
    assert cm.load(db,'MDT')['framework']==context['framework']
    with db(True) as (_,c):
        c.execute("INSERT INTO mp_jobs VALUES('framework-meeting','ABT',%s,'running',NOW())",(json.dumps({'companyMemoryRequested':True,'workerToken':'qa'}),))
    frozen_framework=meeting_memory.freeze(db,'framework-meeting','ABT','qa')
    assert frozen_framework['framework']['revision']==1
    assert fc.post(fu,json={'requestId':str(uuid.uuid4()),'revision':1,'body':{}}).status_code==201
    assert not any(cm.load(db,'ABT')['framework']['body'].values())
    restore={'requestId':str(uuid.uuid4()),'revision':2,'sourceRevision':1}
    assert fc.post(fu,json=restore).status_code==201
    assert fc.post(fu,json=restore).json['replayed']
    assert len(fc.get(fu).json['versions'])==3
    assert cm.load(db,'ABT')['framework']['revision']==3
    assert cm.load(db,'ABT')['framework']['body']['philosophy']==fbody['philosophy']
    assert meeting_memory.freeze(db,'framework-meeting','ABT','qa')==frozen_framework
    assert meeting_memory.freeze(db,'meeting','ABT','qa')==frozen
    print('PASS: framework save/replay/conflict, cross-company context, clearing, non-destructive restoration and frozen meeting version.')
    print('PASS: 64-record pagination, no overlap, literal search, dates, independent write revision and isolated prior-record lookup.')
    print('PASS: immutable decisions, request replay, conflicts, supersession and memory inclusion; old job stays frozen.')
    print('PASS: meeting snapshot persists across later revisions and rejects wrong worker.')
    print('PASS: latest revision, provenance, ticker isolation, stable snapshots, missing case table and no-store route.')
finally:
    with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    admin.close()
