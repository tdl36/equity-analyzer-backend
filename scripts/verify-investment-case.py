#!/usr/bin/env python3
"""Local disposable PostgreSQL verification; never uses production credentials."""
import sys, uuid
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask
from investment_case import create_blueprint

schema='charlie_case_qa_'+uuid.uuid4().hex
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
    app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client()
    url='/api/research/investment-case/ABBV'
    assert client.get(url).json['revision']==0
    first={'requestId':str(uuid.uuid4()),'revision':0,'body':{'thesis':'Frozen initial view'}}
    assert client.post(url,json=first).status_code==200
    assert client.post(url,json=first).json['replayed']
    def save(i):
        with app.test_client() as c:return c.post(url,json={'requestId':str(uuid.uuid4()),'revision':1,'body':{'thesis':f'Concurrent edit {i}'}}).status_code
    with ThreadPoolExecutor(2) as pool:assert sorted(pool.map(save,[1,2]))==[200,409]
    result=client.get(url).json
    assert result['revision']==2 and len(result['versions'])==2
    assert result['versions'][1]['body']['thesis']=='Frozen initial view'
    assert client.post('/api/research/investment-case/PFE',json=first).status_code==409
    # Restore and evidence operations use the same atomic revision/receipt contract.
    restore={'requestId':str(uuid.uuid4()),'revision':2,'mode':'restore','sourceRevision':1}
    assert client.post(url,json=restore).json['body']['thesis']=='Frozen initial view'
    assert client.post(url,json=restore).json['replayed']
    assert client.get(url).json['revision']==3
    assumption=str(uuid.uuid4());job_id=str(uuid.uuid4())
    with db(commit=True) as (_,cur):
        cur.execute("CREATE TABLE mp_jobs(id TEXT PRIMARY KEY,ticker TEXT,stage TEXT,status TEXT,result JSONB,input JSONB DEFAULT '{}'::jsonb,error TEXT,updated_at TIMESTAMPTZ DEFAULT NOW(),created_at TIMESTAMPTZ DEFAULT NOW())")
        cur.execute("CREATE TABLE document_files(ticker TEXT,filename TEXT,file_data TEXT,file_type TEXT)")
        import json
        result={'changes':[{'id':'0','after':'Management guided to steady growth, subject to demand.','reason':'New guidance','passageMatched':True,'reviewPassed':True,'evidence':[{'sourceId':'s','status':'passage_matched','excerpt':'We expect steady growth, subject to demand.'}]}],'sources':[{'id':'s','filename':'Original transcript','extractionHash':'immutable-hash'}]}
        cur.execute("INSERT INTO mp_jobs(id,ticker,stage,status,result) VALUES(%s,'ABBV','evidence_amendment','awaiting_approval',%s::jsonb)",(job_id,json.dumps(result)))
    assert len(client.get(url+'/source-proposals').json['proposals'])==1
    assert client.get('/api/research/investment-case/PFE/source-proposals').json['proposals']==[]
    assert client.post(url,json={'requestId':str(uuid.uuid4()),'revision':3,'body':{'assumptions':[{'id':assumption,'claim':'Working belief'}],'evidenceLinks':[{'forged':True}]}}).status_code==200
    assert client.get(url).json['body']['evidenceLinks']==[]
    selection={'jobId':job_id,'changeId':'0','assumptionId':assumption,'field':'support'}
    change={'requestId':str(uuid.uuid4()),'revision':4,'mode':'source_change','sourceChange':selection}
    response=client.post(url,json=change);assert response.status_code==200,response.json
    assert response.json['body']['evidenceLinks'][0]['evidence'][0]['source']['extractionHash']=='immutable-hash'
    assert client.post(url,json=change).json['replayed']
    assert client.get(url).json['revision']==5
    assert client.post(url,json={**change,'requestId':str(uuid.uuid4())}).status_code==409
    with db(commit=True) as (_,cur):cur.execute("UPDATE mp_jobs SET status='dismissed' WHERE id=%s",(job_id,))
    assert client.post(url,json={**change,'requestId':str(uuid.uuid4()),'revision':5,'sourceChange':{**selection,'field':'contrary'}}).status_code==409
    assert client.post(url,json={'requestId':str(uuid.uuid4()),'revision':5,'mode':'restore','sourceRevision':4}).json['revision']==6
    assert not client.get(url).json['body']['evidenceLinks']
    assert len(client.get(url).json['versions'])==6
    print('PASS: real SQL schema, immutable history, receipt replay, cross-ticker request conflict, concurrent stale-edit rejection. No production or provider calls.')
finally:
    with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    admin.close()
