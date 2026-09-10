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
    print('PASS: real SQL schema, immutable history, receipt replay, cross-ticker request conflict, concurrent stale-edit rejection. No production or provider calls.')
finally:
    with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    admin.close()
