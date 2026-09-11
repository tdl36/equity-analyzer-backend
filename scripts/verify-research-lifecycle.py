#!/usr/bin/env python3
"""Recorded-time isolation in a disposable local PostgreSQL schema only."""
import sys, uuid, json
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask
from research_lifecycle import create_blueprint
schema='charlie_lifecycle_qa_'+uuid.uuid4().hex
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
        for table in ['investment_case_versions','research_work_versions','research_decisions']:
            c.execute(f'CREATE TABLE {table}(ticker TEXT,id TEXT,revision INTEGER,body JSONB,created_at TIMESTAMPTZ)')
            for ticker,revision,date in [('ABT',1,'2020-01-01'),('ABT',2,'2020-03-01'),('MDT',1,'2020-01-01')]:
                c.execute(f'INSERT INTO {table} VALUES(%s,%s,%s,%s,%s)',(ticker,ticker,revision,json.dumps({'thesis':ticker+str(revision)}),date))
    app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client()
    r=client.get('/api/research/lifecycle/ABT?asOf=2020-02-01T00:00:00Z')
    assert r.status_code==200,r.json
    for key in ['cases','work','decisions']:
        assert len(r.json[key])==1 and r.json[key][0]['body']['thesis']=='ABT1'
    assert client.get('/api/research/lifecycle/ABT').json['cases'][0]['revision']==2
    assert client.get('/api/research/lifecycle/ABT?asOf=2099-01-01T00:00:00Z').status_code==400
    assert client.get('/api/research/lifecycle/ABT?asOf=2019-01-01T00:00:00Z').json['cases']==[]
    with db(True) as (_,c):
        for i in range(3,105):c.execute('INSERT INTO investment_case_versions VALUES(%s,%s,%s,%s,%s)',('ABT','ABT',i,'{}','2020-04-01'))
    r=client.get('/api/research/lifecycle/ABT')
    assert r.json['limited']['cases'] and len(r.json['cases'])==100
    assert r.headers['Cache-Control']=='no-store'
    print('PASS: recorded-time cutoff, ticker isolation, empty history, future rejection and explicit history bounds.')
finally:
    with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    admin.close()
