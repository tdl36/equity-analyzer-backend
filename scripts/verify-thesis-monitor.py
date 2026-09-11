#!/usr/bin/env python3
"""Disposable local database; fake proposal submission, no model or production writes."""
import sys,uuid,json
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask,Blueprint,jsonify
import thesis_monitor
schema='charlie_monitor_qa_'+uuid.uuid4().hex
admin=psycopg2.connect(dbname='postgres',host='/tmp');admin.autocommit=True
@contextmanager
def db(commit=False):
 c=psycopg2.connect(dbname='postgres',host='/tmp')
 try:
  cur=c.cursor(cursor_factory=RealDictCursor);cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
  yield c,cur
  if commit:c.commit()
 finally:c.close()
calls=[];fail=[False]
def submit(ticker,payload,target):
 calls.append(payload['requestId'])
 if fail[0]:fail[0]=False;raise ValueError('Simulated lost receipt')
 return jsonify(jobId=payload['requestId']),202
try:
 with admin.cursor() as c:c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
 with db(True) as (_,c):
  c.execute('CREATE TABLE document_files(ticker TEXT,filename TEXT,file_data TEXT)')
  c.execute('CREATE TABLE investment_case_versions(ticker TEXT,revision INTEGER,body JSONB)')
  c.execute('CREATE TABLE mp_jobs(id TEXT)')
  c.execute("INSERT INTO investment_case_versions VALUES('ABT',1,'{\"assumptions\":[{\"id\":\"a\"}]}')")
  c.execute("INSERT INTO document_files VALUES('ABT','old.pdf','old')")
 app=Flask(__name__);bp=Blueprint('test_monitor',__name__);thesis_monitor.register(bp,db,submit);app.register_blueprint(bp);client=app.test_client()
 url='/api/research/investment-case/ABT/monitor';tick='/api/agent/advance-thesis-monitors'
 assert client.get(url).json['revision']==0
 assert client.post(url,json={'enabled':True,'revision':0}).status_code==200
 assert client.post(tick).json['outcomes'][0]['state']=='watching' and not calls
 with db(True) as (_,c):c.execute("INSERT INTO document_files VALUES('ABT','new.pdf','new')")
 fail[0]=True
 assert client.post(tick).json['outcomes'][0]['state']=='blocked'
 assert client.get(url).json['pending'] is not None
 assert client.post(tick).json['outcomes'][0]['state']=='submitted'
 assert calls[0]==calls[1]
 assert client.post(tick).json['outcomes'][0]['state']=='watching' and len(calls)==2
 with db(True) as (_,c):c.execute("INSERT INTO document_files VALUES('ABT','renamed.pdf','new')")
 assert client.post(tick).json['outcomes'][0]['state']=='watching'
 assert client.post(url,json={'enabled':False,'revision':0}).status_code==409
 assert client.post(url,json={'enabled':False,'revision':1}).status_code==200
 with db(True) as (_,c):c.execute("UPDATE document_files SET file_data='changed' WHERE filename='new.pdf'")
 assert client.post(tick).json['outcomes']==[]
 assert client.post(url,json={'enabled':True,'revision':2}).status_code==200
 assert client.post(tick).json['outcomes'][0]['state']=='submitted'
 assert len(calls)==3
 print('PASS: prospective enrollment, content-change detection, renamed-content dedup, crash receipt replay, pause/resume, stale settings.')
finally:
 with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
 admin.close()
