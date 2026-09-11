#!/usr/bin/env python3
"""Disposable local SQL + deterministic provider fixture; no production credentials."""
import sys,uuid,json,base64
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask
import investment_case,research_amendments
schema='charlie_proposal_qa_'+uuid.uuid4().hex
admin=psycopg2.connect(dbname='postgres',host='/tmp');admin.autocommit=True
@contextmanager
def db(commit=False):
 c=psycopg2.connect(dbname='postgres',host='/tmp')
 try:
  cur=c.cursor(cursor_factory=RealDictCursor);cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
  yield c,cur
  if commit:c.commit()
 finally:c.close()
class InlineThread:
 def __init__(self,target,args,**kwargs):self.target=target;self.args=args
 def start(self):self.target(*self.args)
aid=str(uuid.uuid4());excerpt='Management expects margin recovery subject to underlying demand.'
calls=[];fail_review=False
def model(prompt,key,tokens):
 global fail_review
 calls.append(tokens)
 if tokens==5000:
  if fail_review:raise ValueError('Synthetic reviewer interruption')
  return {'checks':[{'id':'0','verdict':'pass','issue':''},{'id':'condition-0','verdict':'pass','issue':''}]}
 # Read source catalog identity from the actual assembled prompt.
 sources=json.loads(prompt.split('SOURCE DOCUMENTS:\n')[-1])
 assert 'CASE CONTEXT' in prompt
 return {'changes':[{'path':f'assumptions.{aid}.support','after':excerpt,'reason':'Updated guidance','source_id':sources[0]['id'],'source_excerpt':excerpt}], 'condition_assessments':[{'work_id':'w','condition_id':'c','assessment':'partly_met','reason':'Improvement is reported, but sustainability remains unresolved.','source_id':sources[0]['id'],'source_excerpt':excerpt},{'work_id':'assumptions.not-a-work-record','condition_id':'A textual signpost','assessment':'met','reason':'Invalid optional model output'}]}
try:
 with admin.cursor() as c:c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
 with db(commit=True) as (_,c):
  c.execute("CREATE TABLE mp_jobs(id TEXT PRIMARY KEY,stage TEXT,ticker TEXT,status TEXT,input JSONB,result JSONB,error TEXT,created_at TIMESTAMPTZ DEFAULT NOW(),updated_at TIMESTAMPTZ DEFAULT NOW())")
  c.execute("CREATE TABLE research_work_versions(id TEXT,ticker TEXT,revision INTEGER,body JSONB)")
  c.execute("INSERT INTO research_work_versions VALUES('w','ABBV',1,%s::jsonb)",(json.dumps({'kind':'underweight','status':'open','title':'Recovery review','reviewConditions':[{'id':'c','trigger':'Margin improvement'}]}),))
  c.execute("CREATE TABLE document_files(ticker TEXT,filename TEXT,file_data TEXT,file_type TEXT)")
  c.execute("INSERT INTO document_files VALUES('ABBV','Transcript.txt',%s,'text/plain')",(base64.b64encode(excerpt.encode()).decode(),))
 app=Flask(__name__);app.register_blueprint(investment_case.create_blueprint(db));app.register_blueprint(research_amendments.create_blueprint(db,model,lambda _: 'fixture-key'))
 client=app.test_client();url='/api/research/investment-case/ABBV'
 assert client.get(url).status_code==200
 assert client.post(url,json={'requestId':str(uuid.uuid4()),'revision':0,'body':{'assumptions':[{'id':aid,'claim':'Margins recover'}]}}).status_code==200
 req={'requestId':str(uuid.uuid4()),'revision':1,'filenames':['Transcript.txt'],'instructions':'Check margins'}
 with patch('research_amendments.threading.Thread',InlineThread),patch('research_amendments.notegen.extract_file_text',lambda *a,**k:excerpt):
  assert client.post(url+'/proposals',json={**req,'expectedStoredHashes':{'Transcript.txt':'wrong'}}).status_code==409
  fail_review=True
  assert client.post(url+'/proposals',json=req).status_code==202
  jobs=client.get(url+'/source-proposals').json['proposals'];assert jobs[0]['status']=='failed'
  assert calls==[12000,5000]
  assert client.post(url+'/proposals',json=req).status_code==200
  fail_review=False
  r=client.post('/api/research/amendment/'+req['requestId']+'/resume',json={});assert r.status_code==202,r.json
  assert calls==[12000,5000,5000],calls
  job=client.get(url+'/source-proposals').json['proposals'][0]
  assert job['status']=='awaiting_approval' and job['result']['changes'][0]['assumptionId']==aid
  assert len(job['result']['conditionWarnings'])==1
  assert job['result']['conditionAssessments'][0]['reviewPassed']
  assert job['result']['conditionAssessments'][0]['workRevision']==1
  assert client.get(url).json['revision']==1
  assert client.get('/api/research/amendments/ABBV').json['jobs']==[]
  assert client.post('/api/research/amendment/'+req['requestId']+'/decide',json={'action':'apply','acceptedIds':['0']}).status_code==409
  accept={'requestId':str(uuid.uuid4()),'revision':1,'mode':'source_change','sourceChange':{'jobId':req['requestId'],'changeId':'0','assumptionId':aid,'field':'support'}}
  r=client.post(url,json=accept);assert r.status_code==200,r.json
  assert r.json['body']['assumptions'][0]['support']==excerpt
  assert client.post(url,json=accept).json['replayed']
  assert len(calls)==3
  decision_url='/api/research/amendment/'+req['requestId']+'/decide'
  assert client.post(decision_url,json={'action':'dismiss','reviewDecision':{'outcome':'no_change','rationale':''}}).status_code==400
  assert client.post(decision_url,json={'action':'dismiss','reviewDecision':{'outcome':'changes_reviewed','rationale':'Accepted the source-supported change after reviewing management qualifiers.'}}).status_code==200
  first=client.get(url+'/source-proposals').json['proposals'][0]['result']['reviewDecision']
  assert client.post(decision_url,json={'action':'dismiss','reviewDecision':{'outcome':'rejected','rationale':'Different later wording'}}).status_code==200
  assert client.get(url+'/source-proposals').json['proposals'][0]['result']['reviewDecision']==first
 print('PASS: queued generation, source provenance, review failure + checkpoint-only resume, no auto-apply, legacy isolation and idempotent acceptance. Local synthetic data only.')
finally:
 with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
 admin.close()
