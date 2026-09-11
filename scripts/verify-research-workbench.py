#!/usr/bin/env python3
"""Disposable local PostgreSQL acceptance; never touches production research."""
import sys,json,uuid,hashlib
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask
import research_workbench as w,company_memory as cm
schema='charlie_work_qa_'+uuid.uuid4().hex
admin=psycopg2.connect(dbname='postgres',host='/tmp');admin.autocommit=True
@contextmanager
def db(commit=False):
 c=psycopg2.connect(dbname='postgres',host='/tmp')
 try:
  cur=c.cursor(cursor_factory=RealDictCursor);cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
  yield c,cur
  if commit:c.commit()
 finally:c.close()
try:
 with admin.cursor() as c:c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
 aid=str(uuid.uuid4());text='Reported operating margin was 10 percent; durability is uncertain.'
 with db(True) as (_,c):
  c.execute('CREATE TABLE investment_case_versions(ticker TEXT,revision INTEGER,body JSONB,created_at TIMESTAMPTZ DEFAULT NOW())')
  c.execute('INSERT INTO investment_case_versions VALUES(%s,1,%s,NOW())',('ABT',json.dumps({'assumptions':[{'id':aid,'claim':'Margin recovery'}]})))
  c.execute('CREATE TABLE portfolio_analyses(ticker TEXT,analysis JSONB,updated_at TIMESTAMPTZ)')
  c.execute('CREATE TABLE mp_companies(id INTEGER,ticker TEXT)');c.execute("INSERT INTO mp_companies VALUES(1,'ABT'),(2,'MDT')")
  c.execute('CREATE TABLE mp_meetings(id INTEGER,company_id INTEGER,meeting_date DATE)');c.execute("INSERT INTO mp_meetings VALUES(1,1,'2026-01-01'),(2,2,'2026-01-01')")
  c.execute('CREATE TABLE mp_documents(id INTEGER,meeting_id INTEGER,filename TEXT,extracted_text TEXT,doc_date TEXT)');c.execute("INSERT INTO mp_documents VALUES(1,1,'source.pdf',%s,'2026-01-01'),(2,2,'other.pdf','secret','2026-01-01')",(text,))
  c.execute('CREATE TABLE mp_past_questions(id INTEGER,company_id INTEGER,meeting_id INTEGER,question TEXT,response_notes TEXT,status TEXT,topic TEXT)')
  c.execute("INSERT INTO mp_past_questions VALUES(1,1,1,'Margins?','Not yet structural','answered','Margins'),(2,1,1,'Future?','Unasked','planned','Margins')")
 app=Flask(__name__);app.register_blueprint(w.create_blueprint(db));client=app.test_client();url='/api/research/workbench/ABT'
 options=client.get(url).json;assert len(options['sources'])==1 and len(options['answers'])==1
 model={'kind':'model','title':'Margin bridge','owner':'Analyst','rationale':'Interpretation, not management guidance','nextAction':'Ask about durability','dueDate':'2026-01-01','status':'open','caseRevision':1,'assumptionId':aid,'sourceId':1,'sourceHash':hashlib.sha256(text.encode()).hexdigest(),'passage':text,
 'inputs':dict(revenueMillions='1000',sharesMillions='100',beforeMarginPct='10',afterMarginPct='12',taxPct='25',baselineEPS='2',multiple='20',referencePrice='40',currency='USD',period='FY2026',basis='Adjusted diluted',asOf='2026-01-01')}
 def payload(body,revision=0,ident=None):return dict(requestId=str(uuid.uuid4()),id=ident or str(uuid.uuid4()),revision=revision,body=body)
 p=payload(model);assert client.post(url,json=p).status_code==201;assert client.post(url,json=p).json['replayed']
 row=client.get(url).json['records'][0];assert row['body']['calculation']['epsDelta']=='0.1500';assert row['body']['calculation']['afterValue']=='43.0000'
 assert client.post('/api/research/workbench/MDT',json=p).status_code==409
 assert client.post(url,json=payload({**model,'passage':'invented'})).status_code==409
 reviewed=payload({**model,'status':'reviewed','outcome':'Keep as sensitivity, not guidance'},1,p['id']);assert client.post(url,json=reviewed).status_code==201
 assert len(client.get(url+'/'+p['id']+'/history').json['versions'])==2
 assert client.post(url,json=payload(model,1,p['id'])).status_code==409
 follow={**model,'kind':'followup','title':'Follow up on margin answer','answerId':1,'answerHash':options['answers'][0]['hash'],'resolution':'partial'}
 assert client.post(url,json=payload(follow)).status_code==201
 with db(True) as (_,c):c.execute("UPDATE mp_past_questions SET response_notes='Changed response' WHERE id=1")
 assert client.post(url,json=payload(follow)).status_code==409
 under={**model,'kind':'underweight','title':'Nonownership review','mandate':'Value','benchmark':'Test benchmark','asOf':'2026-01-01','holdingPct':'0','benchmarkPct':'2','reason':'valuation','valuationAssessment':'Current valuation needs verification'}
 assert client.post(url,json=payload(under)).status_code==201
 queue=client.get('/api/research/workbench-queue').json;assert len(queue['items'])==2 and all(i['due'] for i in queue['items'])
 context=cm.load(db,'ABT');assert len([e for e in context['entries'] if e['kind']=='analyst_work'])==3
 with db(True) as (_,c):c.execute('INSERT INTO investment_case_versions VALUES(%s,2,%s,NOW())',('ABT',json.dumps({'assumptions':[{'id':aid,'claim':'New case'}]})))
 assert client.post(url,json=payload(model)).status_code==409
 assert all(e['caseBaselineChanged'] for e in cm.load(db,'ABT')['entries'] if e['kind']=='analyst_work')
 print('PASS: model arithmetic, source attribution, actual-answer snapshots, reviewed history, replay/conflicts, stale baselines, underweight active weight, due queue and fresh memory.')
finally:
 with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
 admin.close()
