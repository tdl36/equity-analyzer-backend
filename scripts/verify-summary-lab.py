"""Disposable LOCAL schema only. Never imports app_v3 or makes provider calls."""
import sys,uuid,time,threading
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask
import summary_lab
schema='summary_lab_qa_'+uuid.uuid4().hex
admin=psycopg2.connect(dbname='postgres',host='/tmp');admin.autocommit=True
@contextmanager
def db(commit=False):
 c=psycopg2.connect(dbname='postgres',host='/tmp',options='-c search_path='+schema)
 try:
  with c.cursor(cursor_factory=RealDictCursor) as cur:
   yield c,cur
   if commit:c.commit()
 finally:c.close()
entered=threading.Event();release=threading.Event();calls=[]
def fake(source,state,ask,save,focus):
 calls.append(source)
 state['parts']={'0':{'record':source}};save(state)
 if len(calls)==1:
  entered.set();release.wait(5);raise ValueError('test interruption')
 state['sections']={'brief':'Experimental'};save(state)
summary_lab.generate=fake
try:
 with admin.cursor() as c:c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
 with db(True) as (_,c):
  c.execute('CREATE TABLE meeting_summaries(id text,title text,raw_notes text,brief text,summary text,questions text,assessment text,meeting_summary text,created_at timestamptz DEFAULT NOW())')
  c.execute("INSERT INTO meeting_summaries(id,title,raw_notes,brief) VALUES ('fixture','Original','Complete source','Original brief')")
 app=Flask(__name__);app.register_blueprint(summary_lab.create_blueprint(db));client=app.test_client()
 assert len(client.get('/api/summary-lab/sources').json['sources'])==1
 r=client.post('/api/summary-lab',json={'summaryId':'fixture','apiKey':'not-used'});assert r.status_code==202
 jid=r.json['id'];path='/api/summary-lab/'+jid
 assert entered.wait(5)
 client.post(path+'/retry',json={'apiKey':'not-used'});time.sleep(.1);assert len(calls)==1
 release.set()
 for _ in range(100):
  row=client.get(path).json
  if row['status']=='failed':break
  time.sleep(.03)
 assert row['status']=='failed' and row['state']['parts']
 client.post(path+'/retry',json={'apiKey':'not-used'})
 for _ in range(100):
  row=client.get(path).json
  if row['status']=='complete':break
  time.sleep(.03)
 assert row['status']=='complete' and row['baseline']['brief']=='Original brief'
 assert 'not-used' not in str(row)
 assert client.post(path+'/feedback',json={'feedback':'More useful'}).status_code==200
 assert client.get(path).json['feedback']=='More useful'
 client.post(path+'/retry',json={'apiKey':'not-used'});time.sleep(.1);assert len(calls)==2
 with db() as (_,c):
  c.execute("SELECT raw_notes,brief FROM meeting_summaries WHERE id='fixture'");r=c.fetchone()
  assert r['raw_notes']=='Complete source' and r['brief']=='Original brief'
 assert client.post('/api/summary-lab',json={'summaryId':'missing','apiKey':'not-used'}).status_code==404
 print('PASS: independent SQL persistence, frozen baseline, checkpoints, duplicate worker exclusion, retry, completed replay protection, feedback, no persisted key; original unchanged.')
finally:
 release.set()
 with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
 admin.close()
