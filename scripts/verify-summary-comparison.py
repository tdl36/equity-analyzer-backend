#!/usr/bin/env python3
"""Real SQL/HTTP verification in disposable LOCAL schema. Model work is mocked."""
import sys,uuid,threading,time,json,os
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flask import Flask
import summary_comparison as comparison
schema='charlie_summary_qa_'+uuid.uuid4().hex
admin=psycopg2.connect(dbname='postgres',host='/tmp');admin.autocommit=True
@contextmanager
def db(commit=False):
 c=psycopg2.connect(dbname='postgres',host='/tmp')
 try:
  cur=c.cursor(cursor_factory=RealDictCursor);cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
  yield c,cur
  if commit:c.commit()
 finally:c.close()
entered=threading.Event();release=threading.Event();finished=threading.Event();calls=[]
def fake(source,state,ask,save):
 calls.append(source);state['parts']={'0':{'record':'Preserved checkpoint'}};save(state);entered.set()
 assert release.wait(8)
 if len(calls)==1:raise ValueError('Simulated interruption')
 assert state['parts']['0']['record']=='Preserved checkpoint'
 state['sections']={'brief':'Improved draft'};finished.set();return state
comparison.generate=fake
try:
 with admin.cursor() as c:c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
 with db(True) as (_,c):
  c.execute('CREATE TABLE meeting_summaries(id text,title text,raw_notes text,brief text,summary text,questions text,assessment text,meeting_summary text)')
  c.execute("INSERT INTO meeting_summaries VALUES ('fixture','Original','Full transcript','Original brief','Original takeaways','Original questions','Original assessment','Original record')")
 app=Flask(__name__);bp=comparison.create_blueprint(db);app.register_blueprint(bp);client=app.test_client();url='/api/summaries/fixture/comparisons'
 automatic_id=bp.enqueue('fixture','test-never-used',automatic=True)
 assert automatic_id
 assert bp.enqueue('fixture','test-never-used',automatic=True)==automatic_id
 assert entered.wait(5)
 rows=client.get(url).json['comparisons'];jid=rows[0]['id'];assert rows[0]['state']['parts']
 assert client.post(url,json={'apiKey':'test-never-used'}).json['id']==jid
 time.sleep(.1);assert len(calls)==1,'Duplicate worker ran'
 release.set()
 for _ in range(100):
  if client.get(url).json['comparisons'][0]['status']=='failed':break
  time.sleep(.02)
 assert client.get(url).json['comparisons'][0]['status']=='failed'
 assert client.post(url,json={'apiKey':'test-never-used','resumeId':jid}).status_code==202
 assert finished.wait(5)
 for _ in range(100):
  if client.get(url).json['comparisons'][0]['status']=='complete':break
  time.sleep(.02)
 row=client.get(url).json['comparisons'][0];assert row['status']=='complete'
 assert row['baseline']['brief']=='Original brief'
 assert client.post(url+'/'+jid+'/feedback',json={'feedback':'Prefer original depth'}).status_code==200
 assert client.get(url).json['comparisons'][0]['feedback']=='Prefer original depth'
 assert client.post('/api/summaries/other/comparisons/'+jid+'/feedback',json={'feedback':'Wrong note'}).status_code==404
 with db() as (_,c):
  c.execute("SELECT * FROM meeting_summaries WHERE id='fixture'");r=c.fetchone();assert r['brief']=='Original brief' and r['raw_notes']=='Full transcript'
 assert client.post(url,json={'apiKey':'test-never-used','resumeId':jid}).status_code==202
 time.sleep(.1);assert len(calls)==2,'Completed comparison regenerated'
 assert bp.enqueue('fixture','test-never-used',automatic=True)==jid
 time.sleep(.1);assert len(calls)==2,'Automatic save replayed completed work'
 # Only a stale eligible job can recover; completed/failed/legacy jobs stay put.
 with db(True) as (_,c):
  for ident,status,enabled,attempts in [('abandoned','running',True,0),('exhausted','running',True,2),('legacy','running',False,0),('failed','failed',True,0),('owned','running',True,0)]:
   c.execute("INSERT INTO summary_comparisons(id,summary_id,source_hash,version,source,baseline,state,status,recovery_enabled,recovery_attempts,updated_at) SELECT %s,summary_id,%s,version,source,baseline,state,%s,%s,%s,NOW()-INTERVAL '10 minutes' FROM summary_comparisons WHERE id=%s",(ident,ident,status,enabled,attempts,jid))
 oldkey=os.environ.get('ANTHROPIC_API_KEY');os.environ['ANTHROPIC_API_KEY']='test-never-used'
 held=psycopg2.connect(dbname='postgres',host='/tmp')
 try:
  with held.cursor() as c:c.execute("SELECT pg_advisory_lock(hashtext('summary-comparison:owned'))")
  bp.recover_once()
  with db() as (_,c):
   c.execute('SELECT id,status,recovery_attempts FROM summary_comparisons');states={r['id']:dict(r) for r in c.fetchall()}
  assert states['abandoned']['status']=='complete' and states['abandoned']['recovery_attempts']==1
  assert states['exhausted']['status']=='failed'
  assert states['legacy']['status']=='running' and states['failed']['status']=='failed'
  assert states['owned']['status']=='running' and states['owned']['recovery_attempts']==0
  assert len(calls)==3
 finally:
  held.close()
  if oldkey is None:os.environ.pop('ANTHROPIC_API_KEY',None)
  else:os.environ['ANTHROPIC_API_KEY']=oldkey
 print('PASS: automatic enqueue and replay protection; real SQL checkpoints, concurrent duplicate exclusion, resume, immutable original, scoped feedback, completed replay protection. No provider calls.')
 print('PASS: automatic abandoned-job recovery reuses checkpoints, skips live owners/failed/legacy jobs, and stops at recovery limit.')
finally:
 release.set()
 with admin.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
 admin.close()
