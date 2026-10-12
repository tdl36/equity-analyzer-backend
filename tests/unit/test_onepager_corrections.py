import base64
import copy
import hashlib
import unittest
from contextlib import contextmanager
from flask import Flask
from onepager_corrections import preparation, prepare, create_blueprint

RID='6977b8d3-e286-5848-b377-159dcbfc0f21'
CID='d998e805-0bd4-49ce-aafa-2af123e1aace'
TEXT='Synthetic source passage about revenue and operating performance.'

def fixture():
 sha=hashlib.sha256(TEXT.encode()).hexdigest()
 run={'id':RID,'ticker':'TEST','status':'complete','input':{'hashes':{'fixture.txt':sha}},'baseline':{'revision':1},'sources':[{'id':'s1','filename':'fixture.txt','text':TEXT,'originalHash':sha}], 'state':{'report':{'summary':{'earnings_direction':'Old claim'}},'citations':{}}}
 doc={'filename':'fixture.txt','file_data':base64.b64encode(TEXT.encode()).decode(),'metadata':{}}
 body={'requestId':CID,'fingerprint':preparation(run)['fingerprint'],'path':'/summary/earnings_direction','statement':'Proposed claim','basis':'interpretation','reason':'Needs qualification','sourceId':'s1','excerpt':TEXT,'locator':'Page 1'}
 return run,doc,body

class Cursor:
 def __init__(self):self.run,self.doc,_=fixture();self.saved={};self.row=None;self.rows=[];self.sql=[];self.exists=False
 def execute(self,q,args=()):
  self.sql.append(q)
  if 'FROM stock_analysis_runs' in q:self.row=self.run
  elif 'to_regclass' in q:self.row={'name':'onepager_corrections' if self.exists else None}
  elif 'CREATE TABLE' in q:self.exists=True
  elif 'FROM document_files' in q:self.row=self.doc
  elif 'FROM onepager_corrections WHERE id' in q:self.row=self.saved.get(args[0])
  elif 'FROM onepager_corrections WHERE run_id' in q:self.rows=list(self.saved.values())
  elif 'INSERT INTO onepager_corrections' in q:
   import json
   self.saved[args[0]]={'id':args[0],'run_id':args[1],'fingerprint':args[2],'payload_hash':args[3],'proposal':json.loads(args[4]),'created_at':'2026-10-12'}
 def fetchone(self):return self.row
 def fetchall(self):return self.rows

class CorrectionTests(unittest.TestCase):
 def test_private_preparation_and_immutable_proposal(self):
  run,doc,body=fixture();before=copy.deepcopy(run);p=preparation(run)
  self.assertNotIn(TEXT,str(p));ident,_,proposal=prepare(run,body,doc)
  self.assertEqual(ident,CID);self.assertEqual(proposal['review'],'pending');self.assertTrue(proposal['passageMatched'])
  self.assertEqual(proposal['before'],'Old claim');self.assertEqual(run,before)
 def test_all_baselines_invalidate_submission(self):
  run,doc,body=fixture()
  for key in ['state','sources','input','baseline','ticker']:
   changed=copy.deepcopy(run)
   if key=='ticker':changed[key]='OTHER'
   elif key=='sources':changed[key][0]['text']+=' changed'
   else:changed[key]['changed']=True
   with self.subTest(key=key),self.assertRaises(ValueError):prepare(changed,body,doc)
 def test_no_forged_support_unknown_paths_or_silent_truncation(self):
  run,doc,body=fixture()
  for patch in [{'review':'supported'},{'path':'/summary'},{'path':'/unknown'},{'path':[]},{'statement':'x'*2001},{'statement':'Old claim'},{'basis':'approved'},{'sourceId':'another'},{'excerpt':'invented supporting excerpt that is not present'},{'locator':''}]:
   with self.subTest(patch=patch),self.assertRaises(ValueError):prepare(run,{**body,**patch},doc)
 def test_original_bytes_and_usage_rechecked(self):
  run,doc,body=fixture()
  for changed in [None,{**doc,'metadata':{'usage':'reference_only'}},{**doc,'file_data':base64.b64encode(b'changed').decode()}]:
   with self.assertRaises(ValueError):prepare(run,body,changed)
 def test_get_read_only_post_pending_and_replay_safe(self):
  c=Cursor();commits=[]
  @contextmanager
  def db(commit=False):commits.append(commit);yield None,c
  app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client();url=f'/api/research/stock-analysis-run/{RID}/corrections'
  result=client.get(url);self.assertEqual(result.status_code,200);self.assertEqual(commits,[False]);self.assertFalse(any('CREATE' in q or 'INSERT' in q for q in c.sql))
  body=fixture()[2];first=client.post(url,json=body);self.assertEqual(first.status_code,201);self.assertEqual(first.json['status'],'pending')
  self.assertTrue(client.post(url,json=body).json['reused']);self.assertEqual(len(c.saved),1)
  self.assertEqual(client.post(url,json={**body,'statement':'Changed wording'}).status_code,409)
  self.assertEqual(client.get(url).json['corrections'][0]['proposal']['statement'],'Proposed claim')
  c.run['baseline']['revision']=2
  self.assertTrue(client.get(url).json['corrections'][0]['stale'])
  self.assertFalse(any(q.startswith('UPDATE') for q in c.sql))
 def test_policy_preflight_exposes_restriction_and_keeps_history_readable(self):
  c=Cursor()
  @contextmanager
  def db(commit=False):
   self.assertFalse(commit);yield None,c
  app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client();url=f'/api/research/stock-analysis-run/{RID}/corrections'
  self.assertTrue(client.get(url).json['sourcePolicyReady'])
  c.doc['metadata']={'aiAllowed':False}
  data=client.get(url).json
  self.assertFalse(data['sourcePolicyReady']);self.assertEqual(data['sources'][0]['policyStatus'],'restricted');self.assertIn('corrections',data)
  c.doc=None
  self.assertEqual(client.get(url).json['sources'][0]['policyStatus'],'missing')
  self.assertFalse(any('UPDATE' in q or 'INSERT' in q for q in c.sql))
 def test_invalid_body_and_incomplete_report(self):
  run,doc,body=fixture()
  with self.assertRaises(ValueError):preparation({**run,'status':'running'})
  with self.assertRaises(ValueError):prepare(run,[],doc)

from tests.unit import test_stock_analysis_postgres as pg_tests
@unittest.skipUnless((pg_tests.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class CorrectionPostgresTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg_tests.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):c.execute('CREATE TABLE stock_analysis_runs(id TEXT PRIMARY KEY,ticker TEXT,status TEXT,input JSONB,baseline JSONB,state JSONB,sources JSONB)')
 @classmethod
 def tearDownClass(cls):pg_tests.StockPostgresTests.tearDownClass.__func__(cls)
 def test_concurrent_submission_and_restart_keep_one_pending_original(self):
  import json
  from concurrent.futures import ThreadPoolExecutor
  run,doc,body=fixture()
  with self.db(True) as (_,c):
   c.execute('INSERT INTO stock_analysis_runs VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb)',tuple([run[k] for k in ['id','ticker','status']]+[json.dumps(run[k]) for k in ['input','baseline','state','sources']]))
   c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s::jsonb)',(run['ticker'],doc['filename'],doc['file_data'],'{}'))
  def client():
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));return app.test_client()
  url=f'/api/research/stock-analysis-run/{RID}/corrections'
  def submit(_):
   r=client().post(url,json=body);self.assertIn(r.status_code,[200,201],r.json);return r.json
  with ThreadPoolExecutor(max_workers=2) as pool:a,b=list(pool.map(submit,range(2)))
  self.assertEqual(a['id'],b['id']);self.assertEqual(sum(r['reused'] for r in [a,b]),1)
  self.assertTrue(submit(0)['reused'])
  r=client().get(url);self.assertEqual(len(r.json['corrections']),1);self.assertEqual(r.headers['Cache-Control'],'private, no-store')
  with self.db() as (_,c):
   c.execute('SELECT state FROM stock_analysis_runs WHERE id=%s',(RID,));self.assertEqual(c.fetchone()['state'],run['state'])
  changed={**body,'requestId':'c998e805-0bd4-49ce-aafa-2af123e1aace','excerpt':'Unmatched text that should cause the transaction to roll back.'}
  self.assertEqual(client().post(url,json=changed).status_code,400)
  self.assertEqual(len(client().get(url).json['corrections']),1)
