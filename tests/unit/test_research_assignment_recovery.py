"""Recovery uses a disposable PostgreSQL database and synthetic originals only."""
import copy,json,unittest,uuid
from concurrent.futures import ThreadPoolExecutor
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit.test_onepager_edition_contract import edition_fixture,AID,EID
from onepager_edition_contract import baseline,prepare_edition
from onepager_corrections import digest
from thesis_imports import row_hash
from research_assignment_recovery import deliver

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class RecoveryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):
   c.execute('''CREATE TABLE mp_jobs(id TEXT PRIMARY KEY,ticker TEXT,status TEXT,stage TEXT,input JSONB,result JSONB,error TEXT,updated_at TIMESTAMP DEFAULT NOW());
   CREATE TABLE stock_analysis_runs(id TEXT PRIMARY KEY,ticker TEXT,status TEXT,input JSONB,baseline JSONB,state JSONB,sources JSONB);
   CREATE TABLE onepager_corrections(id UUID PRIMARY KEY,run_id TEXT,fingerprint TEXT,payload_hash TEXT,proposal JSONB);
   CREATE TABLE onepager_editions(id UUID PRIMARY KEY,assignment_id TEXT,record JSONB);
   CREATE TABLE external_thesis_drafts(id UUID PRIMARY KEY,ticker TEXT,status TEXT,package JSONB,baseline_hash TEXT,fingerprint TEXT);
   CREATE TABLE portfolio_analyses(ticker TEXT,company TEXT,analysis JSONB,updated_at TIMESTAMP);''')
 @classmethod
 def tearDownClass(cls):pg.StockPostgresTests.tearDownClass.__func__(cls)
 def setUp(self):
  run,a,body,corrections,docs,f=edition_fixture();self.tid=str(uuid.uuid4())
  a['input'].update(outputs=['stock_summary','thesis','visual'],thesisBaseline={'baseline':{'hash':row_hash(None)}})
  a['result'].update(collection={'status':'complete'},artifacts={'stock_summary':'old summary','visual':'old visual'},sources=[{'sha256':s['originalHash']} for s in run['sources']],visualFundamentals=f)
  body.update({k:v for k,v in baseline(run,a,f).items() if k!='ticker'})
  record=prepare_edition(run,a,body,corrections,docs,f)
  package={'baseline':{'hash':row_hash(None)},'sourceRegister':[{'sha256':s['originalHash']} for s in run['sources']]}
  self.original=copy.deepcopy(a['result']);self.run=copy.deepcopy(run)
  with self.db(True) as (_,c):
   for t in ['mp_jobs','stock_analysis_runs','onepager_corrections','onepager_editions','external_thesis_drafts','portfolio_analyses','document_files']:c.execute('DELETE FROM '+t)
   c.execute('INSERT INTO mp_jobs(id,ticker,status,stage,input,result,error) VALUES(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)',(AID,'TEST','attention','research_assignment',json.dumps(a['input']),json.dumps(a['result']),'Original thesis quality failure'))
   c.execute('INSERT INTO stock_analysis_runs VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb)',tuple([run[k] for k in ['id','ticker','status']]+[json.dumps(run[k]) for k in ['input','baseline','state','sources']]))
   for r in corrections:c.execute('INSERT INTO onepager_corrections VALUES(%s,%s,%s,%s,%s::jsonb)',(r['id'],r['run_id'],r['fingerprint'],r['payload_hash'],json.dumps(r['proposal'])))
   for d in docs.values():c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s::jsonb)',('TEST',d['filename'],d['file_data'],json.dumps(d['metadata'])))
   c.execute('INSERT INTO onepager_editions VALUES(%s,%s,%s::jsonb)',(EID,AID,json.dumps(record)))
   c.execute('INSERT INTO external_thesis_drafts VALUES(%s,%s,%s,%s::jsonb,%s,%s)',(self.tid,'TEST','pending',json.dumps(package),row_hash(None),digest(package)))
  self.body={'requestId':str(uuid.uuid4()),'thesisDraftId':self.tid,'editionId':EID}
 def test_delivery_replay_concurrency_preserves_unapproved_originals(self):
  with ThreadPoolExecutor(max_workers=2) as ex:r=list(ex.map(lambda _:deliver(self.db,AID,self.body),range(2)))
  self.assertEqual(sum(x['reused'] for x in r),1)
  with self.db() as (_,c):
   c.execute('SELECT status,result FROM mp_jobs');a=c.fetchone();self.assertEqual(a['status'],'complete');self.assertEqual(a['result']['artifacts'],self.original['artifacts'])
   self.assertEqual(a['result']['reviewPackage']['originalFailure'],'Original thesis quality failure')
   c.execute('SELECT status FROM external_thesis_drafts');self.assertEqual(c.fetchone()['status'],'pending')
   c.execute('SELECT state FROM stock_analysis_runs');self.assertEqual(c.fetchone()['state'],self.run['state'])
   c.execute('SELECT count(*) AS n FROM portfolio_analyses');self.assertEqual(c.fetchone()['n'],0)
  with self.assertRaises(ValueError):deliver(self.db,AID,{**self.body,'requestId':str(uuid.uuid4())})
 def test_stale_restricted_missing_or_wrong_pack_cannot_complete(self):
  mutations=["UPDATE mp_jobs SET status='cancelled'","UPDATE external_thesis_drafts SET baseline_hash='changed'","UPDATE external_thesis_drafts SET status='approved'","UPDATE external_thesis_drafts SET package=jsonb_set(package,'{sourceRegister}','[]')","UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'", "UPDATE document_files SET file_data='Y2hhbmdlZA=='", "DELETE FROM onepager_editions", "UPDATE mp_jobs SET result=jsonb_set(result,'{visualFundamentals,price}','999')"]
  for sql in mutations:
   self.setUp()
   with self.db(True) as (_,c):c.execute(sql)
   with self.subTest(sql=sql),self.assertRaises(ValueError):deliver(self.db,AID,self.body)
   with self.db() as (_,c):c.execute('SELECT result FROM mp_jobs');self.assertNotIn('reviewPackage',c.fetchone()['result'])
 def test_revision_preserves_history_and_replays_without_reverting(self):
  first=deliver(self.db,AID,self.body)['package']
  revision={**self.body,'requestId':str(uuid.uuid4()),'previousRequestId':first['requestId']}
  second=deliver(self.db,AID,revision)['package']
  self.assertEqual(deliver(self.db,AID,self.body)['package'],first)
  self.assertTrue(deliver(self.db,AID,revision)['reused'])
  with self.db() as (_,c):
   c.execute('SELECT result FROM mp_jobs');state=c.fetchone()['result']
   self.assertEqual(state['reviewPackage'],second)
   self.assertEqual(state['reviewPackageHistory'],[first])
   self.assertEqual(state['artifacts'],self.original['artifacts'])
   self.assertEqual(second['originalFailure'],first['originalFailure'])
   c.execute('SELECT count(*) AS n FROM portfolio_analyses');self.assertEqual(c.fetchone()['n'],0)
  with self.assertRaises(ValueError):deliver(self.db,AID,{**revision,'requestId':str(uuid.uuid4())})
 def test_concurrent_revision_and_failed_validation_preserve_current(self):
  first=deliver(self.db,AID,self.body)['package']
  revision={**self.body,'requestId':str(uuid.uuid4()),'previousRequestId':first['requestId']}
  with self.assertRaises(ValueError):deliver(self.db,AID,{**revision,'thesisDraftId':str(uuid.uuid4())})
  def attempt(body):
   try:return deliver(self.db,AID,body)
   except ValueError:return None
  with ThreadPoolExecutor(max_workers=2) as ex:
   results=list(ex.map(attempt,[revision,{**revision,'requestId':str(uuid.uuid4())}]))
  self.assertEqual(sum(x is not None for x in results),1)
  with self.db() as (_,c):
   c.execute('SELECT result FROM mp_jobs');state=c.fetchone()['result']
   self.assertEqual(state['reviewPackageHistory'],[first])
 def test_approval_flags_rejected(self):
  with self.assertRaises(ValueError):deliver(self.db,AID,{**self.body,'approved':True})

class RecoverySummaryTests(unittest.TestCase):
 def test_summary_route_rechecks_both_saved_outputs(self):
  from flask import Flask
  from research_assignments import Coordinator
  app=Flask(__name__);app.config['TESTING']=True
  draft={'ticker':'TEST','status':'pending','stale':False,'fingerprint':'f','package':{'analysis':{'thesis':{'summary':'<script>','pillars':[]},'signposts':[],'threats':[],'conclusion':'Pending'},'sourceRegister':[]}}
  package={'thesisDraftId':'t','editionId':'e','thesisFingerprint':'f'}
  ok=[True]
  def invoke(endpoint,*args):return (draft,200) if endpoint=='thesis_imports.detail' else ({},200 if ok[0] else 409)
  co=Coordinator(app,None,invoke,lambda:True);co.read=lambda _: {'result':{'reviewPackage':package}}
  app.register_blueprint(co.bp);client=app.test_client()
  url='/api/research/assignments/x/review-package/summary'
  r=client.get(url);self.assertEqual(r.status_code,200);self.assertIn('&lt;script&gt;',r.text);self.assertNotIn('<script>',r.text)
  draft['status']='approved';approved=client.get(url);self.assertEqual(approved.status_code,200);self.assertIn('Thesis approved.',approved.text);self.assertNotIn('No thesis changes approved.',approved.text)
  draft['status']='pending'
  ok[0]=False;self.assertEqual(client.get(url).status_code,409)
  ok[0]=True;draft['stale']=True;self.assertEqual(client.get(url).status_code,409)
