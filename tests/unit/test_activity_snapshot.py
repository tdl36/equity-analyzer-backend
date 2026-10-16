import unittest
from datetime import datetime,timezone,timedelta
from flask import Flask
from activity_snapshot import assemble,stamp,create_blueprint,SOURCES
from tests.unit import test_stock_analysis_postgres as pg_tests
PG=pg_tests.PG
NOW=datetime(2026,10,10,tzinfo=timezone.utc)
def row(**kw):return dict(id='1',source='mp_jobs',stage='research_assignment',label='Workflow',view='desk',ticker='UNH',status='running',created_at=NOW-timedelta(days=1),updated_at=NOW-timedelta(hours=2),**kw)
class ActivityTests(unittest.TestCase):
 def test_quality_alert_does_not_promise_another_paid_retry(self):
  r=row();r.update(status='attention',error='No supported thesis statements were produced. Resume explicitly to retry.')
  item=assemble([r],{},NOW)[0]
  self.assertIn('manual-review options',item['error']);self.assertNotIn('Resume explicitly',item['error'])
  self.assertIn('Resume explicitly',r['error'])

 def test_parent_deduplicates_children_and_keeps_unrelated(self):
  p=row(report_id='r1',refresh_id='c1')
  child={**p,'source':'stock_analysis_runs','stage':None,'id':'r1'}
  command={**p,'stage':'collection_control','id':'cmd','assignment_id':'1'}
  other={**child,'id':'other','ticker':'SYK'}
  items=assemble([p,child,command,other],{'requests':[{'id':'c1','status':'queued'}]},NOW)
  self.assertEqual(len(items),2);self.assertEqual(items[0]['view'],'stockanalysis');self.assertTrue(items[0]['stale'])
 def test_attention_survives_age_and_unknown_collection_time_stays_unknown(self):
  r=row();r.update(status='attention',error='Check authentication')
  items=assemble([r],{'requests':[{'id':'x','ticker':'PFE','status':'queued','created':NOW.timestamp()}]},NOW)
  self.assertEqual(items[0]['bucket'],'attention');self.assertIsNone(items[1]['updatedAt']);self.assertTrue(items[1]['stale'])
 def test_review_requests_are_not_running_and_old_work_is_retained(self):
  records=[]
  for i,status in enumerate(('needs_review','awaiting_approval','proposed','submitted','unrecognized')):
   r=row();r.update(id=str(i),status=status);records.append(r)
  old=row();old.update(id='old',updated_at=NOW-timedelta(days=20));records.append(old)
  result={r['jobId']:r for r in assemble(records,{},NOW)}
  self.assertEqual(result['0']['bucket'],'attention');self.assertEqual(result['1']['bucket'],'attention')
  self.assertEqual(result['2']['bucket'],'attention');self.assertEqual(result['3']['bucket'],'recent')
  self.assertEqual(result['4']['bucket'],'attention');self.assertEqual(result['old']['bucket'],'older')
 def test_catalyst_reason_and_workflow_are_projected_without_payload(self):
  r=row();r.update(stage='catalyst_signal',status='needs_review',reason='Check possible duplicate',input={'private':'secret'})
  item=assemble([r],{},NOW)[0]
  self.assertEqual(item['title'],'Catalyst signal');self.assertEqual(item['workflow'],'catalyst_signal')
  self.assertEqual(item['step'],'Check possible duplicate');self.assertNotIn('input',item)
 def test_legacy_agent_uses_progress_time_and_keeps_uncertain_execution_visible(self):
  base={**row(),'source':'agent_runs','stage':None,'label':'Analyst team','updated_at':NOW-timedelta(days=200)}
  for last in (None,'invalid',(NOW-timedelta(days=200)).isoformat(),(NOW+timedelta(hours=1)).isoformat()):
   item=assemble([{**base,'last_log_at':last}],{},NOW)[0]
   self.assertEqual(item['bucket'],'attention');self.assertTrue(item['executionUnconfirmed'])
   self.assertEqual(item['status'],'running');self.assertIn('another charge',item['warning'])
  fresh=assemble([{**base,'last_log_at':NOW.isoformat()}],{},NOW)[0]
  self.assertEqual(fresh['bucket'],'active');self.assertFalse(fresh['stale'])
  self.assertNotIn('executionUnconfirmed',fresh)
  done=assemble([{**base,'status':'complete','last_log_at':NOW.isoformat()}],{},NOW)[0]
  self.assertEqual(done['bucket'],'recent');self.assertNotIn('executionUnconfirmed',done)
 def test_recorded_legacy_outcome_stays_visible_even_when_old(self):
  r={**row(),'source':'agent_runs','stage':None,'status':'attention','legacy_unresolved':True,'updated_at':NOW-timedelta(days=200)}
  item=assemble([r],{},NOW)[0]
  self.assertEqual(item['bucket'],'attention')
  self.assertEqual(item['step'],'Legacy outcome unresolved')
  self.assertIn('No cancellation or retry',item['warning'])
 def test_timestamps_are_utc(self):
  self.assertEqual(stamp(NOW.replace(tzinfo=None)),NOW.isoformat());self.assertEqual(stamp(NOW.timestamp()),NOW.isoformat());self.assertIsNone(stamp('invalid'))

@unittest.skipUnless((PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class ActivityPostgresTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg_tests.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):
   c.execute('CREATE TABLE mp_jobs(id TEXT,stage TEXT,ticker TEXT,status TEXT,error TEXT,input JSONB,result JSONB,created_at TIMESTAMP,updated_at TIMESTAMP); CREATE TABLE app_settings(key TEXT,value JSONB,updated_at TIMESTAMP); CREATE TABLE agent_heartbeats(last_seen TIMESTAMP);')
   c.execute('CREATE TABLE mp_meetings(id SERIAL PRIMARY KEY,company_id INTEGER,meeting_date DATE,meeting_type TEXT,status TEXT); CREATE TABLE mp_question_sets(id SERIAL PRIMARY KEY,meeting_id INTEGER,status TEXT);')
   c.execute("INSERT INTO mp_jobs VALUES('old','research_assignment','UNH','running',NULL,'{}','{\"step\":\"Waiting for Mac\"}',NOW()-INTERVAL '2 days',NOW()-INTERVAL '2 days')")
   c.execute("INSERT INTO mp_jobs SELECT 'recent-'||i,'pipeline','SYK','done',NULL,'{}','{}',NOW(),NOW() FROM generate_series(1,40)i")
 @classmethod
 def tearDownClass(cls):pg_tests.StockPostgresTests.tearDownClass.__func__(cls)
 def test_old_active_is_not_hidden_by_recent_history_and_no_private_payload(self):
  app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
  self.assertEqual(d['unavailable'],[]);self.assertEqual(d['counts']['active'],1);self.assertEqual(d['counts']['recent'],20)
  old=next(x for x in d['items'] if x['jobId']=='old');self.assertTrue(old['stale']);self.assertNotIn('input',old);self.assertNotIn('result',old)
 def test_review_backlog_cannot_hide_older_active_work(self):
  with self.db(True) as (_,c):c.execute("INSERT INTO mp_jobs SELECT 'review-'||i,'evidence_amendment','TEST','needs_review',NULL,'{}','{}',NOW(),NOW() FROM generate_series(1,205)i")
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
   self.assertTrue(any(r['jobId']=='old' for r in d['items']));self.assertIn('Research workflows (attention)',d['truncated'])
  finally:
   with self.db(True) as (_,c):c.execute("DELETE FROM mp_jobs WHERE id LIKE 'review-%'")

 def test_background_backlogs_cannot_hide_assignment_delivery_or_recovery(self):
  with self.db(True) as (_,c):
   for status in ('running','attention','complete'):
    c.execute("INSERT INTO mp_jobs SELECT 'scope-'||%s||i,'auto_evidence_intake','TEST',%s,NULL,'{\"private\":\"source\"}','{}',NOW(),NOW() FROM generate_series(1,205)i",(status,status))
    c.execute("INSERT INTO mp_jobs VALUES('scope-assignment-'||%s,'research_assignment','ABT',%s,NULL,'{\"private\":\"source\"}','{}',NOW()-INTERVAL '2 hours',NOW()-INTERVAL '2 hours')",(status,status))
   # Legacy NULL stages belong to other workflows and must not disappear.
   c.execute("INSERT INTO mp_jobs VALUES('scope-null',NULL,'TEST','queued',NULL,'{}','{}',NOW(),NOW()+INTERVAL '1 second')")
   c.execute("SELECT id,status,input,result,updated_at FROM mp_jobs ORDER BY id");before=c.fetchall()
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
   self.assertEqual(d['unavailable'],[])
   ids=[x['jobId'] for x in d['items']]
   self.assertEqual(len(ids),len(set(ids)))
   self.assertTrue({'scope-assignment-running','scope-assignment-attention','scope-assignment-complete','old','scope-null'}<=set(ids))
   self.assertTrue(all('Research workflows ('+k+')' in d['truncated'] for k in ('active','attention','recent')))
   self.assertFalse(any(x.startswith('Research assignments') for x in d['truncated']))
   self.assertNotIn('private',str(d))
   with self.db() as (_,c):
    c.execute("SELECT id,status,input,result,updated_at FROM mp_jobs ORDER BY id");self.assertEqual(c.fetchall(),before)
  finally:
   with self.db(True) as (_,c):c.execute("DELETE FROM mp_jobs WHERE id LIKE 'scope-%'")

 def test_alerts_ignore_unrelated_backlog_and_recovered_child(self):
  with self.db(True) as (_,c):
   c.execute("INSERT INTO mp_jobs SELECT 'alert-noise-'||i,'evidence_amendment','TEST','attention',NULL,'{}','{}',NOW(),NOW() FROM generate_series(1,250)i")
   c.execute("INSERT INTO mp_jobs VALUES('alert-real','research_assignment','ABT','blocked','Needs recovery','{}','{}',NOW()-INTERVAL '10 days',NOW()-INTERVAL '10 days'),('alert-parent','research_assignment','DONE','complete',NULL,'{}','{\"reportId\":\"alert-child\"}',NOW(),NOW())")
   c.execute("CREATE TABLE stock_analysis_runs(id TEXT,ticker TEXT,status TEXT,error TEXT,created_at TIMESTAMP,updated_at TIMESTAMP)")
   c.execute("INSERT INTO stock_analysis_runs VALUES('alert-child','DONE','attention','old failure',NOW(),NOW()),('alert-independent','OTHER','failed','inspect',NOW(),NOW())")
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));r=app.test_client().get('/api/activity/research-alerts');d=r.json
   self.assertEqual(d['unavailable'],[]);self.assertEqual(d['truncated'],[])
   self.assertEqual({x['ticker'] for x in d['items']},{'ABT','OTHER'})
   self.assertEqual(r.headers['Cache-Control'],'no-store')
   self.assertTrue(all('input' not in x and 'result' not in x for x in d['items']))
  finally:
   with self.db(True) as (_,c):c.execute("DELETE FROM mp_jobs WHERE id LIKE 'alert-%'; DROP TABLE stock_analysis_runs")

 def test_legacy_agent_progress_projection_is_read_only(self):
  with self.db(True) as (_,c):
   c.execute('CREATE TABLE agent_runs(id TEXT,ticker TEXT,status TEXT,created_at TIMESTAMP,completed_at TIMESTAMP,logs JSONB,llm_provider TEXT,llm_model TEXT,report TEXT,decision TEXT)')
   c.execute("INSERT INTO agent_runs(id,ticker,status,created_at,completed_at,logs) VALUES('legacy','WELL','running',NOW()-INTERVAL '200 days',NULL,'[]')")
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
   self.assertEqual(d['unavailable'],[])
   item=next(x for x in d['items'] if x['jobId']=='legacy')
   self.assertTrue(item['executionUnconfirmed']);self.assertEqual(item['bucket'],'attention')
   from agent_execution import queued_log
   for recover in (False,True):
    for minutes in (1,20,20000):
     with self.db(True) as (_,c):
      c.execute("DELETE FROM agent_runs WHERE id='pending'")
      c.execute("INSERT INTO agent_runs(id,ticker,status,created_at,logs) VALUES('pending','TEST','queued',(NOW() AT TIME ZONE 'UTC')-(%s * INTERVAL '1 minute'),%s)",(minutes,queued_log(recover)))
     result=app.test_client().get('/api/activity').json
     pending=next(x for x in result['items'] if x['jobId']=='pending')
     self.assertTrue(pending['pendingStart']);self.assertEqual(pending['startRecovery'],recover)
     self.assertNotIn('executionUnconfirmed',pending)
     self.assertEqual(pending['bucket'],'active' if minutes==1 else 'attention')
     self.assertEqual(pending['step'],'Waiting for first execution')
   with self.db(True) as (_,c):
    c.execute("UPDATE agent_runs SET logs=logs || '[{\"event\":\"execution_started\"}]'::jsonb WHERE id='pending'")
   attempted=next(x for x in app.test_client().get('/api/activity').json['items'] if x['jobId']=='pending')
   self.assertFalse(attempted['pendingStart']);self.assertTrue(attempted['executionUnconfirmed'])

   with self.db() as (_,c):
    c.execute("SELECT status FROM agent_runs WHERE id='legacy'");self.assertEqual(c.fetchone()['status'],'running')
  finally:
   with self.db(True) as (_,c):c.execute('DROP TABLE agent_runs')
 def test_exact_run_inspection_is_bounded_and_does_not_return_report_bodies(self):
  import json
  run='00000000-0000-4000-8000-000000000167'
  with self.db(True) as (_,c):
   c.execute('CREATE TABLE agent_runs(id TEXT,ticker TEXT,status TEXT,created_at TIMESTAMP,completed_at TIMESTAMP,logs JSONB,llm_provider TEXT,llm_model TEXT,report TEXT,decision TEXT)')
   c.execute("INSERT INTO agent_runs VALUES(%s,'TEST','running',NOW(),NULL,%s,'anthropic','fixture','private report','private decision')",(run,json.dumps([{'ts':NOW.isoformat(),'agent':'system','message':str(i)} for i in range(110)])))
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));client=app.test_client()
   self.assertEqual(client.get('/api/activity/analyst-run/invalid').status_code,400)
   self.assertEqual(client.get('/api/activity/analyst-run/00000000-0000-4000-8000-000000000168').status_code,404)
   response=client.get('/api/activity/analyst-run/'+run);d=response.json
   self.assertEqual(response.headers['Cache-Control'],'no-store');self.assertEqual(d['id'],run)
   self.assertTrue(d['hasReport']);self.assertTrue(d['hasDecision']);self.assertNotIn('report',d)
   self.assertEqual(len(d['logs']),100);self.assertEqual(d['logs'][0]['message'],'10');self.assertTrue(d['truncated'])
   self.assertEqual(d['status'],'running')
   self.assertEqual(d['executionEvidence']['reservationCount'],0)
   self.assertEqual(d['executionEvidence']['providerOutcome'],'unverified')
   # Enrollment metadata must not imply research exists, irrespective of status.
   for status in ('queued','running','failed','complete'):
    for report,expected in ((None,False),('  ',False),(' { "batchId": "fixture" } ',False),('{}',False),('# Saved research',True)):
     with self.db(True) as (_,c):
      c.execute('UPDATE agent_runs SET status=%s,report=%s WHERE id=%s',(status,report,run))
     result=client.get('/api/activity/analyst-run/'+run)
     self.assertEqual(result.status_code,200)
     self.assertEqual(result.json['hasReport'],expected,(status,report))
     self.assertNotIn('report',result.json)
     with self.db() as (_,c):
      c.execute('SELECT status,report FROM agent_runs WHERE id=%s',(run,))
      saved=c.fetchone();self.assertEqual(saved['status'],status);self.assertEqual(saved['report'],report)
   with self.db(True) as (_,c):
    c.execute("UPDATE agent_runs SET status='running',report='private report' WHERE id=%s",(run,))

   with self.db() as (_,c):
    c.execute('SELECT status,report FROM agent_runs WHERE id=%s',(run,));r=c.fetchone();self.assertEqual(r['status'],'running');self.assertEqual(r['report'],'private report')
  finally:
   with self.db(True) as (_,c):c.execute('DROP TABLE agent_runs')
 def test_blocked_assignment_survives_more_than_200_active_jobs(self):
  with self.db(True) as (_,c):
   c.execute("INSERT INTO mp_jobs SELECT 'alert-active-'||i,'pipeline','TEST','running',NULL,'{}','{}',NOW(),NOW() FROM generate_series(1,205)i")
   c.execute("INSERT INTO mp_jobs VALUES('alert-blocked','research_assignment','ABT','attention','Needs source review','{}','{}',NOW()-INTERVAL '1 day',NOW()-INTERVAL '1 day')")
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
   self.assertTrue(any(x['jobId']=='alert-blocked' and x['status']=='attention' for x in d['items']))
   self.assertIn('Research workflows (active)',d['truncated'])
   self.assertNotIn('Research workflows (attention)',d['truncated'])
   with self.db(True) as (_,c):
    c.execute("INSERT INTO mp_jobs SELECT 'alert-attention-'||i,'research_assignment','TEST','attention','Review','{}','{}',NOW(),NOW() FROM generate_series(1,205)i")
   d=app.test_client().get('/api/activity').json
   self.assertIn('Research assignments (attention)',d['truncated'])
   with self.db() as (_,c):
    c.execute("SELECT status FROM mp_jobs WHERE id='alert-blocked'");self.assertEqual(c.fetchone()['status'],'attention')
  finally:
   with self.db(True) as (_,c):c.execute("DELETE FROM mp_jobs WHERE id LIKE 'alert-%'")
 def test_delivery_aggregate_includes_old_commands_beyond_recent_history(self):
  with self.db(True) as (_,c):
   c.execute("INSERT INTO mp_jobs SELECT 'delivery-'||i,'collection_control','TEST','applied',NULL,'{}','{}',NOW(),NOW() FROM generate_series(1,60)i")
   c.execute("INSERT INTO mp_jobs VALUES('delivery-old','collection_control','TEST','queued',NULL,'{}','{}',NOW()-INTERVAL '20 hours',NOW()-INTERVAL '20 hours')")
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
   self.assertEqual(d['collectionDelivery']['pending'],1)
   self.assertIsNotNone(d['collectionDelivery']['oldestPendingAt'])
  finally:
   with self.db(True) as (_,c):c.execute("DELETE FROM mp_jobs WHERE id LIKE 'delivery-%'")
 def test_intake_wait_explanation_uses_saved_policy_and_company_gate(self):
  with self.db(True) as (_,c):
   c.execute('CREATE TABLE IF NOT EXISTS portfolio_analyses(ticker TEXT)')
   c.execute("INSERT INTO app_settings VALUES('research_auto_intake_v1','{\"enabled\":true,\"dailyLimit\":2,\"used\":0}',NOW())")
   c.execute("INSERT INTO mp_jobs VALUES('intake-fixture','auto_evidence_intake','TEST','waiting',NULL,'{}','{}',NOW(),NOW())")
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
   item=next(x for x in d['items'] if x['jobId']=='intake-fixture')
   self.assertEqual(item['waitCode'],'baseline');self.assertIn('saved thesis',item['step'])
   self.assertEqual(d['unavailable'],[])
   with self.db() as (_,c):
    c.execute("SELECT status FROM mp_jobs WHERE id='intake-fixture'");self.assertEqual(c.fetchone()['status'],'waiting')
  finally:
   with self.db(True) as (_,c):
    c.execute("DELETE FROM mp_jobs WHERE id='intake-fixture'")
    c.execute("DELETE FROM app_settings WHERE key='research_auto_intake_v1'")

 def test_due_review_excludes_superseded_and_future_without_mutation(self):
  with self.db(True) as (_,c):
   c.execute('CREATE TABLE research_decisions(id TEXT,ticker TEXT,revision INTEGER,body JSONB,created_at TIMESTAMP)')
   c.execute("INSERT INTO research_decisions VALUES('old','TEST',1,'{\"issue\":\"Old question\",\"reviewDate\":\"2000-01-01\"}',NOW()-INTERVAL '20 days'),('followup','TEST',2,'{\"issue\":\"Old question\",\"supersedes\":\"old\",\"reviewDate\":\"2099-01-01\"}',NOW()),('due','TEST',3,'{\"issue\":\"Current question\",\"reviewDate\":\"2000-01-01\"}',NOW()-INTERVAL '20 days')")
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db));d=app.test_client().get('/api/activity').json
   rows=[r for r in d['items'] if r['workflow']=='research_decision']
   self.assertEqual([r['jobId'] for r in rows],['due']);self.assertEqual(rows[0]['bucket'],'attention')
   self.assertEqual(d['unavailable'],[])
   with self.db() as (_,c):
    c.execute('SELECT COUNT(*) AS n FROM research_decisions');self.assertEqual(c.fetchone()['n'],3)
  finally:
   with self.db(True) as (_,c):c.execute('DROP TABLE research_decisions')
