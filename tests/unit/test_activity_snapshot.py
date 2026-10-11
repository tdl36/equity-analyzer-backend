import unittest
from datetime import datetime,timezone,timedelta
from flask import Flask
from activity_snapshot import assemble,stamp,create_blueprint,SOURCES
from tests.unit import test_stock_analysis_postgres as pg_tests
PG=pg_tests.PG
NOW=datetime(2026,10,10,tzinfo=timezone.utc)
def row(**kw):return dict(id='1',source='mp_jobs',stage='research_assignment',label='Workflow',view='desk',ticker='UNH',status='running',created_at=NOW-timedelta(days=1),updated_at=NOW-timedelta(hours=2),**kw)
class ActivityTests(unittest.TestCase):
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
 def test_timestamps_are_utc(self):
  self.assertEqual(stamp(NOW.replace(tzinfo=None)),NOW.isoformat());self.assertEqual(stamp(NOW.timestamp()),NOW.isoformat());self.assertIsNone(stamp('invalid'))

@unittest.skipUnless((PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class ActivityPostgresTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg_tests.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):
   c.execute('CREATE TABLE mp_jobs(id TEXT,stage TEXT,ticker TEXT,status TEXT,error TEXT,input JSONB,result JSONB,created_at TIMESTAMP,updated_at TIMESTAMP); CREATE TABLE app_settings(key TEXT,value JSONB,updated_at TIMESTAMP); CREATE TABLE agent_heartbeats(last_seen TIMESTAMP);')
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
   self.assertTrue(any(r['jobId']=='old' for r in d['items']));self.assertIn('Research workflows',d['truncated'])
  finally:
   with self.db(True) as (_,c):c.execute("DELETE FROM mp_jobs WHERE id LIKE 'review-%'")

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
