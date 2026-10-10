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
