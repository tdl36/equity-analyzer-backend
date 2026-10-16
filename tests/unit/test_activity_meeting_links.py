"""Read-only meeting destination checks against disposable PostgreSQL."""
import json
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from flask import Flask
from activity_snapshot import meeting_destinations, assemble, create_blueprint
from tests.unit.test_activity_snapshot import row
from tests.unit import test_stock_analysis_postgres as pg_tests

class MeetingLinkTests(unittest.TestCase):
 def test_invalid_and_unrelated_ids_do_not_access_database(self):
  def unavailable():raise AssertionError('No lookup expected')
  rows=[{**row(),'stage':'pipeline','requested_meeting_id':x} for x in (None,True,0,-1,'01',' 1','1.0','1e2','1;SELECT',2147483648,{},[],1.0)]
  unrelated={**row(),'source':'research_pipeline_jobs','stage':'pipeline','requested_meeting_id':'1'}
  self.assertTrue(meeting_destinations(unavailable,rows+[unrelated]))
  self.assertTrue(all(r['meeting_link_state']=='missing' and r['meeting_id'] is None for r in rows))
  self.assertNotIn('meeting_link_state',unrelated)
 def test_lookup_failure_keeps_jobs_and_hides_unverified_identity(self):
  @contextmanager
  def unavailable():raise RuntimeError('private database error');yield
  rows=[{**row(),'stage':'command_meeting','requested_meeting_id':'24','input':{'private':'source'}}]
  self.assertFalse(meeting_destinations(unavailable,rows))
  item=assemble(rows,{},datetime.now(timezone.utc))[0]
  self.assertEqual(item['meetingLinkState'],'unavailable');self.assertIsNone(item['meetingId'])
  self.assertEqual(item['jobId'],'1');self.assertNotIn('requested_meeting_id',item);self.assertNotIn('input',item)

@unittest.skipUnless((pg_tests.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class MeetingLinkPostgresTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg_tests.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):
   c.execute("INSERT INTO mp_companies(id,ticker) VALUES(1,'ABT'),(2,'MDT')")
   c.execute('CREATE TABLE mp_meetings(id INTEGER,company_id INTEGER,meeting_date DATE,meeting_type TEXT,status TEXT);CREATE TABLE mp_question_sets(meeting_id INTEGER,status TEXT)')
   c.execute("INSERT INTO mp_meetings(id,company_id) VALUES(24,1),(25,2)")
   c.execute('CREATE TABLE mp_jobs(id TEXT,stage TEXT,ticker TEXT,status TEXT,error TEXT,input JSONB,result JSONB,created_at TIMESTAMP,updated_at TIMESTAMP)')
   for ident,stage,ticker,mid in [('linked','pipeline','ABT',24),('command','command_meeting','MDT','25'),('mismatch','pipeline','MDT',24),('deleted','command_meeting','ABT',99),('malformed','pipeline','ABT','bad')]:
    c.execute("INSERT INTO mp_jobs VALUES(%s,%s,%s,'complete',NULL,%s,'{}',NOW(),NOW())",(ident,stage,ticker,json.dumps({'meetingId':mid,'private':'source'})))
 @classmethod
 def tearDownClass(cls):pg_tests.StockPostgresTests.tearDownClass.__func__(cls)
 def test_snapshot_links_only_existing_same_company_meetings_without_writes(self):
  with self.db() as (_,c):c.execute('SELECT * FROM mp_jobs ORDER BY id');before=c.fetchall()
  app=Flask(__name__);app.register_blueprint(create_blueprint(self.db))
  response=app.test_client().get('/api/activity');d=response.json
  self.assertEqual(response.headers['Cache-Control'],'no-store')
  self.assertNotIn('Meeting destinations',d['unavailable'])
  rows={r['jobId']:r for r in d['items']}
  self.assertEqual((rows['linked']['meetingId'],rows['linked']['meetingLinkState']),(24,'linked'))
  self.assertEqual(rows['command']['meetingId'],25)
  for key,state in [('mismatch','mismatch'),('deleted','missing'),('malformed','missing')]:
   self.assertIsNone(rows[key]['meetingId']);self.assertEqual(rows[key]['meetingLinkState'],state)
  self.assertNotIn('private',str(d));self.assertNotIn('requested_meeting_id',str(d))
  with self.db() as (_,c):c.execute('SELECT * FROM mp_jobs ORDER BY id');self.assertEqual(c.fetchall(),before)
 def test_missing_lookup_dependency_does_not_hide_job_inventory(self):
  with self.db(True) as (_,c):c.execute('ALTER TABLE mp_meetings RENAME TO offline_meetings')
  try:
   app=Flask(__name__);app.register_blueprint(create_blueprint(self.db))
   d=app.test_client().get('/api/activity').json
   self.assertIn('Meeting destinations',d['unavailable']);self.assertEqual(len(d['items']),5)
   rows={r['jobId']:r for r in d['items']}
   self.assertEqual(rows['linked']['meetingLinkState'],'unavailable');self.assertIsNone(rows['linked']['meetingId'])
  finally:
   with self.db(True) as (_,c):c.execute('ALTER TABLE offline_meetings RENAME TO mp_meetings')
