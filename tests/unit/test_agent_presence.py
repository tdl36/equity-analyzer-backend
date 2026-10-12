import json
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from agent_presence import Presence, describe, inspect
from tests.unit import test_stock_analysis_postgres as pg

class PresenceDescriptionTests(unittest.TestCase):
 def test_fresh_expired_exited_and_mismatched_receipts(self):
  now=datetime.now(timezone.utc);execution=str(uuid4())
  row=dict(value=dict(executionId=execution,state='reporting'),updated_at=now)
  self.assertEqual(describe(row,[execution],now)['state'],'reporting')
  self.assertEqual(describe(row,[execution],now+timedelta(seconds=91))['state'],'stale')
  self.assertEqual(describe(row,[],now)['state'],'unknown')
  self.assertEqual(describe(row,[execution],now-timedelta(seconds=1))['state'],'unknown')
  row['value']['state']='exited'
  self.assertEqual(describe(row,[execution],now+timedelta(days=1))['state'],'exited')
  for value in [None,'bad',[],{'state':'reporting'}]:
   self.assertEqual(describe(dict(row,value=value),[execution],now)['state'],'unknown')
 def test_initial_persistence_failure_does_not_start_reporter(self):
  @contextmanager
  def broken(commit=False):
   raise RuntimeError('unavailable')
   yield
  with self.assertRaises(RuntimeError):Presence(broken,str(uuid4()),str(uuid4()))

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class PresencePersistenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):c.execute('CREATE TABLE app_settings(key TEXT PRIMARY KEY,value JSONB,updated_at TIMESTAMP)')
 @classmethod
 def tearDownClass(cls):pg.StockPostgresTests.tearDownClass.__func__(cls)
 def test_reporting_and_exit_preserve_exact_execution_identity(self):
  run,execution=str(uuid4()),str(uuid4())
  reporter=Presence(self.db,run,execution)
  try:
   self.assertEqual(inspect(self.db,run,[execution])['state'],'reporting')
   self.assertEqual(inspect(self.db,run,[str(uuid4())])['state'],'unknown')
  finally:reporter.close()
  self.assertEqual(inspect(self.db,run,[execution])['state'],'exited')
  self.assertFalse(reporter.thread.is_alive())
