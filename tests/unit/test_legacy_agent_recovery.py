import json
import unittest
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from legacy_agent_recovery import eligible, record, EVENT
from tests.unit import test_stock_analysis_postgres as pg

class EligibilityTests(unittest.TestCase):
 def test_old_legacy_only_and_unknown_times_fail_closed(self):
  row=dict(status='running',created_at=datetime.now(timezone.utc)-timedelta(days=2),logs=[],report=None)
  self.assertTrue(eligible(row))
  for changes in [dict(status='complete'),dict(report='saved'),dict(has_decision=True),dict(created_at=None),dict(logs='bad'),dict(logs=[None]),dict(logs=[{'ts':'invalid'}]),dict(logs=[{'ts':datetime.now(timezone.utc).isoformat()}])]:
   self.assertFalse(eligible(dict(row,**changes)),changes)
  for event in ['execution_queued','execution_started','batch_enrolled_v1','model_call_v1']:
   self.assertFalse(eligible(dict(row,logs=[{'event':event}])))

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class DispositionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):
   c.execute('CREATE TABLE agent_runs(id TEXT PRIMARY KEY,status TEXT,created_at TIMESTAMP,completed_at TIMESTAMP,logs JSONB,report TEXT,decision TEXT)')
 @classmethod
 def tearDownClass(cls):pg.StockPostgresTests.tearDownClass.__func__(cls)
 def setUp(self):
  with self.db(True) as (_,c):
   c.execute('DELETE FROM agent_runs')
   c.execute("INSERT INTO agent_runs VALUES('legacy','running',NOW()-INTERVAL '2 days',NULL,'[{\"message\":\"saved original step\"}]',NULL,NULL)")
 def test_concurrent_disposition_preserves_evidence_without_completion(self):
  with ThreadPoolExecutor(max_workers=4) as pool:
   results=list(pool.map(lambda _:record(self.db,'legacy'),range(4)))
  self.assertEqual(results.count(True),1)
  with self.db() as (_,c):
   c.execute("SELECT * FROM agent_runs WHERE id='legacy'");row=c.fetchone()
  self.assertEqual(row['status'],'attention');self.assertIsNone(row['completed_at'])
  self.assertIsNone(row['report']);self.assertIsNone(row['decision'])
  self.assertEqual(row['logs'][0],{'message':'saved original step'})
  self.assertEqual([x.get('event') for x in row['logs']].count(EVENT),1)
 def test_reserved_batch_cannot_release_slot(self):
  with self.db(True) as (_,c):
   c.execute("UPDATE agent_runs SET logs=%s::jsonb WHERE id='legacy'",(json.dumps([dict(event='batch_enrolled_v1',batchId='batch',concurrency=1)]),))
  with self.assertRaises(ValueError):record(self.db,'legacy')
  with self.db() as (_,c):
   c.execute("SELECT status FROM agent_runs WHERE id='legacy'");self.assertEqual(c.fetchone()['status'],'running')
