import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from agent_execution import claim, queued_log, recover_pending
from tests.unit import test_stock_analysis_postgres as pg

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class ExecutionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):
   c.execute('CREATE TABLE agent_runs(id TEXT PRIMARY KEY,status TEXT,logs JSONB,completed_at TIMESTAMP,created_at TIMESTAMP DEFAULT NOW(),ticker TEXT,analysis_date DATE,llm_provider TEXT,llm_model TEXT)')
 @classmethod
 def tearDownClass(cls):pg.StockPostgresTests.tearDownClass.__func__(cls)
 def add(self,id,status='queued',logs=None):
  with self.db(True) as (_,c):c.execute('INSERT INTO agent_runs(id,status,logs,completed_at) VALUES(%s,%s,%s,NULL)',(id,status,logs if logs is not None else queued_log()))
 def test_concurrent_claims_reserve_exactly_once(self):
  self.add('concurrent')
  with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(lambda _:claim(self.db,'concurrent'),range(6)))
  self.assertEqual(sum(bool(x) for x in results),1)
  with self.db() as (_,c):
   c.execute("SELECT status,logs FROM agent_runs WHERE id='concurrent'");r=c.fetchone()
  self.assertEqual(r['status'],'running');self.assertEqual(len(r['logs']),2)
  self.assertEqual(r['logs'][-1]['executionId'],next(x for x in results if x))
  self.assertIsNone(claim(self.db,'concurrent'))
 def test_legacy_terminal_and_requeued_attempts_are_not_replayed(self):
  for id,status,logs in [('legacy','running','[]'),('legacy-queued','queued','[]'),('done','complete',queued_log()),('requeued','queued','[{"event":"execution_queued"},{"event":"execution_started"}]')]:
   self.add(id,status,logs);self.assertIsNone(claim(self.db,id))
  self.assertIsNone(claim(self.db,'missing'))
 def test_recovery_only_enrolled_unstarted_singles_and_saved_parameters(self):
  import json
  for id,logs,status,old in [('recover-safe',queued_log(True),'queued',True),('recover-batch',queued_log(),'queued',True),('recover-fresh',queued_log(True),'queued',False),('recover-old','[]','queued',True),('recover-started',json.dumps(json.loads(queued_log(True))+[{'event':'execution_started'}]),'queued',True),('recover-terminal',queued_log(True),'complete',True)]:
   self.add(id,status,logs)
   with self.db(True) as (_,c):
    c.execute("UPDATE agent_runs SET ticker='TEST',analysis_date='2026-10-09',llm_provider='saved-provider',llm_model='saved-model',created_at=NOW()-(%s * INTERVAL '1 minute') WHERE id=%s",(10 if old else 0,id))
  calls=[]
  def run(*args):
   if claim(self.db,args[0]):calls.append(args)
  with ThreadPoolExecutor(max_workers=4) as pool:
   list(pool.map(lambda _:recover_pending(self.db,run),range(4)))
  self.assertEqual(calls,[('recover-safe','TEST','2026-10-09','saved-provider','saved-model')])
  self.assertEqual(recover_pending(self.db,run),0)
  with self.db() as (_,c):
   c.execute("SELECT id,status FROM agent_runs WHERE id LIKE 'recover-%'")
   states={r['id']:r['status'] for r in c.fetchall()}
  self.assertEqual(states['recover-batch'],'queued');self.assertEqual(states['recover-started'],'queued')
  self.assertEqual(states['recover-terminal'],'complete');self.assertEqual(states['recover-safe'],'running')
 def test_batch_and_single_enroll_and_graph_reserves_before_import(self):
  tree=ast.parse(Path('app_v3.py').read_text())
  funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
  for name in ['start_agent_run','start_agent_batch_run']:
   self.assertIn('queued_log',ast.unparse(funcs[name]))
  self.assertIn('recover_single=True',ast.unparse(funcs['start_agent_run']))
  self.assertNotIn('recover_single=True',ast.unparse(funcs['start_agent_batch_run']))
  graph=ast.unparse(funcs['_run_trading_agent'])
  self.assertLess(graph.index('claim(get_db, run_id)'),graph.index('TradingAgentsGraph'))

class GraphGuardTests(unittest.TestCase):
 def test_failed_reservation_returns_before_graph_or_report_writes(self):
  from unittest.mock import patch
  tree=ast.parse(Path('app_v3.py').read_text())
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_trading_agent')
  namespace={'get_db':object()}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'guard_fixture','exec'),namespace)
  with patch('agent_execution.claim',return_value=None) as reserved:
   namespace['_run_trading_agent']('saved-id','TEST','2026-10-11','anthropic','fixture')
   reserved.assert_called_once_with(namespace['get_db'],'saved-id')
