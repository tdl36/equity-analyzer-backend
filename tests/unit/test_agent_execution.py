import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from agent_execution import claim, queued_log
from tests.unit import test_stock_analysis_postgres as pg

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class ExecutionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pg.StockPostgresTests.setUpClass.__func__(cls)
  with cls.db(True) as (_,c):
   c.execute('CREATE TABLE agent_runs(id TEXT PRIMARY KEY,status TEXT,logs JSONB,completed_at TIMESTAMP)')
 @classmethod
 def tearDownClass(cls):pg.StockPostgresTests.tearDownClass.__func__(cls)
 def add(self,id,status='queued',logs=None):
  with self.db(True) as (_,c):c.execute('INSERT INTO agent_runs VALUES(%s,%s,%s,NULL)',(id,status,logs if logs is not None else queued_log()))
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
 def test_batch_and_single_enroll_and_graph_reserves_before_import(self):
  tree=ast.parse(Path('app_v3.py').read_text())
  funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
  for name in ['start_agent_run','start_agent_batch_run']:
   self.assertIn('queued_log',ast.unparse(funcs[name]))
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
