import unittest
from unittest.mock import Mock, patch
from psycopg2.pool import PoolError
from database_pool import acquire_connection, pooled_cursor

class DatabasePoolTests(unittest.TestCase):
    def test_waits_for_returned_connection_without_increasing_pool(self):
        pool=Mock(); conn=Mock();pool.getconn.side_effect=[PoolError('connection pool exhausted'),conn]
        with patch('database_pool.time.sleep') as sleep:
            self.assertIs(acquire_connection(pool),conn)
            sleep.assert_called_once()
        self.assertEqual(pool.getconn.call_count,2)
    def test_wait_is_bounded(self):
        pool=Mock();pool.getconn.side_effect=PoolError('connection pool exhausted')
        with self.assertRaisesRegex(PoolError,'Database is busy'):
            acquire_connection(pool,timeout=0)
    def test_closed_pool_is_not_retried(self):
        pool=Mock();pool.getconn.side_effect=PoolError('connection pool is closed')
        with self.assertRaisesRegex(PoolError,'closed'):acquire_connection(pool)
        pool.getconn.assert_called_once()
    def setup_pool(self):
        conn=Mock(closed=False);pool=Mock();pool.getconn.return_value=conn
        return pool,conn
    def test_success_returns_clean_connection(self):
        pool,conn=self.setup_pool()
        with pooled_cursor(pool,commit=True) as (actual,cur):self.assertIs(actual,conn)
        conn.commit.assert_called_once();conn.rollback.assert_called_once();conn.cursor.return_value.close.assert_called_once();pool.putconn.assert_called_once_with(conn,close=False)
    def test_cursor_failure_does_not_leak(self):
        pool,conn=self.setup_pool();conn.cursor.side_effect=ValueError('cursor failed')
        with self.assertRaisesRegex(ValueError,'cursor failed'):
            with pooled_cursor(pool):pass
        pool.putconn.assert_called_once_with(conn,close=False)
    def test_query_failure_rolls_back_without_retry(self):
        pool,conn=self.setup_pool()
        with self.assertRaisesRegex(ValueError,'query failed'):
            with pooled_cursor(pool,commit=True):raise ValueError('query failed')
        conn.commit.assert_not_called();conn.rollback.assert_called_once();pool.getconn.assert_called_once();pool.putconn.assert_called_once()
    def test_failed_rollback_discards_connection_preserves_query_error(self):
        pool,conn=self.setup_pool();conn.rollback.side_effect=RuntimeError('broken socket')
        with self.assertRaisesRegex(ValueError,'query failed'):
            with pooled_cursor(pool):raise ValueError('query failed')
        pool.putconn.assert_called_once_with(conn,close=True)

    def test_concurrent_pool_initialization_creates_one_pool(self):
        import ast, threading, time
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor
        tree=ast.parse(Path('app_v3.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_get_pool')
        pool=Mock(closed=False)
        def create(**kwargs):
            time.sleep(.01)
            return pool
        factory=Mock(side_effect=create)
        scope={'_pool':None,'_pool_lock':threading.Lock(),'ThreadedConnectionPool':factory,
               '_get_database_url':lambda:'test-only','RealDictCursor':object,'print':lambda *a:None}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'app_v3.py','exec'),scope)
        with ThreadPoolExecutor(max_workers=8) as workers:
            result=list(workers.map(lambda _:scope['_get_pool'](),range(8)))
        self.assertTrue(all(item is pool for item in result))
        factory.assert_called_once()

if __name__=='__main__':unittest.main()
