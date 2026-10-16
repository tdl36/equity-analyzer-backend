import unittest
from datetime import datetime,timezone,date
from meeting_agenda import upcoming
from tests.unit import test_stock_analysis_postgres as pg_tests
class AgendaTests(unittest.TestCase):
    def test_absent_table_is_unavailable_not_an_empty_agenda(self):
        class Cursor:
            def execute(self,sql,args=()):self.sql=sql
            def fetchone(self):return {'name':None}
        with self.assertRaises(RuntimeError):upcoming(Cursor(),datetime(2026,10,11,1,tzinfo=timezone.utc))
@unittest.skipUnless((pg_tests.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class AgendaPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg_tests.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):
            c.execute('''CREATE TABLE mp_meetings(id SERIAL PRIMARY KEY,company_id INTEGER,meeting_date DATE,meeting_type TEXT,status TEXT);
            CREATE TABLE mp_question_sets(id SERIAL PRIMARY KEY,meeting_id INTEGER,status TEXT);
            INSERT INTO mp_companies(ticker,name) VALUES('SYNTH','Synthetic');
            INSERT INTO mp_meetings(company_id,meeting_date,meeting_type,status) VALUES
             (1,'2026-10-10','management','draft'),(1,'2026-10-16','expert','draft'),
             (1,'2026-10-09','other','draft'),(1,'2026-10-17','other','draft'),
             (1,NULL,'other','draft'),(1,'2026-10-10','other','cancelled');
            INSERT INTO mp_question_sets(meeting_id,status) VALUES(1,'ready'),(1,'ready'),(2,'failed');''')
    @classmethod
    def tearDownClass(cls):pg_tests.StockPostgresTests.tearDownClass.__func__(cls)
    def test_bounded_date_selection_no_duplicates_or_mutation(self):
        with self.db() as (_,c):
            d=upcoming(c,datetime(2026,10,11,1,tzinfo=timezone.utc))
            self.assertEqual([r['id'] for r in d['items']],[1,2]);self.assertTrue(d['items'][0]['prepared']);self.assertFalse(d['items'][1]['prepared']);self.assertFalse(d['truncated'])
            c.execute('SELECT COUNT(*) AS n FROM mp_meetings');self.assertEqual(c.fetchone()['n'],6)
    def test_limit_disclosed(self):
        with self.db() as (_,c):
            c.execute("INSERT INTO mp_meetings(company_id,meeting_date,status) SELECT 1,'2026-10-12','draft' FROM generate_series(1,60)")
            d=upcoming(c,datetime(2026,10,11,1,tzinfo=timezone.utc));self.assertEqual(len(d['items']),50);self.assertTrue(d['truncated'])
