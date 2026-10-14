"""Archive navigation on disposable PostgreSQL; never wakes research."""
import unittest
import uuid
from unittest.mock import Mock
from tests.unit import test_research_assignments as fixtures
from tests.unit import test_stock_analysis_postgres as pg


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.AssignmentPostgresTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):fixtures.AssignmentPostgresTests.tearDownClass.__func__(cls)
    setUp=fixtures.AssignmentPostgresTests.setUp
    invoke=fixtures.AssignmentPostgresTests.invoke

    def seed(self,count=65,ticker=None):
        ids=[str(uuid.uuid4()) for _ in range(count)]
        with self.db(True) as (_,cur):
            for ident in ids:
                cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result,created_at) VALUES(%s,'research_assignment',%s,'complete','{}','{\"private\":\"body\"}','2025-01-01')",(ident,ticker or self.ticker))
        return ids

    def get(self,cursor=None,ticker=None):
        query={'ticker':ticker or self.ticker}
        if cursor is not None:query['cursor']=cursor
        return self.client.get('/api/research/assignment-history',query_string=query)

    def test_older_assignments_paginate_without_duplicates_or_body_exposure(self):
        ids=self.seed();self.co.wake=Mock(side_effect=AssertionError('Archive must not wake'))
        found=[];cursor=None
        for size in (30,30,5):
            r=self.get(cursor);self.assertEqual(r.status_code,200,r.json)
            self.assertEqual(r.headers['Cache-Control'],'private, no-store')
            self.assertEqual(len(r.json['assignments']),size)
            found.extend(row['id'] for row in r.json['assignments']);cursor=r.json['nextCursor']
            self.assertNotIn('private',r.text);self.assertTrue(r.json['readOnly'])
        self.assertIsNone(cursor);self.assertEqual(found,sorted(ids,reverse=True))
        self.co.wake.assert_not_called()

    def test_new_insert_between_pages_does_not_shift_old_page(self):
        ids=self.seed(32);first=self.get().json
        self.seed(1)
        with self.db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET created_at='2026-01-01' WHERE ticker=%s AND id<>ALL(%s)",(self.ticker,ids))
        second=self.get(first['nextCursor']).json
        self.assertEqual([r['id'] for r in second['assignments']],sorted(ids,reverse=True)[30:])

    def test_filters_cursor_scope_and_malformed_inputs(self):
        self.seed(31);cursor=self.get().json['nextCursor']
        self.assertEqual(self.get(cursor,'OTHER').status_code,400)
        self.assertEqual(self.get(ticker='OTHER').json['assignments'],[])
        for value in ('!','x'*513,'W10='):
            self.assertEqual(self.get(value).status_code,400)
        self.assertEqual(self.client.get('/api/research/assignment-history?ticker=A&ticker=B').status_code,400)
        self.assertEqual(self.client.post('/api/research/assignment-history').status_code,405)
