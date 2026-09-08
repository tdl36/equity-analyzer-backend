import unittest
from contextlib import contextmanager
from datetime import datetime
from flask import Flask
from research_history import records,timestamp,create_blueprint


class HistoryTests(unittest.TestCase):
    def test_dates_are_utc_and_unknown_is_not_now(self):
        self.assertEqual(timestamp(datetime(2026,9,8)),'2026-09-08T00:00:00+00:00')
        self.assertEqual(timestamp(0),'1970-01-01T00:00:00+00:00')
        self.assertIsNone(timestamp('bad'));self.assertIsNone(timestamp(None))

    def test_latest_records_keep_distinct_identity_and_explicit_links(self):
        data=records({'source':[{'id':1,'ticker':'DE','created_at':'2026-09-07'}],
            'note':[{'id':1,'ticker':'DE','created_at':'2026-09-08','updated_at':'2026-09-09','parent_id':'original','proposal_id':'proposal','status':'draft'}]})
        self.assertEqual([r['id'] for r in data],['note:1','source:1'])
        self.assertEqual(data[0]['parentId'],'original');self.assertIsNone(data[1]['parentId'])
        self.assertEqual(data[0]['status'],'draft')

    def test_limit_preserves_latest_creation_not_old_updated_records(self):
        data=records({'note':[{'id':i,'created_at':i} for i in range(120)]})
        self.assertEqual(len(data),100);self.assertEqual(data[0]['recordId'],'119')

    def test_endpoint_uses_bound_ticker_and_returns_collection_snapshot(self):
        calls=[]
        class Cursor:
            def execute(self,sql,args=()):calls.append((sql,args))
            def fetchall(self):return []
            def fetchone(self):return {'updated_at':'2026-09-08','value':{'requests':[{'id':'one','ticker':'DE','created':1,'status':'complete'},{'id':'two','ticker':'AMT','created':2}]}}
        @contextmanager
        def db():yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client()
        r=client.get('/api/research/workspace-history?ticker=de')
        self.assertEqual(r.status_code,200);self.assertEqual(r.json['records'][0]['ticker'],'DE')
        self.assertEqual(len(r.json['records']),1);self.assertEqual(r.headers['Cache-Control'],'no-store')
        self.assertTrue(all(args==('DE',) for sql,args in calls[:-1]))
        before=len(calls);self.assertEqual(client.get('/api/research/workspace-history?ticker=DE%27').status_code,400);self.assertEqual(len(calls),before)
