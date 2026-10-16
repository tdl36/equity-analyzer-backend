"""Synthetic SQLite and disposable PostgreSQL only; no collection or model calls."""
import json
import sqlite3
import unittest
import uuid
from flask import Flask, request
from tests.unit import test_stock_analysis_postgres as pg
from collection_history import terminal, local_history, lookup, remember, requested_history, PREFIX
from collection_control import create_blueprint


def receipt(ticker='ABT', status='complete'):
    return {'id':str(uuid.uuid4()),'ticker':ticker,'status':status,'created':100.0,
            'result':{'privateSourceText':'must not persist'},'config':{'instructions':'private'}}


class LocalHistory(unittest.TestCase):
    def test_only_bounded_exact_terminal_metadata_is_returned(self):
        db=sqlite3.connect(':memory:');self.addCleanup(db.close);db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE refresh_requests(id TEXT,ticker TEXT,status TEXT,created REAL)')
        values=[receipt(status=s) for s in ('complete','cancelled','collecting','attention')]
        for row in values:db.execute('INSERT INTO refresh_requests VALUES(?,?,?,?)',tuple(row[k] for k in ('id','ticker','status','created')))
        requests=[{'id':r['id'],'ticker':r['ticker']} for r in values]
        result=local_history(db,requests)
        self.assertEqual([r['status'] for r in result],['complete','cancelled','unavailable','unavailable'])
        self.assertEqual(local_history(db,[{**requests[0],'ticker':'CVS'}])[0]['status'],'unavailable')
        self.assertEqual(len(local_history(db,[requests[0]]*60)),50)
        self.assertNotIn('result',result[0]);self.assertTrue(result[0]['receiptOnly'])
        self.assertEqual(db.execute('SELECT COUNT(*) FROM refresh_requests').fetchone()[0],4)

    def test_invalid_receipts_never_become_completion(self):
        base=receipt()
        for patch in ({'id':'bad'},{'ticker':'../ABT'},{'status':'queued'},{'created':True},{'created':float('nan')},{'created':0}):
            self.assertIsNone(terminal({**base,**patch}))
        self.assertEqual(set(terminal(base)),{'id','ticker','status','created','historicalReceipt','receiptOnly'})


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class CloudHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):c.execute('''CREATE TABLE app_settings(key TEXT PRIMARY KEY,value TEXT,updated_at TIMESTAMP DEFAULT NOW());
            CREATE TABLE mp_jobs(id TEXT PRIMARY KEY,stage TEXT,ticker TEXT,status TEXT,input JSONB,result JSONB,error TEXT,created_at TIMESTAMP DEFAULT NOW(),updated_at TIMESTAMP DEFAULT NOW());''')
    tearDownClass=classmethod(pg.StockPostgresTests.tearDownClass.__func__)

    def setUp(self):
        with self.db(True) as (_,c):c.execute('DELETE FROM app_settings; DELETE FROM mp_jobs')
        app=Flask('synthetic-history');app.testing=True
        app.register_blueprint(create_blueprint(self.db,lambda:request.headers.get('X-Agent')=='test'))
        self.client=app.test_client();self.headers={'X-Agent':'test'}

    def command(self,row):
        with self.db(True) as (_,c):c.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input,result) VALUES(%s,'collection_control',%s,'applied','{}',%s::jsonb)",
            (str(uuid.uuid4()),row['ticker'],json.dumps({'refreshRequestId':row['id']})))

    def sync(self,rows=(),history=()):
        response=self.client.post('/api/agent/collection-control',headers=self.headers,json={'snapshot':{'requests':list(rows)},'terminalReceipts':list(history)})
        self.assertEqual(response.status_code,200,response.json)

    def test_eviction_beyond_fifty_and_replay_preserve_exact_status(self):
        rows=[receipt() for _ in range(70)];rows[0]['status']='cancelled'
        self.sync(rows[:35]);self.sync(rows[35:]);self.sync([])
        with self.db() as (_,c):
            self.assertEqual(lookup(c,{},rows[0]['id'],'ABT')['status'],'cancelled')
            self.assertEqual(lookup(c,{},rows[-1]['id'],'ABT')['status'],'complete')
            self.assertIsNone(lookup(c,{},rows[0]['id'],'CVS'))
            c.execute('SELECT COUNT(*) AS n FROM app_settings WHERE key LIKE %s',(PREFIX+'%',));self.assertEqual(c.fetchone()['n'],70)
        self.sync([{**rows[0],'status':'complete'}])
        with self.db() as (_,c):
            self.assertEqual(lookup(c,{},rows[0]['id'],'ABT')['status'],'cancelled')
            c.execute('SELECT value FROM app_settings WHERE key=%s',(PREFIX+rows[0]['id'],));self.assertNotIn('private',c.fetchone()['value'])

    def test_live_snapshot_wins_and_receipt_lookup_never_writes(self):
        row=receipt();self.sync([row])
        active={**row,'status':'collecting'}
        with self.db() as (conn,c):
            c.execute('SET TRANSACTION READ ONLY')
            self.assertEqual(lookup(c,{'requests':[active]},row['id'],'ABT'),active)
            self.assertTrue(lookup(c,{},row['id'],'ABT')['historicalReceipt'])
            self.assertIsNone(lookup(c,{},str(uuid.uuid4()),'ABT'))

    def test_history_backfill_requires_exact_existing_command_and_keeps_jobs(self):
        row=receipt();self.command(row)
        response=self.client.get('/api/agent/collection-control',headers=self.headers)
        self.assertEqual(response.json['historyRequests'],[{'id':row['id'],'ticker':'ABT'}])
        self.sync([], [{**row,'ticker':'CVS'}])
        with self.db() as (_,c):self.assertIsNone(lookup(c,{},row['id'],'ABT'))
        self.sync([], [row]);self.sync([], [row])
        self.assertEqual(self.client.get('/api/agent/collection-control',headers=self.headers).json['historyRequests'],[])
        with self.db() as (_,c):
            self.assertTrue(lookup(c,{},row['id'],'ABT')['receiptOnly'])
            c.execute('SELECT status,result FROM mp_jobs');job=c.fetchone()
            self.assertEqual(job['status'],'applied');self.assertEqual(job['result'],{'refreshRequestId':row['id']})

    def test_unknown_history_rotates_without_fabricating_a_terminal_receipt(self):
        rows=[receipt() for _ in range(51)]
        for row in rows:self.command(row)
        first=self.client.get('/api/agent/collection-control',headers=self.headers).json['historyRequests']
        self.assertEqual(len(first),50)
        self.sync([], [{**r,'status':'unavailable'} for r in first])
        second=self.client.get('/api/agent/collection-control',headers=self.headers).json['historyRequests']
        self.assertNotIn(second[0]['id'],{r['id'] for r in first})
        with self.db() as (_,c):self.assertIsNone(lookup(c,{},first[0]['id'],'ABT'))

    def test_authentication_and_batch_bound_precede_history_writes(self):
        self.assertEqual(self.client.post('/api/agent/collection-control',json={'snapshot':{}}).status_code,403)
        r=self.client.post('/api/agent/collection-control',headers=self.headers,json={'snapshot':{},'terminalReceipts':[receipt()]*51})
        self.assertEqual(r.status_code,400)
        with self.db() as (_,c):c.execute('SELECT COUNT(*) AS n FROM app_settings');self.assertEqual(c.fetchone()['n'],0)
