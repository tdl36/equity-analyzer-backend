import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Blueprint
from company_research import eligible
from research_source_restriction import register
from tests.unit.test_onepager_corrections import fixture, RID
from tests.unit import test_stock_analysis_postgres as pg


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class RestrictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_, c):
            c.execute('CREATE TABLE stock_analysis_runs(id TEXT PRIMARY KEY,ticker TEXT,input JSONB,sources JSONB,state JSONB)')
    @classmethod
    def tearDownClass(cls):
        pg.StockPostgresTests.tearDownClass.__func__(cls)
    def setUp(self):
        self.run, self.doc, _ = fixture()
        with self.db(True) as (_, c):
            c.execute('DELETE FROM stock_analysis_runs'); c.execute('DELETE FROM document_files')
            c.execute('INSERT INTO stock_analysis_runs VALUES(%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)',
                      (RID,self.run['ticker'],json.dumps(self.run['input']),json.dumps(self.run['sources']),json.dumps(self.run['state'])))
            c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s::jsonb)',
                      (self.run['ticker'],self.doc['filename'],self.doc['file_data'],json.dumps({'usage':'research','retained':'yes'})))
        self.source=self.run['sources'][0]
        self.url=f'/api/research/stock-analysis-run/{RID}/source/{self.source["id"]}/restrict'
        self.body={'sha256':self.source['originalHash'],'reason':'Explicit prohibition observed in synthetic original.','locator':'page 2'}
    def client(self):
        app=Flask(__name__);bp=Blueprint('restriction',__name__)
        register(bp,self.db,'stock_analysis_runs','stock-analysis');app.register_blueprint(bp)
        return app.test_client()
    def test_concurrent_restriction_preserves_bytes_research_and_prior_metadata(self):
        def submit(_):
            r=self.client().post(self.url,json=self.body);self.assertEqual(r.status_code,200);return r.json
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(submit,range(2)))
        self.assertEqual(sum(r['reused'] for r in results),1)
        self.assertTrue(submit(0)['reused'])
        with self.db() as (_,c):
            c.execute('SELECT file_data,metadata FROM document_files');doc=c.fetchone()
            self.assertEqual(doc['file_data'],self.doc['file_data']);self.assertEqual(doc['metadata']['retained'],'yes')
            self.assertEqual(doc['metadata']['usage'],'research');self.assertTrue(eligible(doc['metadata']));self.assertIs(doc['metadata']['aiAllowed'],False)
            self.assertEqual(doc['metadata']['restrictionObservation']['locator'],'page 2')
            c.execute('SELECT state FROM stock_analysis_runs');self.assertEqual(c.fetchone()['state'],self.run['state'])
    def test_invalid_hash_bytes_and_mutation_fields_fail_without_changes(self):
        for body in ({**self.body,'sha256':'0'*64},{**self.body,'aiAllowed':True},{**self.body,'locator':''}):
            self.assertEqual(self.client().post(self.url,json=body).status_code,400)
        with self.db(True) as (_,c):
            c.execute("UPDATE document_files SET file_data='Y2hhbmdlZA=='")
        self.assertEqual(self.client().post(self.url,json=self.body).status_code,400)
        with self.db() as (_,c):
            c.execute('SELECT metadata FROM document_files');self.assertTrue(eligible(c.fetchone()['metadata']))
