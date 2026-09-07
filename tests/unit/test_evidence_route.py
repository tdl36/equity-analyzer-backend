"""Exercise the real route with a read-only fake DB, without importing server startup."""
import ast
import re
import unittest
from contextlib import contextmanager
from pathlib import Path
from flask import Flask, jsonify

class EvidenceRouteTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        calls=self.calls
        class Cursor:
            def execute(self,sql,args): calls.append((sql,args))
            def fetchall(self):
                return [{'id':'old','state':{'thesis':['Preserved thesis']}}] if 'investment_reviews' in calls[-1][0] else [{'filename':'report.pdf'}]
            def fetchone(self): return {'company':'Deere','analysis':{'thesis':{'summary':'Saved judgment'}}}
        @contextmanager
        def get_db(): yield None,Cursor()
        app=Flask(__name__)
        tree=ast.parse(Path('app_v3.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='research_evidence_workspace')
        exec(compile(ast.Module(body=[function],type_ignores=[]),'app_v3.py','exec'),{'app':app,'get_db':get_db,'re':re,'jsonify':jsonify,'_local_file_manifest':{}})
        self.client=app.test_client()
    def test_get_reads_two_versions_and_marks_legacy_unchecked(self):
        response=self.client.get('/api/research/evidence/de')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers['Cache-Control'],'no-store')
        self.assertEqual(response.json['current']['quality']['status'],'needs_review')
        self.assertEqual(self.calls[0][1],('DE',))
        self.assertIn('LIMIT 2',self.calls[0][0])
        self.assertTrue(all(sql.lstrip().startswith('SELECT') for sql,args in self.calls))
        self.assertEqual(response.json['savedThesis']['thesis']['summary'],'Saved judgment')
        self.assertEqual(response.json['documents']['uploaded'][0]['filename'],'report.pdf')
    def test_invalid_ticker_never_reaches_database(self):
        self.assertEqual(self.client.get('/api/research/evidence/DE%27').status_code,400)
        self.assertEqual(self.calls,[])
    def test_mutations_not_supported(self):
        self.assertEqual(self.client.post('/api/research/evidence/DE').status_code,405)
        self.assertEqual(self.calls,[])

if __name__=='__main__': unittest.main()
