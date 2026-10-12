import ast,json,unittest
from pathlib import Path
from flask import Flask,request,jsonify
from tests.unit import test_stock_analysis_postgres as fixtures

@unittest.skipUnless((fixtures.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class PreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,cur):
            cur.execute('ALTER TABLE document_files ADD COLUMN mime_type TEXT, ADD COLUMN file_size INTEGER; CREATE UNIQUE INDEX doc_identity ON document_files(ticker,filename)')
    @classmethod
    def tearDownClass(cls):fixtures.StockPostgresTests.tearDownClass.__func__(cls)
    def test_insert_only_preserves_bytes_and_restrictions_on_duplicate(self):
        node=next(n for n in ast.parse(Path('app_v3.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='save_documents');node.decorator_list=[]
        scope=dict(request=request,jsonify=jsonify,json=json,get_db=self.db)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'app_v3.py','exec'),scope)
        app=Flask(__name__);app.add_url_rule('/save',view_func=scope['save_documents'],methods=['POST']);client=app.test_client()
        document={'filename':'one.pdf','fileData':'b3JpZ2luYWw=','metadata':{'usage':'reference_only'}}
        payload={'ticker':'SAFE','documents':[document],'preserveExisting':True}
        self.assertEqual(client.post('/save',json=payload).json['savedCount'],1)
        payload['documents']=[{**document,'fileData':'Y2hhbmdlZA==','metadata':{}}]
        self.assertEqual(client.post('/save',json=payload).json['savedCount'],0)
        with self.db() as (_,cur):
            cur.execute("SELECT file_data,metadata FROM document_files WHERE ticker='SAFE'");row=cur.fetchone()
        self.assertEqual(row['file_data'],document['fileData']);self.assertEqual(row['metadata'],document['metadata'])
        self.assertEqual(client.post('/save',json={**payload,'preserveExisting':'true'}).status_code,400)
