import base64,json,unittest,uuid
from flask import Flask
from tests.unit import test_stock_analysis_postgres as fixtures
from pipeline_lab import create_blueprint

@unittest.skipUnless((fixtures.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Imports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StockPostgresTests.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):c.execute('''ALTER TABLE document_files ADD COLUMN id SERIAL, ADD COLUMN mime_type TEXT, ADD COLUMN file_size INTEGER;
        CREATE UNIQUE INDEX doc_name ON document_files(ticker,filename);
        CREATE TABLE app_settings(key TEXT PRIMARY KEY,value TEXT,updated_at TIMESTAMP DEFAULT NOW());
        CREATE TABLE research_categories(id TEXT PRIMARY KEY,name TEXT);
        CREATE TABLE research_documents(id TEXT PRIMARY KEY,category_id TEXT,published_date TEXT);
        CREATE TABLE research_document_files(id SERIAL PRIMARY KEY,document_id TEXT,filename TEXT,file_type TEXT,file_data TEXT);''')
    tearDownClass=classmethod(fixtures.StockPostgresTests.tearDownClass.__func__)
    def setUp(self):
        self.tk='T'+uuid.uuid4().hex[:8].upper();self.key='pipeline_lab_v1:'+self.tk
        a=Flask('lab-import');a.register_blueprint(create_blueprint(self.db));self.client=a.test_client();self.url='/api/pipeline-lab/import-research/'+self.tk
        with self.db(True) as (_,c):
            c.execute('INSERT INTO research_categories VALUES(%s,%s)',(self.tk,self.tk));c.execute('INSERT INTO research_documents VALUES(%s,%s,%s)',(self.tk,self.tk,'2026-10-01'))
            self.refs=[]
            for name in ['a.txt','b.txt']:
                c.execute('INSERT INTO research_document_files(document_id,filename,file_type,file_data) VALUES(%s,%s,%s,%s) RETURNING id',(self.tk,name,'txt',base64.b64encode(name.encode()).decode()));self.refs.append('charlie:research_'+str(c.fetchone()['id']))
            c.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s)',(self.key,json.dumps({'revision':1,'selection':{'thesis':self.refs,'note':[self.refs[0]]}})))
        self.body={'revision':1,'task':'thesis','requestId':str(uuid.uuid4())}
    def test_exact_copy_replay_and_separate_task(self):
        r=self.client.post(self.url,json=self.body);self.assertEqual(r.status_code,200,r.json);self.assertFalse(r.json['researchStarted'])
        self.assertEqual(len(r.json['receipts']),2)
        saved=self.client.get('/api/pipeline-lab/config/'+self.tk).json
        self.assertEqual(saved['revision'],2);self.assertEqual(saved['selection']['note'],[self.refs[0]])
        self.assertTrue(all(x.startswith('charlie:thesis_') for x in saved['selection']['thesis']))
        self.assertTrue(self.client.post(self.url,json=self.body).json['replayed'])
        with self.db() as (_,c):
            c.execute('SELECT COUNT(*) AS n FROM document_files WHERE ticker=%s',(self.tk,));self.assertEqual(c.fetchone()['n'],2)
            c.execute('SELECT COUNT(*) AS n FROM research_document_files WHERE document_id=%s',(self.tk,));self.assertEqual(c.fetchone()['n'],2)
    def test_conflict_rolls_back_entire_import(self):
        with self.db(True) as (_,c):c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s)',(self.tk,'b.txt','Yw==','{}'))
        r=self.client.post(self.url,json=self.body);self.assertEqual(r.status_code,409,r.json)
        self.assertEqual(self.client.get('/api/pipeline-lab/config/'+self.tk).json['revision'],1)
        with self.db() as (_,c):
            c.execute('SELECT filename,file_data FROM document_files WHERE ticker=%s',(self.tk,));rows=c.fetchall();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['file_data'],'Yw==')
    def test_stale_and_cross_stock_selection_do_not_copy(self):
        self.assertEqual(self.client.post(self.url,json={**self.body,'revision':2}).status_code,409)
        with self.db(True) as (_,c):c.execute('UPDATE research_categories SET name=%s WHERE id=%s',('OTHER',self.tk))
        self.assertEqual(self.client.post(self.url,json=self.body).status_code,409)
        with self.db() as (_,c):
            c.execute('SELECT COUNT(*) AS n FROM document_files WHERE ticker=%s',(self.tk,));self.assertEqual(c.fetchone()['n'],0)

    def test_identical_existing_copy_is_reused(self):
        with self.db(True) as (_,c):c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s)',(self.tk,'a.txt','YS50eHQ=','{}'))
        self.assertEqual(self.client.post(self.url,json=self.body).status_code,200)
        with self.db() as (_,c):
            c.execute('SELECT COUNT(*) AS n FROM document_files WHERE ticker=%s',(self.tk,));self.assertEqual(c.fetchone()['n'],2)
        self.assertEqual(self.client.post(self.url,json={**self.body,'task':'note'}).status_code,409)
    def test_reference_only_copy_blocks_without_changing_settings(self):
        with self.db(True) as (_,c):c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s)',(self.tk,'a.txt','YS50eHQ=','{"usage":"reference_only"}'))
        self.assertEqual(self.client.post(self.url,json=self.body).status_code,409)
        self.assertEqual(self.client.get('/api/pipeline-lab/config/'+self.tk).json['revision'],1)
    def test_invalid_bytes_roll_back(self):
        with self.db(True) as (_,c):c.execute('UPDATE research_document_files SET file_data=%s WHERE document_id=%s AND filename=%s',('invalid!',self.tk,'b.txt'))
        self.assertEqual(self.client.post(self.url,json=self.body).status_code,409)
        with self.db() as (_,c):
            c.execute('SELECT COUNT(*) AS n FROM document_files WHERE ticker=%s',(self.tk,));self.assertEqual(c.fetchone()['n'],0)
