import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from flask import Flask
import summary_lab
import working_qa

class WorkingQATests(unittest.TestCase):
    def test_only_additional_section_is_generated_and_reviewed(self):
        calls=[]
        def ask(system,prompt,tokens):
            calls.append(prompt)
            if 'Return JSON only:' in prompt:
                return json.dumps({'record':'Margin target conditional on costs.','passages':['Costs remain uncertain.'],'issues':[]})
            return '**Q: Margin outlook?**\n\n**A:**\n- Target unchanged; costs still uncertain'
        state={'outputMode':'working_qa','editorialPolicy':summary_lab.editorial.POLICY}
        summary_lab.generate('Costs remain uncertain.',state,ask,lambda _:None)
        self.assertEqual(state['completedSections'],['working_qa'])
        self.assertEqual(set(state['sections']),{'working_qa'})
        self.assertIn('working_qa',state['checks'])
        self.assertTrue(any(working_qa.INSTRUCTION in x for x in calls))
        self.assertTrue(state['finalReview'])
        before=len(calls);summary_lab.generate('Costs remain uncertain.',state,ask,lambda _:None)
        self.assertEqual(len(calls),before)
        self.assertNotIn('working_qa',summary_lab.sections_for_mode('english'))

    def test_source_identity_and_export(self):
        self.assertEqual(working_qa.identity('summary','s','one'),working_qa.identity('summary','s','one'))
        self.assertNotEqual(working_qa.identity('summary','s','one'),working_qa.identity('summary','s','two'))
        row=summary_lab.export_row({'title':'Meeting','state':{'sections':{'working_qa':'**Q: Outlook?**\n\n**A:**\n- Could improve\n- <script>bad</script>'},'finalReview':'PRIVATE'}},'working_qa')
        self.assertIn('<li>',row['working_qa']);self.assertNotIn('<script>',row['working_qa'])
        self.assertNotIn('PRIVATE',row['working_qa']);self.assertNotIn('meeting_notes',row)

class RouteTests(unittest.TestCase):
    def setUp(self):
        self.parent={'title':'Synthetic meeting','raw_notes':'Q: Growth? A: Could improve.','source_meta':{}}
        self.rows={};self.inserts=[];owner=self
        class Cursor:
            def execute(self,sql,params=None):
                self.sql=sql;self.params=params
                if 'INSERT INTO summary_lab_experiments' in sql:
                    owner.inserts.append(params)
                    owner.rows[params[0]]={'id':params[0],'title':params[1],'source':params[2],'source_hash':params[3],
                        'state':json.loads(params[8]),'status':'queued','document_ids':json.loads(params[11])}
            def fetchone(self):
                if 'FROM meeting_summaries' in self.sql:return owner.parent
                if 'FROM summary_lab_experiments' in self.sql:return owner.rows.get(self.params[0])
                return None
            def fetchall(self):return []
        @contextmanager
        def db(commit=False):yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(summary_lab.create_blueprint(db,lambda *a,**k:b'PK-test'))
        self.client=app.test_client();self.path='/api/summary-lab/working-qa/summary/s1'
        self.key=patch('summary_lab.provider_credentials.get',return_value='test-key');self.key.start();self.addCleanup(self.key.stop)
        self.thread=patch('summary_lab.threading.Thread');self.worker=self.thread.start();self.addCleanup(self.thread.stop)

    def test_read_has_no_generation_and_duplicate_post_reopens_same_run(self):
        self.assertEqual(self.client.get(self.path).json,{'experiment':None})
        self.worker.assert_not_called()
        a=self.client.post(self.path,json={});b=self.client.post(self.path,json={})
        self.assertEqual(a.status_code,202);self.assertEqual(b.status_code,200)
        self.assertEqual(a.json['experiment']['id'],b.json['experiment']['id'])
        self.assertEqual(len(self.inserts),1);self.worker.assert_called_once()
        self.assertEqual(self.parent['raw_notes'],'Q: Growth? A: Could improve.')
        self.assertEqual(self.inserts[0][10],'working_qa')

    def test_missing_source_or_key_does_not_dispatch(self):
        self.parent['raw_notes']=''
        self.assertEqual(self.client.post(self.path,json={}).status_code,400)
        self.parent=None
        self.assertEqual(self.client.post(self.path,json={}).status_code,404)
        self.worker.assert_not_called()

    def test_lab_retains_original_documents_and_language(self):
        self.rows['p']={'id':'p','title':'Meeting','source':'frozen manifest','status':'complete','output_mode':'korean_only',
            'model':summary_lab.MODEL,'document_ids':['doc1'],'state':{'documents':[{'id':'doc1'}]}}
        r=self.client.post('/api/summary-lab/working-qa/lab/p',json={})
        self.assertEqual(r.status_code,202)
        state=r.json['experiment']['state']
        self.assertEqual(state['documents'],[{'id':'doc1'}]);self.assertEqual(state['qaLanguage'],'korean')
        self.assertEqual(json.loads(self.inserts[0][11]),['doc1'])

    def test_active_parent_does_not_start_parallel_paid_branch(self):
        self.rows['p']={'source':'Q: Growth?','status':'running'}
        r=self.client.post('/api/summary-lab/working-qa/lab/p',json={})
        self.assertEqual(r.status_code,409);self.worker.assert_not_called()

    def test_unfinished_download_is_blocked(self):
        self.rows['p']={'status':'running','state':{'sections':{'working_qa':'draft'}}}
        self.assertEqual(self.client.get('/api/summary-lab/p/working-qa.docx').status_code,409)
        self.rows['p']['status']='complete'
        r=self.client.get('/api/summary-lab/p/working-qa.docx')
        self.assertEqual(r.status_code,200);self.assertEqual(r.data,b'PK-test')
