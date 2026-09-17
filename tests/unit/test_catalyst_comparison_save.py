"""Route contract tests with a fake cursor; never import live application startup."""
import ast
import json
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from flask import Flask, request, jsonify

class SaveTests(unittest.TestCase):
    def call(self, variant='', comparison=True, editorial=False):
        node=next(n for n in ast.parse(Path('app_v3.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='save_catalyst_to_docs')
        node.decorator_list=[]
        calls=[]
        class Cursor:
            def execute(self,sql,args):calls.append((sql,args))
            def fetchone(self):return {'ticker':'MMM','status':'complete','steps_detail':{'topic':'Conference'},'result':{'markdown':'original','catalystComparison':({'markdown':'private audit','editorialVersion':1,'shareMarkdown':'clean detailed note' if comparison else ''} if editorial else {'markdown':'improved'} if comparison else {})}}
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        scope=dict(get_db=db,request=request,jsonify=jsonify,json=json,cache=SimpleNamespace(invalidate=lambda *_:None))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'app_v3.py','exec'),scope)
        with Flask(__name__).test_request_context('/?variant='+variant,method='POST'):
            result=scope['save_catalyst_to_docs']('job')
        return result,calls
    def test_improved_save_is_separate(self):
        response,calls=self.call('improved')
        self.assertEqual(response.json['docId'],'catalyst-job-improved')
        insert=next(args for sql,args in calls if 'INSERT INTO research_documents' in sql)
        self.assertEqual(insert[-1],'improved');self.assertIn('Improved catalyst',insert[-2])
    def test_editorial_save_excludes_audit(self):
        response,calls=self.call('improved',True,True)
        self.assertEqual(response.json['docId'],'catalyst-job-improved')
        self.assertEqual(next(args for sql,args in calls if 'INSERT INTO research_documents' in sql)[-1],'clean detailed note')
    def test_failed_editorial_cannot_export_audit_as_note(self):
        response,calls=self.call('improved',False,True)
        self.assertEqual(response[1],409)
        self.assertFalse(any('INSERT' in sql for sql,args in calls))
    def test_original_save_unchanged(self):
        response,calls=self.call()
        self.assertEqual(response.json['docId'],'catalyst-job')
        self.assertEqual(next(args for sql,args in calls if 'INSERT INTO research_documents' in sql)[-1],'original')
    def test_missing_improved_never_saves_original_under_trial_name(self):
        response,calls=self.call('improved',False)
        self.assertEqual(response[1],409)
        self.assertFalse(any('INSERT' in sql for sql,args in calls))

class UploadedCatalystTests(unittest.TestCase):
    def test_long_upload_reaches_parallel_trial_in_full(self):
        from unittest.mock import patch
        node=next(n for n in ast.parse(Path('app_v3.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_run_catalyst_synthesis_backend')
        node.decorator_list=[]
        updates=[];prompts=[];received=[]
        class Cursor:
            def execute(self,sql,args):updates.append((sql,args))
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        def llm(**kwargs):prompts.append(kwargs['messages'][0]['content']);return {'text':'<p>Legacy recap</p>'}
        def compare(parts,*args,**kwargs):received.extend(parts);return {'status':'ready','markdown':'improved'}
        scope=dict(get_db=db,json=json,call_llm=llm,CATALYST_LENGTH_PRESETS={'standard':{'instruction':'standard'}},
                   CATALYST_SYNTHESIS_PROMPT='{source_content}',_catalyst_comparison_baseline=lambda *a:'',
                   _generate_note_docx=lambda *a:'docx')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'app_v3.py','exec'),scope)
        transcript='BEGIN '+('Long transcript text. '*5000)+' END'
        with patch('recap_validation.audit',return_value={}),patch('catalyst_comparison.run_safely',side_effect=compare):
            scope['_run_catalyst_synthesis_backend']('job','MMM',{'uploadedFiles':[{'name':'transcript.txt','text':transcript}]})
        self.assertEqual(received[0]['content'],transcript)
        self.assertGreater(len(prompts),1)
        self.assertTrue(any('END' in p for p in prompts))
        completed=[args for sql,args in updates if 'result=%s' in sql]
        self.assertEqual(completed[-1][3],'complete')
        self.assertEqual(json.loads(completed[-1][2])['catalystComparison']['markdown'],'improved')

if __name__=='__main__':unittest.main()
