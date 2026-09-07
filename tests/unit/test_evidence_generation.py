"""Run the actual review orchestration with stubbed model, rendering and DB boundaries."""
import ast
import json
import unittest
import uuid
import base64
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import investment_review as ir

class EvidenceGenerationTests(unittest.TestCase):
    def run_pipeline(self, readable=True):
        statements='Revenue increased by twelve percent during the fiscal year ended December 2025.'
        docs=[{'filename':'test.txt','send_as':'text','extracted_text':statements if readable else ''}]
        writes=[]; model_calls=[]
        class Cursor:
            def execute(self,sql,args):
                if sql.startswith('INSERT'): writes.append(args)
            def fetchall(self): return docs
            def fetchone(self): return {'company':'Test Co','analysis':{}}
        @contextmanager
        def get_db(**kwargs): yield None,Cursor()
        def model(**kwargs):
            model_calls.append(kwargs)
            if len(model_calls)==1:
                return {'text':json.dumps({'thesis':['Test judgment'],'facts':[]})}
            raise RuntimeError('Synthetic reviewer failure')
        jobs={}
        ns={'investment_review':ir,'get_db':get_db,'_review_jobs':jobs,
            'notegen':SimpleNamespace(prepare_documents=lambda *a,**k:docs,plan_batches=lambda *a,**k:[docs]),
            '_review_prior_state':lambda t:None,'_live_price_context':lambda t:'',
            '_call_pinned_long':model,'_extract_json':json.loads,
            '_review_state_from_dict':lambda t,p:ir.ReviewState(ticker=t,thesis=p['thesis']),
            '_latest_close':lambda t:None,'json':json,'datetime':datetime,'uuid':uuid,'base64':base64}
        tree=ast.parse(Path('app_v3.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_investment_review')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'app_v3.py','exec'),ns)
        with patch.object(ir,'render_pdf',return_value=b''),patch.object(ir,'render_html',return_value=''),patch.object(ir,'render_markdown',return_value=''):
            ns['_run_investment_review']('test','DE','fake-key')
        return jobs,writes,model_calls
    def test_failed_reviewer_is_persisted_as_needs_review_with_evidence(self):
        jobs,writes,calls=self.run_pipeline()
        self.assertEqual(jobs['test']['status'],'complete')
        metadata=json.loads(writes[0][-1])
        self.assertEqual(metadata['evidence']['version'],1)
        self.assertEqual(metadata['readiness']['status'],'needs_review')
        self.assertTrue(any('Independent review' in s for s in metadata['readiness']['issues']))
        self.assertIn('EVIDENCE CONTRACT',str(calls[0]['messages']))
        self.assertIn('EVIDENCE SNAPSHOT',str(calls[1]['messages']))
    def test_unreadable_documents_stop_before_any_model_call(self):
        jobs,writes,calls=self.run_pipeline(False)
        self.assertEqual(jobs['test']['status'],'failed')
        self.assertEqual(writes,[])
        self.assertEqual(calls,[])

if __name__=='__main__': unittest.main()
