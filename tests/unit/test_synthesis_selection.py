import ast
import json
import uuid
import unittest
from pathlib import Path
from contextlib import contextmanager
from flask import Flask, request, jsonify

class SelectionTests(unittest.TestCase):
    def run_route(self, payload, returned=()):
        node = next(n for n in ast.parse(Path('app_v3.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name=='catalyst_proposals_approve')
        node.decorator_list=[]
        calls=[]
        class Cursor:
            def execute(self, sql, args): calls.append((sql,args))
            def fetchall(self): return [{'id':v} for v in returned]
        @contextmanager
        def db(**kwargs): yield None, Cursor()
        scope=dict(get_db=db,request=request,jsonify=jsonify,uuid=uuid)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'app_v3.py','exec'),scope)
        with Flask(__name__).test_request_context(json=payload):
            result=scope[node.name]()
            response,status=result if isinstance(result,tuple) else (result,200)
            return response.json,status,calls
    def test_selection_is_bounded_and_validated_before_write(self):
        for ids in ['not-list',['bad'],['a']*101,[None]]:
            _,status,calls=self.run_route({'jobIds':ids})
            self.assertEqual(status,400);self.assertEqual(calls,[])
    def test_only_still_proposed_synthesis_is_queued_and_acknowledged(self):
        first,second=str(uuid.uuid4()),str(uuid.uuid4())
        body,status,calls=self.run_route({'jobIds':[first,first,second]},[first])
        self.assertEqual(status,200);self.assertEqual(body,{'approved':1,'jobIds':[first]})
        self.assertEqual(calls[0][1],([first,second],))
        self.assertIn("status='proposed' AND job_type='synthesis'",calls[0][0])
        self.assertEqual(calls[1][1],([first],))
    def test_replay_does_not_acknowledge_or_create_another_job(self):
        body,_,calls=self.run_route({'jobIds':[str(uuid.uuid4())]})
        self.assertEqual(body['approved'],0);self.assertEqual(len(calls),1)
        self.assertNotIn('INSERT',calls[0][0])
