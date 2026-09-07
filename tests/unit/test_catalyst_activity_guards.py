"""Execute production route functions with a fake DB; never import app startup."""
import ast
from pathlib import Path
from contextlib import contextmanager
import unittest
import json
from datetime import datetime
from flask import Flask, request, jsonify


def route(name, row):
    node = next(n for n in ast.parse(Path('app_v3.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    calls = []
    class Cursor:
        def execute(self, sql, args): calls.append(sql)
        def fetchone(self): return row
    @contextmanager
    def db(**kw): yield None, Cursor()
    scope = {'get_db': db, 'request': request, 'jsonify': jsonify, 'json':json, 'datetime':datetime, '_dispatch_activity_run':lambda *a,**kw: ({'ok':True,'instructions':kw['custom_instructions']},200)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'app_v3.py', 'exec'), scope)
    return scope[name], calls


class ActivityGuardTests(unittest.TestCase):
    def test_regenerate_does_not_erase_a_running_recap(self):
        fn, calls = route('analyst_activities_regenerate', {'status': 'running', 'output': {'synthesisMarkdown': 'prior'}})
        with Flask(__name__).test_request_context(json={}):
            response, code = fn('activity')
        self.assertEqual(code, 409)
        self.assertFalse(any(q.strip().startswith('UPDATE') for q in calls))

    def test_approval_does_not_save_an_empty_recap(self):
        fn, calls = route('analyst_activities_approve', {'ticker': 'MDT', 'activity_type': 'earnings_recap', 'status': 'pending_review', 'input': {'topic':'event'}, 'output': {}})
        with Flask(__name__).test_request_context(json={}):
            response, code = fn('activity')
        self.assertEqual(code, 409)
        self.assertFalse(any('INSERT' in q or 'UPDATE' in q for q in calls))

    def test_revision_passes_instructions_and_prior_draft_to_analyst(self):
        fn,calls=route('analyst_activities_regenerate', {'status':'pending_review','output':{'synthesisMarkdown':'Existing draft'}})
        with Flask(__name__).test_request_context(json={'customInstructions':'Correct the guidance period'}):
            response,code=fn('activity')
        self.assertEqual(code,200)
        self.assertIn('Correct the guidance period',response.json['instructions'])
        self.assertIn('Existing draft',response.json['instructions'])
        self.assertTrue(any('UPDATE' in q for q in calls))

    def test_invalid_revision_instruction_never_writes(self):
        fn,calls=route('analyst_activities_regenerate', {'status':'pending_review','output':{}})
        with Flask(__name__).test_request_context(json={'customInstructions':'x'*6001}):
            response,code=fn('activity')
        self.assertEqual(code,400);self.assertEqual(calls,[])
