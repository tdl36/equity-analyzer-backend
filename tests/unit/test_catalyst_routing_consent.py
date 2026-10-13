"""Run the production routing handler with synthetic DB rows; no paid calls."""
import ast
import json
import uuid
import unittest
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from flask import Flask, request, jsonify


class RoutingConsentTests(unittest.TestCase):
    def invoke(self, linked=True, deferred=False, earnings=True):
        app = Flask(__name__)
        cur = Mock()
        cur.fetchall.return_value = [{'id':'analyst','name':'Analyst',
                                     'auto_mode':{'earnings':earnings}}]
        cur.fetchone.return_value = None
        @contextmanager
        def db(**kwargs):
            yield None, cur
        dispatch = Mock(return_value=({'ok':True},200))
        tree = ast.parse(Path('app_v3.py').read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                  and n.name == 'analysts_queue_catalyst_activity')
        fn.decorator_list = []
        env = dict(request=request,jsonify=jsonify,get_db=db,json=json,uuid=uuid,
                   datetime=datetime,_dispatch_activity_run=dispatch)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'routing','exec'),env)
        body={'ticker':'UNH','topic':'UNH 3Q26 Earnings','fingerprint':'revision',
              'deferAutomaticRun':deferred}
        if linked:
            body['catalystJobId']='existing-folder-job'
        with app.test_request_context(json=body):
            response=env['analysts_queue_catalyst_activity']()
            self.assertNotIsInstance(response,tuple)
            result=response.get_json()
        return dispatch,result,cur

    def test_linked_folder_job_stays_pending_even_with_analyst_auto_on(self):
        dispatch,result,cur=self.invoke()
        dispatch.assert_not_called()
        self.assertEqual(result['createdCount'],1)
        self.assertFalse(result['created'][0]['autoRan'])
        inserts=[c for c in cur.execute.call_args_list if 'INSERT INTO analyst_activities' in c.args[0]]
        self.assertEqual(len(inserts),1)
        self.assertIn("'pending_review'",inserts[0].args[0])

    def test_unlinked_authorized_automatic_routing_still_works(self):
        dispatch,result,_=self.invoke(linked=False)
        dispatch.assert_called_once()
        self.assertTrue(result['created'][0]['autoRan'])

    def test_explicit_deferral_or_disabled_analyst_never_dispatches(self):
        for options in ({'linked':False,'deferred':True},{'linked':False,'earnings':False}):
            dispatch,_,_=self.invoke(**options)
            dispatch.assert_not_called()
