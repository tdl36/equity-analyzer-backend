import ast,json,os,unittest,uuid
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import Mock
from flask import Flask,request,jsonify
class ManualProfileRouteTests(unittest.TestCase):
    def client(self):
        tree=ast.parse(Path('app_v3.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mp_run_pipeline');node.decorator_list=[]
        cur=Mock();cur.fetchone.return_value=None
        @contextmanager
        def db(**kwargs):yield None,cur
        thread=Mock();scope=dict(request=request,jsonify=jsonify,os=os,json=json,uuid=uuid,get_db=db,threading=Mock(Thread=thread),_run_mp_pipeline_job=Mock(),resolve_picker_model=lambda a,b:b,MEETING_PREP_DEFAULT_MODEL='model')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'app_v3.py','exec'),scope)
        app=Flask(__name__);app.add_url_rule('/run',view_func=scope['mp_run_pipeline'],methods=['POST'])
        return app.test_client(),cur,thread
    def test_profile_saved_and_passed_to_worker(self):
        client,cur,thread=self.client();profile={'format':'hosted_pm','audience':'generalist'}
        response=client.post('/run',json={'apiKey':'test-only','meetingId':1,'docs':[{'id':2}],'meetingProfile':profile})
        self.assertEqual(response.status_code,202)
        self.assertEqual(thread.call_args.kwargs['kwargs']['meeting_profile'],profile)
        persisted=json.loads(next(c.args[1][-1] for c in cur.execute.call_args_list if 'INSERT INTO mp_jobs' in c.args[0]))
        self.assertEqual(persisted['meetingProfile'],profile);self.assertNotIn('apiKey',persisted)
    def test_invalid_profile_rejected_before_work(self):
        client,cur,thread=self.client()
        response=client.post('/run',json={'apiKey':'test-only','meetingId':1,'docs':[{'id':2}],'meetingProfile':{'format':'invalid'}})
        self.assertEqual(response.status_code,400);thread.assert_not_called();cur.execute.assert_not_called()
