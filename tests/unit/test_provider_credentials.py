"""Shared credential control with synthetic provider catalogs; no model calls."""
import unittest
from contextlib import contextmanager
from unittest.mock import patch, Mock
from flask import Flask
import provider_credentials as creds

class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.values={'apiKey':'valid-shared-credential','_provider_preference:anthropic':'validated_saved'}
        outer=self
        class Cursor:
            def execute(self,sql,args):
                self.field=args[0]
                if sql.startswith('INSERT'):outer.values[args[0]]=args[1]
            def fetchone(self):
                return {'value':outer.values[self.field]} if self.field in outer.values else None
        @contextmanager
        def db(commit=False):yield None,Cursor()
        self.db=db;self.old=creds._db
        self.agent=False
        app=Flask(__name__);app.register_blueprint(creds.create_blueprint(db,lambda:self.agent));self.client=app.test_client()
    def tearDown(self):creds._db=self.old
    def test_saved_shared_key_wins_and_rotation_is_live(self):
        with patch.dict('os.environ',{'ANTHROPIC_API_KEY':'stale-environment'}):
            self.assertEqual(creds.get('anthropic'),'valid-shared-credential')
            self.values['apiKey']='replacement'
            self.assertEqual(creds.get('anthropic'),'replacement')
    def test_browser_status_never_returns_secrets_and_agent_gate(self):
        r=self.client.get('/api/provider-credentials')
        self.assertTrue(r.json['providers']['anthropic']['configured'])
        self.assertNotIn('valid-shared-credential',r.text)
        self.assertEqual(r.headers['Cache-Control'],'private, no-store')
        self.assertEqual(self.client.get('/api/agent/provider-credentials/anthropic').status_code,403)
        self.agent=True
        self.assertEqual(self.client.get('/api/agent/provider-credentials/anthropic').json['key'],self.values['apiKey'])
    def test_invalid_or_unavailable_catalog_preserves_working_key(self):
        for code in (401,403,429,None):
            with patch.object(creds,'validate',return_value=code):
                r=self.client.put('/api/provider-credentials/anthropic',json={'key':'invalid-replacement'})
                self.assertEqual(r.status_code,422)
                self.assertEqual(self.values['apiKey'],'valid-shared-credential')
        with patch.object(creds,'validate',return_value=200):
            r=self.client.put('/api/provider-credentials/anthropic',json={'key':'new-valid-credential'})
            self.assertEqual(r.status_code,200)
            self.assertNotIn('new-valid-credential',r.text)
            self.assertEqual(creds.get('anthropic'),'new-valid-credential')
    def test_proxy_uses_only_shared_key_without_retry(self):
        upstream=Mock(status_code=200);upstream.iter_content.return_value=[b'data: test\n\n']
        with patch.object(creds.requests,'post',return_value=upstream) as post:
            r=self.client.post('/api/provider/anthropic/messages',json={'stream':True,'model':'synthetic','apiKey':'stale-phone'})
            self.assertEqual(r.data,b'data: test\n\n')
            self.assertEqual(post.call_args.kwargs['headers']['x-api-key'],'valid-shared-credential')
            post.assert_called_once();upstream.close.assert_called_once()
    def test_legacy_settings_sync_cannot_replace_shared_key(self):
        import ast
        from pathlib import Path
        from flask import request, jsonify
        tree=ast.parse(Path('app_v3.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_settings')
        fn.decorator_list=[]
        cur=Mock()
        @contextmanager
        def db(commit=False):yield None,cur
        env=dict(provider_credentials=creds,request=request,jsonify=jsonify,get_db=db)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'settings','exec'),env)
        app=Flask(__name__)
        with app.test_request_context(json={'apiKey':'stale-phone','geminiApiKey':'stale','openaiApiKey':'stale','_provider_preference:anthropic':'validated_saved','theme':'dark'}):
            self.assertTrue(env['save_settings']().json['success'])
        cur.execute.assert_called_once()
        self.assertEqual(cur.execute.call_args.args[1],('theme','dark'))
    def test_workflow_resolver_ignores_legacy_device_overrides(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path('app_v3.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_get_api_keys')
        env={'provider_credentials':creds}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'keys','exec'),env)
        self.assertEqual(env['_get_api_keys']('stale-phone')['anthropic'],'valid-shared-credential')

    def test_legacy_unvalidated_saved_key_does_not_replace_deployment_key(self):
        self.values['geminiApiKey']='legacy-invalid'
        with patch.dict('os.environ',{'GEMINI_API_KEY':'working-deployment'}):
            self.assertEqual(creds.get('gemini'),'working-deployment')
            with patch.object(creds,'validate',return_value=200):
                self.client.put('/api/provider-credentials/gemini',json={'key':'validated-replacement'})
            self.assertEqual(creds.get('gemini'),'validated-replacement')
