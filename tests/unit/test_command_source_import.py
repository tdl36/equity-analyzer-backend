import base64
import hashlib
import json
import unittest
from contextlib import contextmanager
from unittest.mock import Mock
from flask import Flask
import test_research_commands as fixtures
from research_task_sources import collect
from command_source_import import validate, create_blueprint, import_sources

CID='00000000-0000-4000-8000-000000000001'

class ImportRouteTests(unittest.TestCase):
    def setUp(self):
        raw=b'%PDF-1.7\nSynthetic test original'
        self.payload=dict(commandId=CID,ticker='UNH',topic='UNH event',filename='release.pdf',
            fileData=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),
            sourceUrl='https://example.com/verified',usage='research')
        self.command=dict(ticker='UNH',input={'action':'research_task'},result={'topic':'UNH event'})
    def client(self, responses):
        cur=Mock();cur.fetchone.side_effect=responses
        @contextmanager
        def db(commit=False):yield None,cur
        app=Flask(__name__);app.register_blueprint(create_blueprint(db))
        return app.test_client(),cur
    def test_invalid_restricted_and_hash_mismatch_rejected(self):
        for change in ({'usage':'reference_only'},{'filename':'../release.pdf'},{'sha256':'forged'},
                       {'fileData':'bad'},{'commandId':'invalid'},{'filename':'script.exe'}):
            with self.subTest(change=change),self.assertRaises((ValueError,TypeError)):
                validate({**self.payload,**change})
    def test_new_original_receipt_and_metadata(self):
        client,cur=self.client([self.command,None,{'id':1}])
        r=client.post('/api/agent/command-source',json=self.payload)
        self.assertEqual(r.status_code,200);self.assertEqual(r.json['sha256'],self.payload['sha256'])
        call=next(c for c in cur.execute.call_args_list if 'INSERT' in c.args[0])
        self.assertEqual(json.loads(call.args[1][-1])['commandId'],CID)
    def test_identical_replay_does_not_insert(self):
        client,cur=self.client([self.command,{'file_data':self.payload['fileData']}])
        self.assertEqual(client.post('/api/agent/command-source',json=self.payload).status_code,200)
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
    def test_conflicting_original_and_wrong_topic_rejected(self):
        client,cur=self.client([self.command,{'file_data':base64.b64encode(b'older').decode()}])
        self.assertEqual(client.post('/api/agent/command-source',json=self.payload).status_code,409)
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
        client,_=self.client([self.command])
        self.assertEqual(client.post('/api/agent/command-source',json={**self.payload,'topic':'Other event'}).status_code,409)
    def test_concurrent_insert_cannot_claim_success(self):
        client,_=self.client([self.command,None,None])
        self.assertEqual(client.post('/api/agent/command-source',json=self.payload).status_code,409)

class LocalImportTests(unittest.TestCase):
    setUp=fixtures.PublicSourceTests.setUp
    fetch=fixtures.PublicSourceTests.fetch
    def prepare(self):
        collect(self.m,self.claim['id'],self.claim['owner'],self.fetch)
        return self.m.db.execute('SELECT * FROM refresh_requests').fetchone()
    def test_registered_public_sources_only_and_exact_receipt(self):
        row=self.prepare();(self.root/'unregistered.txt').write_text('Do not automatically import')
        self.m.db.execute("INSERT INTO documents(id,run,ticker,kind,filename,sha256,status,usage) VALUES(?,?,?,?,?,?,?,?)",('held',row['run'],'UNH','report','restricted.pdf','hash','held','reference_only'))
        self.m.db.commit()
        calls=[]
        def post(body):
            calls.append(body)
            return dict(imported=True,filename=body['filename'],sha256=body['sha256'])
        self.assertEqual(import_sources(self.m,row,post),2)
        self.assertEqual(len(calls),2);self.assertTrue(all(b['filename'].startswith('SEC_') for b in calls))
        with self.assertRaises(ValueError):import_sources(self.m,row,lambda b:dict(imported=True,filename=b['filename'],sha256='wrong'))
    def test_changed_original_or_cancelled_request_never_uploads(self):
        row=self.prepare();post=Mock()
        manifest=json.loads(self.m.db.execute('SELECT manifest FROM research_public_sources').fetchone()['manifest'])
        (self.root/manifest['documents'][0]['filename']).write_text('changed')
        with self.assertRaises(ValueError):import_sources(self.m,row,post)
        post.assert_not_called()
        self.m.db.execute("UPDATE refresh_requests SET status='cancelled'");self.m.db.commit()
        with self.assertRaises(ValueError):import_sources(self.m,row,post)
        post.assert_not_called()
