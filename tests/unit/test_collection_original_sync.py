"""Synthetic originals, temporary SQLite and mocked cloud calls only."""
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from flask import Flask
from charlie_collector import Collector
from collection_original_sync import sync,enqueue,snapshot,create_blueprint,validate
from test_charlie_collector import pdf


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        stocks=self.root/'STOCKS';(stocks/'SYK').mkdir(parents=True)
        self.c=Collector(self.root/'state',stocks)
        self.run=self.c.create(['SYK'],'2026-08-01','2026-10-09')['id']
        self.f=self.root/'original.pdf';self.f.write_bytes(pdf())
        self.doc=self.c.stage(self.run,'SYK','transcript',self.f,'https://research.alpha-sense.com/doc/test')['documents'][0]
    def tearDown(self):
        self.c.db.close();self.tmp.cleanup()
    def receipt(self,body):return dict(imported=True,filename=body['filename'],sha256=body['sha256'])
    def test_handoff_queues_and_replay_is_exactly_once(self):
        self.c.handoff(self.doc['id']);self.c.handoff(self.doc['id'])
        post=Mock(side_effect=self.receipt)
        self.assertEqual(sync(self.c,post)[0]['status'],'imported')
        self.assertEqual(sync(self.c,post),[]);self.assertEqual(post.call_count,1)
        self.assertEqual(snapshot(self.c)[0]['status'],'imported')
    def test_unknown_network_outcome_retry_and_restart_preserve_bytes(self):
        self.c.handoff(self.doc['id']);post=Mock(side_effect=TimeoutError('secret provider response'))
        self.assertEqual(sync(self.c,post,clock=lambda:100)[0]['status'],'attention')
        self.assertNotIn('secret',snapshot(self.c)[0]['issue'])
        self.assertEqual(sync(self.c,post,clock=lambda:110),[])
        state,stocks=self.c.state,self.c.stocks;self.c.db.close();self.c=Collector(state,stocks)
        post=Mock(side_effect=self.receipt)
        self.assertEqual(sync(self.c,post,clock=lambda:200)[0]['status'],'imported')
    def test_interrupted_lease_and_wrong_receipt(self):
        self.c.handoff(self.doc['id'])
        self.c.db.execute("UPDATE original_imports SET status='uploading',lease_until=200,owner='old'");self.c.db.commit()
        post=Mock(return_value=dict(imported=True,filename='wrong',sha256='wrong'))
        self.assertEqual(sync(self.c,post,clock=lambda:199),[])
        self.assertEqual(sync(self.c,post,clock=lambda:201)[0]['status'],'attention')
    def test_changed_and_restricted_original_never_uploaded(self):
        d=self.c.handoff(self.doc['id']);Path(d['destination']).write_bytes(b'changed')
        post=Mock();self.assertEqual(sync(self.c,post)[0]['status'],'attention');post.assert_not_called()
        self.c.db.execute("UPDATE documents SET usage='reference_only'");self.c.db.execute('UPDATE original_imports SET next_attempt=0');self.c.db.commit()
        sync(self.c,post);post.assert_not_called()
    def test_no_automatic_backfill_or_event_import(self):
        self.assertEqual(snapshot(self.c),[])
        topic='Q3';(self.c.catalysts/'SYK'/topic).mkdir(parents=True)
        run=self.c.create(['SYK'],'2026-08-01','2026-10-09',topic)['id']
        d=self.c.stage(run,'SYK','transcript',self.f,'https://research.alpha-sense.com/doc/event')['documents'][0]
        self.c.handoff(d['id']);self.assertEqual(snapshot(self.c),[])
    def test_duplicate_local_original_is_importable(self):
        (self.c.stocks/'SYK'/'prior.pdf').write_bytes(self.f.read_bytes())
        self.assertEqual(self.c.handoff(self.doc['id'])['status'],'duplicate')
        self.assertEqual(sync(self.c,self.receipt)[0]['status'],'imported')


class EndpointTests(unittest.TestCase):
    def setUp(self):
        raw=pdf();self.body=dict(ticker='SYK',collectionRun='a'*12,documentId='b'*16,
            filename='original.pdf',fileData=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),
            sourceUrl='https://research.alpha-sense.com/doc/test',usage='research')
    def client(self,results,auth=True):
        cur=Mock();cur.fetchone.side_effect=results
        @contextmanager
        def db(commit=False):yield None,cur
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,lambda:auth))
        return app.test_client(),cur
    def test_auth_invalid_hash_restriction_and_source(self):
        c,cur=self.client([],False);self.assertEqual(c.post('/api/agent/collection-original',json=self.body).status_code,403);cur.execute.assert_not_called()
        for patch in [{'usage':'reference_only'},{'sha256':'wrong'},{'filename':'../bad.pdf'},{'sourceUrl':'https://example.com/x'},{'ticker':'../SYK'},{'documentId':'wrong'}]:
            with self.subTest(patch=patch),self.assertRaises((ValueError,TypeError)):validate({**self.body,**patch})
    def test_new_and_exact_replay(self):
        c,cur=self.client([None,{'id':1}]);r=c.post('/api/agent/collection-original',json=self.body)
        self.assertEqual(r.status_code,200);self.assertEqual(r.json['sha256'],self.body['sha256'])
        c,cur=self.client([{'file_data':self.body['fileData'],'metadata':{}}]);self.assertEqual(c.post('/api/agent/collection-original',json=self.body).status_code,200)
        self.assertFalse(any('INSERT' in x.args[0] for x in cur.execute.call_args_list))
    def test_conflict_or_existing_restriction_is_preserved(self):
        for old in [{'file_data':base64.b64encode(b'other').decode(),'metadata':{}},{'file_data':self.body['fileData'],'metadata':{'aiAllowed':False}}]:
            c,cur=self.client([old]);self.assertEqual(c.post('/api/agent/collection-original',json=self.body).status_code,409)
            self.assertFalse(any('INSERT' in x.args[0] for x in cur.execute.call_args_list))
    def test_concurrent_insert_does_not_claim_success(self):
        c,_=self.client([None,None]);self.assertEqual(c.post('/api/agent/collection-original',json=self.body).status_code,409)

if __name__=='__main__':unittest.main()
