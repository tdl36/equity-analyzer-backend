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
    def test_registered_event_import_retains_folder_and_topic(self):
        from collection_refresh import RefreshManager
        m=RefreshManager(self.c)
        (self.c.catalysts/'SYK').mkdir(parents=True)
        m.save(dict(ticker='SYK',workflow='recap',topic='Q3',hours=0,
                           enabled=True,createFolder=True,lookbackDays=7))
        rid=m.trigger('SYK')
        row=self.c.db.execute('SELECT * FROM refresh_requests WHERE id=?',(rid,)).fetchone()
        d=self.c.stage(row['run'],'SYK','transcript',self.f,'https://research.alpha-sense.com/doc/event')['documents'][0]
        handed=self.c.handoff(d['id'])
        post=Mock(side_effect=self.receipt)
        self.assertEqual(sync(self.c,post)[0]['status'],'imported')
        self.assertEqual(post.call_args.args[0]['topic'],'Q3')
        self.assertEqual(Path(handed['destination']).parent,self.c.catalysts/'SYK'/'Q3')
        self.assertEqual(sync(self.c,post),[])

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
        for old in [{'file_data':base64.b64encode(b'other').decode(),'metadata':{}},{'file_data':self.body['fileData'],'metadata':{'usage':'reference_only'}}]:
            c,cur=self.client([old]);self.assertEqual(c.post('/api/agent/collection-original',json=self.body).status_code,409)
            self.assertFalse(any('INSERT' in x.args[0] for x in cur.execute.call_args_list))
    def test_ai_notice_exact_replay_preserves_original_metadata(self):
        old={'file_data':self.body['fileData'],'metadata':{'aiAllowed':False}}
        c,cur=self.client([old]);self.assertEqual(c.post('/api/agent/collection-original',json=self.body).status_code,200)
        self.assertFalse(any(x.args[0].lstrip().startswith(('INSERT','UPDATE')) for x in cur.execute.call_args_list))
    def test_concurrent_insert_does_not_claim_success(self):
        c,_=self.client([None,None]);self.assertEqual(c.post('/api/agent/collection-original',json=self.body).status_code,409)

    def test_explicit_metadata_recovery_preserves_every_nonempty_record(self):
        body={**self.body,'restoreEmptyMetadata':True}
        c,cur=self.client([{'file_data':body['fileData'],'metadata':{}}])
        self.assertEqual(c.post('/api/agent/collection-original',json=body).status_code,200)
        writes=[x for x in cur.execute.call_args_list if x.args[0].startswith('UPDATE')]
        self.assertEqual(len(writes),1)
        recovered=json.loads(writes[0].args[1][0])
        self.assertEqual(recovered['sha256'],body['sha256'])
        for meta,status in [(recovered,200),({'usage':'reference_only'},409),({'custom':'retain'},409)]:
            c,cur=self.client([{'file_data':body['fileData'],'metadata':meta}])
            self.assertEqual(c.post('/api/agent/collection-original',json=body).status_code,status)
            self.assertFalse(any(x.args[0].startswith('UPDATE') for x in cur.execute.call_args_list))
        for old in [None,{'file_data':base64.b64encode(b'changed').decode(),'metadata':{}}]:
            c,cur=self.client([old]);self.assertEqual(c.post('/api/agent/collection-original',json=body).status_code,409)
            self.assertFalse(any(x.args[0].startswith(('UPDATE','INSERT')) for x in cur.execute.call_args_list))

    def test_recovered_metadata_retains_recorded_source_classification_and_provenance(self):
        body={**self.body,'restoreEmptyMetadata':True,'sourceKind':'broker-report',
              'author':'Fixture analyst','authorEvidence':'Observed synthetic byline',
              'topic':'Q3 event','publisher':'Fixture publisher','published':'2026-10-01'}
        c,cur=self.client([{'file_data':body['fileData'],'metadata':{}}])
        self.assertEqual(c.post('/api/agent/collection-original',json=body).status_code,200)
        update=next(x for x in cur.execute.call_args_list if x.args[0].startswith('UPDATE'))
        recovered=json.loads(update.args[1][0])
        for field in ('sourceKind','author','authorEvidence','topic','publisher','published','collectionRun','documentId','sourceUrl','sha256'):
            self.assertEqual(recovered[field],body[field])
        self.assertNotIn('fileData',recovered);self.assertNotIn('restoreEmptyMetadata',recovered)
        # New imports and explicit recovery have the same evidence metadata.
        new={k:v for k,v in body.items() if k!='restoreEmptyMetadata'}
        c,cur=self.client([None,{'id':1}]);self.assertEqual(c.post('/api/agent/collection-original',json=new).status_code,200)
        insert=next(x for x in cur.execute.call_args_list if 'INSERT INTO document_files' in x.args[0])
        self.assertEqual(json.loads(insert.args[1][-1]),recovered)
        # Identical replay is read-only. Different provenance never rewrites it.
        for patch,status in [({},200),({'author':'Another analyst'},409),({'sourceKind':'transcript'},409),({'topic':'Different event'},409)]:
            c,cur=self.client([{'file_data':body['fileData'],'metadata':recovered}])
            self.assertEqual(c.post('/api/agent/collection-original',json={**body,**patch}).status_code,status)
            self.assertFalse(any(x.args[0].lstrip().startswith(('INSERT','UPDATE')) for x in cur.execute.call_args_list))

    def test_metadata_recovery_rejects_unverified_or_invalid_provenance_before_storage(self):
        for patch in ({'author':'Analyst without observed byline'}, {'sourceKind':'invented-type'}, {'topic':'../other-folder'}, {'authorEvidence':'x'*2001}):
            c,cur=self.client([])
            r=c.post('/api/agent/collection-original',json={**self.body,'restoreEmptyMetadata':True,**patch})
            self.assertEqual(r.status_code,400);cur.execute.assert_not_called()

if __name__=='__main__':unittest.main()
