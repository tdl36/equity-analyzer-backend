import unittest
from flask import Flask,request
from tests.unit import test_pipeline_lab_icloud as fixtures
from pipeline_lab_icloud_api import create_blueprint

class ImportHTTP(fixtures.ICloudTransactions):
    def setUp(self):
        super().setUp()
        app=Flask('lab-icloud-api')
        app.register_blueprint(create_blueprint(self.db,lambda:request.headers.get('X-Test-Agent')=='yes'))
        self.client=app.test_client();self.agent={'X-Test-Agent':'yes'}
        self.url='/api/pipeline-lab/icloud-imports/'+self.tk
    def submit(self):
        return self.client.post(self.url,json={'requestId':self.request,'task':'thesis','revision':1})
    def test_user_cannot_freeze_or_upload(self):
        self.assertEqual(self.submit().status_code,200)
        prefix='/api/agent/pipeline-lab-import/'+self.request
        for action,data in [('freeze',{'manifest':self.manifest}),('complete',{'originals':self.bodies})]:
            self.assertEqual(self.client.post(prefix+'/'+action,json=data).status_code,403)
        self.assertEqual(self.client.get('/api/agent/pipeline-lab-imports').status_code,403)
    def test_http_roundtrip_pending_receipt_and_replay(self):
        self.assertEqual(self.submit().status_code,200)
        pending=self.client.get('/api/agent/pipeline-lab-imports',headers=self.agent).json['imports']
        self.assertIn(self.request,[j['requestId'] for j in pending])
        prefix='/api/agent/pipeline-lab-import/'+self.request
        self.assertEqual(self.client.post(prefix+'/freeze',headers=self.agent,json={'manifest':self.manifest}).status_code,200)
        result=self.client.post(prefix+'/complete',headers=self.agent,json={'originals':self.bodies})
        self.assertEqual(result.status_code,200,result.json);self.assertEqual(result.json['status'],'complete')
        self.assertEqual(self.submit().json['status'],'complete')
        status=self.client.get('/api/pipeline-lab/icloud-import/'+self.request)
        self.assertEqual(status.json['status'],'complete');self.assertIn('no-store',status.headers['Cache-Control'])
        self.assertNotIn('fileData',str(status.json))
    def test_invalid_body_and_atomic_conflict(self):
        self.assertEqual(self.client.post(self.url,json={'junk':'x'*5000}).status_code,409)
        self.assertEqual(self.client.get(self.url).json['imports'],[])
        self.submit();prefix='/api/agent/pipeline-lab-import/'+self.request
        self.client.post(prefix+'/freeze',headers=self.agent,json={'manifest':self.manifest})
        result=self.client.post(prefix+'/complete',headers=self.agent,json={'originals':[dict(reference=self.refs[0],fileData='Yw==')]})
        self.assertEqual(result.status_code,409)
        status=self.client.get('/api/pipeline-lab/icloud-import/'+self.request)
        self.assertEqual(status.json['status'],'frozen')
        self.assertEqual(self.client.post('/api/pipeline-lab/icloud-import/'+self.request+'/cancel',json={}).json['status'],'cancelled')
    def test_attention_is_visible_and_retry_keeps_frozen_identity(self):
        self.submit();prefix='/api/agent/pipeline-lab-import/'+self.request
        self.client.post(prefix+'/freeze',headers=self.agent,json={'manifest':self.manifest})
        r=self.client.post(prefix+'/issue',headers=self.agent,json={'code':'source_unavailable'})
        self.assertEqual(r.json['status'],'attention')
        self.assertNotIn(self.request,[r['requestId'] for r in self.client.get('/api/agent/pipeline-lab-imports',headers=self.agent).json['imports']])
        retry=self.client.post('/api/pipeline-lab/icloud-import/'+self.request+'/retry',json={})
        self.assertEqual(retry.json['status'],'frozen');self.assertEqual(retry.json['manifest'],sorted(self.manifest,key=lambda r:r['reference']))
        self.assertNotIn('issue',retry.json)
