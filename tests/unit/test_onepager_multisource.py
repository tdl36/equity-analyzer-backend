"""Synthetic multi-original proposals; never provider calls or investor decisions."""
import base64
import copy
import hashlib
import unittest
from onepager_corrections import prepare, preparation
from onepager_edition_contract import baseline, prepare_edition
from onepager_edition_evidence import pending_evidence
from tests.unit.test_onepager_corrections import fixture, CID, RID
from tests.unit.test_onepager_edition_contract import AID, EID


def multiple():
    run,doc,body=fixture()
    text='Second independent synthetic source with qualifying assumptions.'
    sha=hashlib.sha256(text.encode()).hexdigest()
    run['sources'].append({'id':'s2','filename':'second.txt','text':text,'originalHash':sha})
    run['input']['hashes']['second.txt']=sha
    docs={doc['filename']:doc,'second.txt':{'filename':'second.txt','file_data':base64.b64encode(text.encode()).decode(),'metadata':{}}}
    body.update(fingerprint=preparation(run)['fingerprint'],additionalEvidence=[{'sourceId':'s2','excerpt':text,'locator':'Section 2'}])
    return run,docs,body


class MultipleSourceTests(unittest.TestCase):
    def test_two_originals_retained_through_edition_preview(self):
        run,docs,body=multiple();original=copy.deepcopy(run)
        ident,signature,proposal=prepare(run,body,docs)
        assignment={'id':AID,'ticker':'TEST','input':{'ticker':'TEST'},'result':{'reportId':RID,'visualFundamentals':{'price':100}}}
        financials={'price':100};pack=baseline(run,assignment,financials)
        selection={k:v for k,v in pack.items() if k!='ticker'}
        selection.update(requestId=EID,correctionIds=[CID])
        rows=[{'id':ident,'run_id':RID,'fingerprint':body['fingerprint'],'payload_hash':signature,'proposal':proposal}]
        edition=prepare_edition(run,assignment,selection,rows,docs,financials)
        preview=pending_evidence(run,assignment,edition,rows,docs,financials)
        claim=preview.claims[body['path']]
        self.assertEqual([e['sourceId'] for e in claim['evidence']],['s1','s2'])
        self.assertEqual(claim['reviewOrigin'],'pending_editorial')
        self.assertEqual(run,original)
        for change in ['restricted','missing','bytes','tamper']:
            documents=copy.deepcopy(docs);records=copy.deepcopy(rows)
            if change=='restricted':documents['second.txt']['metadata']={'usage':'reference_only'}
            if change=='missing':del documents['second.txt']
            if change=='bytes':documents['second.txt']['file_data']=base64.b64encode(b'changed').decode()
            if change=='tamper':records[0]['proposal']['additionalEvidence'][0]['sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):
                prepare_edition(run,assignment,selection,records,documents,financials)

    def test_every_passage_checked_and_no_forged_review(self):
        run,docs,body=multiple()
        for patch in [{'sourceId':'s1'},{'sourceId':'unrelated'},{'excerpt':'Invented passage absent from the second original.'},{'review':'supported'},{'locator':''},{'sourceId':[]}]:
            changed=copy.deepcopy(body);changed['additionalEvidence'][0].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):prepare(run,changed,docs)
        for extra in [[],None,{},body['additionalEvidence']*8]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):prepare(run,{**body,'additionalEvidence':extra},docs)

    def test_legacy_proposal_shape_is_preserved(self):
        run,doc,body=fixture();_,_,proposal=prepare(run,body,doc)
        self.assertNotIn('additionalEvidence',proposal)
        self.assertEqual(prepare(run,body,{doc['filename']:doc})[2],proposal)

    def test_http_submission_loads_both_originals_and_replay_is_exact(self):
        from contextlib import contextmanager
        from flask import Flask
        from onepager_corrections import create_blueprint
        from tests.unit.test_onepager_corrections import Cursor
        run,docs,body=multiple()
        class MultiCursor(Cursor):
            def execute(self,q,args=()):
                super().execute(q,args)
                if 'FROM document_files' in q:self.row=docs.get(args[1])
        c=MultiCursor();c.run=run
        @contextmanager
        def db(commit=False):yield None,c
        app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client()
        url=f'/api/research/stock-analysis-run/{RID}/corrections'
        self.assertEqual(client.post(url,json=body).status_code,201)
        self.assertEqual(len(c.saved[CID]['proposal']['additionalEvidence']),1)
        self.assertTrue(client.post(url,json=body).json['reused'])
        changed=copy.deepcopy(body);changed['additionalEvidence'][0]['locator']='Section 3'
        self.assertEqual(client.post(url,json=changed).status_code,409)
        changed['requestId']=EID;docs['second.txt']['metadata']={'usage':'reference_only'}
        self.assertEqual(client.post(url,json=changed).status_code,400)
        self.assertEqual(len(c.saved),1)
