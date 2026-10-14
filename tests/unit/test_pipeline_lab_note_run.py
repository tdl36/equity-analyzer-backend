import unittest
import base64
import hashlib
from tests.unit import test_pipeline_lab_note_calls as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_plan import schema,freeze
from pipeline_lab_note_sources import prepare
from pipeline_lab_note_run import execute
from pipeline_lab_notes import read


@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Runs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Calls.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):schema(c)
    tearDownClass=classmethod(fixtures.Calls.tearDownClass.__func__)
    def setUp(self):
        fixtures.Calls.setUp(self)
        raw=b'Synthetic source'
        self.pack=[{'reference':'charlie:thesis_1','filename':'fixture.txt','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}]
        self.docs={'charlie:thesis_1':{'filename':'fixture.txt','file_type':'txt','file_data':base64.b64encode(raw).decode(),'metadata':{}}}
        prepared=prepare('ABT',self.pack,self.docs)
        self.plan={'noteId':self.note,'ticker':'ABT','title':'Draft','context':'Frozen context','promptTemplate':'Full-note fixture',
                   'modelKey':'fixture','maxTokens':1000,'sourcePack':self.pack,'representationSha256':prepared['representationSha256'],
                   'batchCount':1,'warnings':prepared['warnings'],'allowPdfText':False}
        with self.db(True) as (_,c):freeze(c,self.plan)
        self.response={**self.response,'text':'===NOTE_START===Synthetic note [1].===NOTE_END==='}
    def test_complete_and_resume_without_repaying_or_reloading_originals(self):
        calls=[]
        def invoke(**kwargs):
            calls.append(kwargs);return self.response
        result=execute(self.db,self.note,lambda plan:self.docs,invoke)
        self.assertEqual(result['status'],'draft')
        self.assertEqual(execute(self.db,self.note,lambda plan:None,invoke),result)
        self.assertEqual(len(calls),1)
        self.assertIn('Frozen context',calls[0]['messages'][0]['content'][-1]['text'])
        with self.db() as (_,c):
            draft=read(c,self.note)
            self.assertTrue(any('embedded images' in issue for issue in draft['qualityIssues']))
            self.assertFalse(draft['investorApproved'])
    def test_provider_failure_stops_and_never_retries(self):
        calls=[]
        def fail(**kwargs):calls.append(True);raise RuntimeError('Synthetic failure')
        for _ in range(2):self.assertEqual(execute(self.db,self.note,lambda plan:self.docs,fail)['status'],'attention')
        self.assertEqual(calls,[True])
    def test_malformed_saved_reply_needs_review_without_more_calls(self):
        calls=[]
        def invoke(**kwargs):calls.append(True);return {**self.response,'text':'Malformed'}
        for _ in range(2):self.assertEqual(execute(self.db,self.note,lambda plan:self.docs,invoke)['reason'],'saved_reply_needs_review')
        self.assertEqual(calls,[True])
    def test_changed_original_blocks_before_provider(self):
        self.docs['charlie:thesis_1']['file_data']='bmV3'
        def forbidden(**kwargs):self.fail('Provider must not run')
        with self.assertRaises(ValueError):execute(self.db,self.note,lambda plan:self.docs,forbidden)
    def test_two_batch_merge_uses_saved_reply_and_stops_if_truncated(self):
        import uuid
        self.note=str(uuid.uuid4())
        self.pack.append({**self.pack[0],'reference':'charlie:thesis_2','filename':'second.txt'})
        self.docs['charlie:thesis_2']={**self.docs['charlie:thesis_1'],'filename':'second.txt'}
        def planner(docs):return [[dict(doc,send_as='file',extracted_text='Synthetic source')] for doc in docs]
        prepared=prepare('ABT',self.pack,self.docs,planner)
        plan={**self.plan,'noteId':self.note,'sourcePack':self.pack,'batchCount':2,
              'representationSha256':prepared['representationSha256'],'warnings':prepared['warnings']}
        with self.db(True) as (_,c):freeze(c,plan)
        calls=[]
        def invoke(**kwargs):
            calls.append(kwargs)
            return {**self.response,'stopReason':'max_tokens'}
        for _ in range(2):
            self.assertEqual(execute(self.db,self.note,lambda p:self.docs,invoke,planner)['reason'],'intermediate_reply_truncated')
        self.assertEqual(len(calls),1)
        self.note=str(uuid.uuid4());plan['noteId']=self.note
        with self.db(True) as (_,c):freeze(c,plan)
        calls.clear()
        def complete(**kwargs):calls.append(kwargs);return self.response
        result=execute(self.db,self.note,lambda p:self.docs,complete,planner)
        self.assertEqual(result['status'],'draft');self.assertEqual(len(calls),2)
        self.assertIn(self.response['text'],calls[1]['messages'][0]['content'][-1]['text'])
        self.assertEqual(execute(self.db,self.note,lambda p:None,complete,planner),result)
        self.assertEqual(len(calls),2)
