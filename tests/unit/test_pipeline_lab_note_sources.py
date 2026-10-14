import base64
import hashlib
import unittest
from pipeline_lab_note_sources import prepare


class Sources(unittest.TestCase):
    def setUp(self):
        raw=b'fixture bytes'
        self.pack=[{'reference':'charlie:thesis_1','filename':'fixture.pdf',
                    'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}]
        self.docs={'charlie:thesis_1':{'filename':'fixture.pdf','file_type':'pdf',
                                     'file_data':base64.b64encode(raw).decode(),'metadata':{}}}
    def plan(self,docs):return [[dict(d,send_as='pages') for d in docs]]
    def test_pdf_uses_exact_original_and_stable_representation(self):
        result=prepare('ABT',self.pack,self.docs,self.plan)
        self.assertEqual(result['batches'][0][0]['source']['data'],self.docs['charlie:thesis_1']['file_data'])
        self.assertEqual(result,prepare('ABT',self.pack,self.docs,self.plan))
        self.assertEqual(result['warnings'],[])
    def test_changed_bytes_missing_source_and_skipped_plan_block(self):
        changed={'charlie:thesis_1':{**self.docs['charlie:thesis_1'],'file_data':'bmV3'}}
        for docs in ({},changed):
            with self.assertRaises(ValueError):prepare('ABT',self.pack,docs,self.plan)
        for planner in (lambda docs:[],lambda docs:[[]],lambda docs:[[dict(docs[0],send_as='skip')]],
                        lambda docs:[[dict(docs[0],send_as='pages')]*2]):
            with self.assertRaises(ValueError):prepare('ABT',self.pack,self.docs,planner)
    def test_text_conversion_requires_explicit_policy_and_retains_warning(self):
        def planner(docs):return [[dict(docs[0],send_as='text',extracted_text='Synthetic text')]]
        with self.assertRaisesRegex(ValueError,'text-only conversion'):prepare('ABT',self.pack,self.docs,planner)
        result=prepare('ABT',self.pack,self.docs,planner,allow_pdf_text=True)
        self.assertEqual(len(result['warnings']),1)
        other=prepare('ABT',self.pack,self.docs,lambda docs:[[dict(docs[0],send_as='text',extracted_text='Changed')]],allow_pdf_text=True)
        self.assertNotEqual(result['representationSha256'],other['representationSha256'])
    def test_original_substitution_by_planner_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'changed a selected original'):
            prepare('ABT',self.pack,self.docs,lambda docs:[[dict(docs[0],send_as='pages',file_data='b3RoZXI=')]])
    def test_existing_planner_retains_reference_for_readable_text(self):
        pack=[{**self.pack[0],'filename':'fixture.txt'}]
        docs={'charlie:thesis_1':{**self.docs['charlie:thesis_1'],'filename':'fixture.txt','file_type':'txt'}}
        result=prepare('ABT',pack,docs)
        self.assertIn('fixture bytes',result['batches'][0][0]['text'])
        self.assertIn('charlie:thesis_1',result['batches'][0][0]['text'])
        self.assertEqual(len(result['warnings']),1)
