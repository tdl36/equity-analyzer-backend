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
    def test_larger_pack_is_processed_in_complete_batches(self):
        pack=[];docs={}
        for i in range(1,26):
            ref=f'charlie:thesis_{i}'
            pack.append({**self.pack[0],'reference':ref,'filename':f'{i}.pdf'})
            docs[ref]={**self.docs['charlie:thesis_1'],'filename':f'{i}.pdf'}
        def planner(rows):
            return [[dict(d,send_as='pages') for d in rows[i:i+5]] for i in range(0,len(rows),5)]
        result=prepare('ABT',pack,docs,planner)
        self.assertEqual(len(result['sourcePack']),25)
        self.assertEqual(len(result['batches']),5)
        with self.assertRaisesRegex(ValueError,'omitted'):
            prepare('ABT',pack,docs,lambda rows:planner(rows[:-1]))

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

    def test_text_truncation_disclosed_without_changing_legacy_representation(self):
        raw=('α'*120_000+'MISSING END').encode()
        pack=[{'reference':'charlie:thesis_1','filename':'long.txt','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}]
        docs={'charlie:thesis_1':{'filename':'long.txt','file_type':'txt','file_data':base64.b64encode(raw).decode(),'metadata':{}}}
        legacy=prepare('ABT',pack,docs)
        current=prepare('ABT',pack,docs,source_disclosure_version=1)
        self.assertEqual(legacy['representationSha256'],current['representationSha256'])
        self.assertEqual(legacy['batches'],current['batches'])
        self.assertNotIn('MISSING END',current['batches'][0][0]['text'])
        self.assertNotIn('omitted',legacy['warnings'][0]['issue'])
        self.assertIn('120000 characters',current['warnings'][0]['issue'])
        self.assertIn('11 extracted characters are omitted',current['warnings'][0]['issue'])
        self.assertEqual(docs['charlie:thesis_1']['file_data'],base64.b64encode(raw).decode())

    def test_spreadsheet_discloses_unread_sheets(self):
        import io
        from openpyxl import Workbook
        workbook=Workbook();sheet=workbook.active
        for _ in range(5):sheet.append(['x'*30_000])
        workbook.create_sheet('Later').append(['END OF WORKBOOK'])
        output=io.BytesIO();workbook.save(output);raw=output.getvalue()
        pack=[{'reference':'charlie:thesis_1','filename':'large.xlsx','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}]
        docs={'charlie:thesis_1':{'filename':'large.xlsx','file_type':'xlsx','file_data':base64.b64encode(raw).decode(),'metadata':{}}}
        result=prepare('ABT',pack,docs,source_disclosure_version=1)
        self.assertIn('Later spreadsheet sheets are omitted',result['warnings'][0]['issue'])
        self.assertNotIn('END OF WORKBOOK',result['batches'][0][0]['text'])

    def test_pdf_middle_extraction_reports_omission_counts(self):
        from unittest.mock import patch
        from notegen import extract_pdf_text
        doc=dict(self.docs['charlie:thesis_1'])
        with patch('pdf_text.extract',return_value={'pages':[{}],'ocrPages':[],'limitations':[]}), patch('pdf_text.render',return_value='x'*100):
            text=extract_pdf_text(doc,max_tokens=10)
        self.assertIn('middle of document omitted',text)
        self.assertEqual(doc['textExtraction']['omittedCharacters'],60)
        self.assertEqual(doc['textExtraction']['retainedCharacters'],40)
        def planner(docs):return [[dict(doc,reference=docs[0]['reference'],send_as='text',extracted_text=text)]]
        current=prepare('ABT',self.pack,self.docs,planner,True,1)
        self.assertIn('60 extracted characters are omitted',current['warnings'][0]['issue'])

    def test_complete_text_has_no_false_truncation_warning(self):
        pack=[{**self.pack[0],'filename':'short.txt'}]
        docs={'charlie:thesis_1':{**self.docs['charlie:thesis_1'],'filename':'short.txt','file_type':'txt'}}
        result=prepare('ABT',pack,docs,source_disclosure_version=1)
        self.assertNotIn('omitted',result['warnings'][0]['issue'])
        for version in (True,2,'1'):
            with self.assertRaisesRegex(ValueError,'disclosure version'):
                prepare('ABT',pack,docs,source_disclosure_version=version)
