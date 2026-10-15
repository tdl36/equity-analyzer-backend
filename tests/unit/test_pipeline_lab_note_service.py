import base64,hashlib,json,unittest,uuid
from unittest.mock import patch
from tests.unit import test_pipeline_lab_note_review as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_service import assignment_note,builder
from pipeline_lab_note_worker import process_one
from pipeline_lab_note_service import documents
from pipeline_lab_notes import read
from pipeline_lab import check_originals
from research_assignment_plan import plan

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Integration(fixtures.Review):
    def test_assignment_pack_to_saved_full_note_and_replay(self):
        ident=str(uuid.uuid4());ticker='FIX'+uuid.uuid4().hex[:6].upper();raw=b'Synthetic source only'
        with self.db(True) as (_,c):
            c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s)',(ticker,'fixture.txt',base64.b64encode(raw).decode(),'{}'))
        pack=[{'filename':'fixture.txt','sha256':hashlib.sha256(raw).hexdigest()}]
        assignment={'ticker':ticker,'thesisBaseline':{'analysis':'Frozen fixture'},'labNotePlan':{'promptTemplate':'Synthetic playbook','modelKey':'fixture','allowPdfText':False}}
        from processing_proposal_note import source_plan
        with self.db() as (_,c):
            c.execute('SELECT id::text AS id,filename,file_data,file_type,metadata FROM document_files WHERE ticker=%s',(ticker,))
            doc=c.fetchone();ref='charlie:thesis_'+doc['id']
            verified=check_originals(ticker,'note',0,[ref],{ref:doc})['verified']
            assignment['labNotePlan']['sourcePreparation']=source_plan(c,ticker,[dict(s,documentId=doc['id']) for s in verified],assignment['labNotePlan'])
        from research_source_context import freeze, GUIDANCE
        assignment['researchPurpose']={'kind':'event_note','event':{'title':'Fixture earnings','date':'2025-03-01'}}
        assignment['processingEvidenceContext']=freeze([dict(s,documentId=doc['id']) for s in verified],{doc['id']:{'role':'supplemental','publishedDate':'2024-03-01','publisher':'Frozen publisher','sourceKind':'transcript','referencedBySavedThesis':False}})
        bad=json.loads(json.dumps(assignment));bad['labNotePlan']['sourcePreparation']['representationSha256']='0'*64
        with self.assertRaisesRegex(ValueError,'representation changed'):
            assignment_note(self.db,ident,bad,pack)
        receipt=assignment_note(self.db,ident,assignment,pack)
        self.assertEqual(receipt['status'],'queued')
        changed=json.loads(json.dumps(assignment));changed['processingEvidenceContext']['sources'][0]['publisher']='Changed caller context'
        self.assertEqual(assignment_note(self.db,ident,changed,pack)['noteId'],receipt['noteId'])
        # Remove the unrelated queued fixture job before claiming this integration job.
        with self.db(True) as (_,c):c.execute('DELETE FROM pipeline_lab_note_jobs WHERE note_id=%s',(self.note,))
        calls=[]
        def originals(p):
            with self.db() as (_,c):return documents(c,p['ticker'],p['sourcePack'])
        def invoke(**kwargs):
            calls.append(kwargs)
            return {'text':'===NOTE_START===Synthetic completed note.===NOTE_END===','model':'fixture','provider':'fixture','stopReason':'end_turn'}
        result=process_one(self.db,originals,invoke)
        self.assertEqual(result['status'],'draft');self.assertEqual(len(calls),1)
        messages=json.dumps(calls[0]);self.assertIn('Frozen publisher',messages);self.assertIn('2024-03-01',messages);self.assertIn(GUIDANCE,messages);self.assertNotIn('Changed caller context',messages)
        self.assertEqual(assignment_note(self.db,ident,assignment,pack)['status'],'draft')
        with self.db() as (_,c):
            draft=read(c,receipt['noteId'])
            self.assertEqual(draft['sourcePack'][0]['sha256'],pack[0]['sha256'])
            self.assertEqual(draft['researchProvenance']['purpose'],assignment['researchPurpose'])
            self.assertEqual(draft['researchProvenance']['sourceEvidence'],assignment['processingEvidenceContext'])
    def test_assignment_changed_bytes_never_enqueue(self):
        ticker='FIX'+uuid.uuid4().hex[:6].upper()
        with self.db(True) as (_,c):c.execute('INSERT INTO document_files(ticker,filename,file_data,metadata) VALUES(%s,%s,%s,%s)',(ticker,'fixture.txt','YQ==','{}'))
        assignment={'ticker':ticker,'thesisBaseline':{},'labNotePlan':{'promptTemplate':'Fixture','modelKey':'fixture','allowPdfText':False}}
        with self.assertRaises(ValueError):assignment_note(self.db,str(uuid.uuid4()),assignment,[{'filename':'fixture.txt','sha256':'a'*64}])
    def test_only_explicit_bounded_note_option(self):
        data={'ticker':'ABT','since':'2026-01-01','until':'2026-01-02','outputs':['summary_note']}
        self.assertNotIn('labFullNote',plan(data))
        self.assertEqual(plan({**data,'labFullNote':{'allowPdfText':False}})['labFullNote'],{'allowPdfText':False})
        for choice in [True,{}, {'allowPdfText':1},{'allowPdfText':False,'model':'other'}]:
            with self.assertRaises(ValueError):plan({**data,'labFullNote':choice})
