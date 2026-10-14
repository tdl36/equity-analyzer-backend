import copy,unittest,uuid
from tests.unit import test_pipeline_lab_import as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_notes import schema,append,obj

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class DraftVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Imports.setUpClass.__func__(cls)
        with cls.db(True) as (_,c):schema(c)
    tearDownClass=classmethod(fixtures.Imports.tearDownClass.__func__)
    def setUp(self):
        self.note=str(uuid.uuid4());self.event=str(uuid.uuid4())
        self.payload={'title':'Event draft','markdown':'Supported statement [1].','sourcePack':[{'reference':'charlie:thesis_1','filename':'report.pdf','sha256':'a'*64,'bytes':10}],'qualityIssues':['Investor review pending.']}
    def save(self,expected=0,event=None,payload=None,ticker='ABT'):
        with self.db(True) as (_,c):return append(c,self.note,ticker,event or self.event,expected,payload or self.payload)
    def test_append_history_and_replay_preserve_all_versions(self):
        self.assertEqual(self.save()['revision'],1)
        updated={**self.payload,'markdown':'Edited interpretation [1].'}
        self.assertEqual(self.save(1,str(uuid.uuid4()),updated)['revision'],2)
        self.assertTrue(self.save()['replayed'])
        with self.db() as (_,c):
            c.execute('SELECT payload FROM pipeline_lab_note_versions WHERE note_id=%s ORDER BY revision',(self.note,));rows=c.fetchall()
            self.assertEqual(len(rows),2);self.assertEqual(obj(rows[0]['payload'])['markdown'],self.payload['markdown'])
            self.assertTrue(all(obj(r['payload'])['status']=='draft' and obj(r['payload'])['investorApproved'] is False for r in rows))
    def test_stale_edit_and_reused_event_do_not_overwrite(self):
        self.save()
        with self.assertRaises(ValueError):self.save(0,str(uuid.uuid4()))
        with self.assertRaises(ValueError):self.save(payload={**self.payload,'markdown':'Other'})
        with self.assertRaises(ValueError):self.save(1,str(uuid.uuid4()),ticker='UNH')
    def test_source_pack_cannot_change_during_edit(self):
        self.save();p=copy.deepcopy(self.payload);p['sourcePack'][0]['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'source pack'):self.save(1,str(uuid.uuid4()),p)
    def test_parallel_edits_have_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        self.save();barrier=Barrier(2)
        def edit(n):
            barrier.wait()
            try:return self.save(1,str(uuid.uuid4()),{**self.payload,'markdown':str(n)})['revision']
            except ValueError:return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(edit,[1,2]))
        self.assertCountEqual(results,[2,'conflict'])
    def test_invalid_or_approval_payload_rejected(self):
        for p in [{**self.payload,'investorApproved':True},{**self.payload,'sourcePack':[]},{**self.payload,'markdown':''},{**self.payload,'qualityIssues':None}]:
            with self.assertRaises(ValueError):self.save(payload=p)
    def test_read_exact_revision_and_bounded_history(self):
        from pipeline_lab_notes import read,history
        self.save()
        self.save(1,str(uuid.uuid4()),{**self.payload,'markdown':'Second'})
        with self.db() as (_,c):
            self.assertEqual(read(c,self.note)['revision'],2)
            self.assertEqual(read(c,self.note,1)['markdown'],self.payload['markdown'])
            self.assertIsNone(read(c,self.note,3))
            page=history(c,self.note,limit=1)
            self.assertEqual(page['nextBefore'],2)
            self.assertNotIn('markdown',page['versions'][0])
            last=history(c,self.note,before=page['nextBefore'],limit=1)
            self.assertEqual(last['versions'][0]['revision'],1)
            self.assertIsNone(last['nextBefore'])
            for kwargs in ({'limit':0},{'limit':51},{'before':True},{'limit':True}):
                with self.assertRaises(ValueError):history(c,self.note,**kwargs)
    def test_prose_edit_preserves_caveats_and_replays_after_later_edit(self):
        from pipeline_lab_notes import edit,read
        self.save();event=str(uuid.uuid4());changes={'title':'Revised','markdown':'New prose'}
        with self.db(True) as (_,c):
            self.assertEqual(edit(c,self.note,event,1,changes)['revision'],2)
            edit(c,self.note,str(uuid.uuid4()),2,{'title':'Third','markdown':'Later'})
            self.assertTrue(edit(c,self.note,event,1,changes)['replayed'])
            latest=read(c,self.note)
            self.assertEqual(latest['qualityIssues'],self.payload['qualityIssues'])
            self.assertEqual(latest['sourcePack'],self.payload['sourcePack'])
            self.assertFalse(latest['investorApproved'])
        with self.assertRaises(ValueError), self.db(True) as (_,c):
            edit(c,self.note,str(uuid.uuid4()),1,changes)
        for extra in ('qualityIssues','sourcePack','investorApproved'):
            with self.assertRaises(ValueError), self.db(True) as (_,c):
                edit(c,self.note,str(uuid.uuid4()),3,{**changes,extra:[]})
    def test_generated_evidence_retained_on_edits_and_cannot_be_replaced(self):
        from pipeline_lab_notes import edit,read
        evidence={'sourcesMarkdown':'Fixture report p2','chartSpecs':[],'stopReason':'max_tokens'}
        payload={**self.payload,'generationEvidence':evidence}
        self.save(payload=payload)
        with self.db(True) as (_,c):
            edit(c,self.note,str(uuid.uuid4()),1,{'title':'Edited','markdown':'New prose'})
            draft=read(c,self.note)
            self.assertEqual(draft['generationEvidence'],evidence)
            self.assertTrue(any('output limit' in i for i in draft['qualityIssues']))
            self.assertTrue(any('investor verification' in i for i in draft['qualityIssues']))
        changed={**payload,'generationEvidence':{**evidence,'sourcesMarkdown':'Different'}}
        with self.assertRaisesRegex(ValueError,'generated evidence'):
            self.save(2,str(uuid.uuid4()),changed)
        with self.assertRaisesRegex(ValueError,'generated evidence'):
            self.save(2,str(uuid.uuid4()),self.payload)
