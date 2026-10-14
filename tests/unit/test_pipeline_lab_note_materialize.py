import unittest
import uuid
from tests.unit import test_pipeline_lab_note_calls as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_note_calls import run_once,reserve
from pipeline_lab_note_response import store_draft
from pipeline_lab_notes import edit,read


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class Materialize(unittest.TestCase):
    setUpClass=classmethod(fixtures.Calls.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.Calls.tearDownClass.__func__)
    setUp=fixtures.Calls.setUp

    def test_saved_reply_becomes_one_draft_even_after_investor_edit(self):
        response={**self.response,'text':'===NOTE_START===Fixture note [1].===NOTE_END==='}
        run_once(self.db,self.request,lambda:response)
        with self.db(True) as (_,c):
            first=store_draft(c,self.note,1,'ABT','Draft')
            self.assertEqual(first['revision'],1)
            edit(c,self.note,str(uuid.uuid4()),1,{'title':'Edited','markdown':'Investor text'})
            replay=store_draft(c,self.note,1,'ABT','Draft')
            self.assertTrue(replay['replayed']);self.assertEqual(replay['revision'],1)
            self.assertEqual(read(c,self.note)['revision'],2)
            self.assertEqual(read(c,self.note)['sourcePack'],self.request['sourcePack'])
        with self.assertRaises(ValueError),self.db(True) as (_,c):
            store_draft(c,self.note,1,'UNH','Draft')

    def test_uncertain_or_malformed_reply_cannot_create_draft(self):
        with self.db(True) as (_,c):reserve(c,self.request)
        with self.assertRaises(ValueError),self.db(True) as (_,c):store_draft(c,self.note,1,'ABT','Draft')
        self.request['noteId']=str(uuid.uuid4());self.note=self.request['noteId']
        run_once(self.db,self.request,lambda:self.response)
        with self.assertRaises(ValueError),self.db(True) as (_,c):store_draft(c,self.note,1,'ABT','Draft')
        with self.db() as (_,c):
            self.assertIsNone(read(c,self.note))
            c.execute('SELECT status FROM pipeline_lab_note_calls WHERE note_id=%s',(self.note,))
            self.assertEqual(c.fetchone()['status'],'complete')
