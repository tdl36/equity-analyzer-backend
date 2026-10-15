import copy
import unittest
from datetime import date
from meeting_answer_evidence import resolve,document,validate_reference,prompt_context
from research_workbench import answer_hash

IDENT='11111111-1111-4111-8111-111111111111'
class Cursor:
    def __init__(self,rows):self.rows=iter(rows);self.calls=[]
    def execute(self,sql,args):self.calls.append((sql,args))
    def fetchone(self):return next(self.rows)

class MeetingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.answer={'id':4,'meeting_id':2,'meeting_date':date(2026,1,1),'question':'What constrains the outlook?','response_notes':'Recorded answer: demand remains uncertain and no guidance change was confirmed.'}
        self.body={'kind':'followup','answerId':4,'answerHash':answer_hash(self.answer),'answerSnapshot':dict(self.answer),'caseRevision':3,'assumptionId':'assumption','rationale':'My interpretation, not a source quotation','nextAction':'Check official results','resolution':'partial','title':'Outlook follow-up'}
        self.row={'ticker':'ABT','revision':2,'body':self.body}
        self.case={'revision':3,'body':{'assumptions':[{'id':'assumption'}]}}
        self.ref={'id':IDENT,'revision':2}
    def test_freezes_provenance_but_only_answer_is_citable(self):
        cur=Cursor([self.row,self.answer,self.case]);e=resolve(cur,'ABT',self.ref)
        self.assertEqual(e['workRevision'],2);self.assertEqual(e['answer']['meeting_date'],'2026-01-01')
        self.assertEqual(document(e)['extracted_text'],self.answer['response_notes'])
        self.assertNotIn(self.body['rationale'],document(e)['extracted_text'])
        self.assertIn('not a verified transcript',prompt_context(e))
        self.assertIn('FOR SHARE OF q',cur.calls[-2][0])
    def test_cross_ticker_and_stale_work_are_rejected(self):
        for changed in ({**self.row,'ticker':'CVS'},{**self.row,'revision':3}):
            with self.assertRaises(ValueError):resolve(Cursor([changed]),'ABT',self.ref)
    def test_changed_deleted_or_unanswered_answer_is_rejected(self):
        for answer in (None,{**self.answer,'response_notes':'Revised answer'}):
            with self.assertRaises(ValueError):resolve(Cursor([self.row,answer]),'ABT',self.ref)
    def test_case_change_and_removed_assumption_are_rejected(self):
        for case in (None,{**self.case,'revision':4},{'revision':3,'body':{'assumptions':[]}}):
            with self.assertRaises(ValueError):resolve(Cursor([self.row,self.answer,case]),'ABT',self.ref)
    def test_reference_rejects_client_supplied_source_text(self):
        for ref in ({**self.ref,'answer':'fabricated'},{'id':IDENT,'revision':True},{'id':'bad','revision':1}):
            with self.assertRaises(ValueError):validate_reference(ref)
    def test_oversized_notes_fail_before_generation(self):
        self.answer['response_notes']='x'*120001;self.body['answerHash']=answer_hash(self.answer)
        with self.assertRaises(ValueError):resolve(Cursor([self.row,self.answer,self.case]),'ABT',self.ref)

class CallReservationTests(unittest.TestCase):
    def test_unknown_attempt_cannot_be_replayed(self):
        from contextlib import contextmanager
        from meeting_answer_evidence import reserve_call
        cur=Cursor([{'status':'running','owner':'owner','result':{'meetingCalls':{'key:draft':'reserved'}}}])
        @contextmanager
        def db(**kwargs):yield None,cur
        with self.assertRaisesRegex(ValueError,'reconcile the provider'):
            reserve_call(db,'job','owner','key','draft')
        self.assertFalse(any(sql.startswith('UPDATE') for sql,_ in cur.calls))
    def test_reservation_preserves_completed_checkpoints(self):
        from contextlib import contextmanager
        from meeting_answer_evidence import reserve_call
        cur=Cursor([{'status':'running','owner':'owner','result':{'checkpoint':{'draft':{'changes':[]}}}}])
        @contextmanager
        def db(**kwargs):yield None,cur
        reserve_call(db,'job','owner','key','review')
        import json
        saved=json.loads(cur.calls[-1][1][0])
        self.assertEqual(saved['checkpoint'],{'draft':{'changes':[]}})
        self.assertEqual(saved['meetingCalls'],{'key:review':'reserved'})
