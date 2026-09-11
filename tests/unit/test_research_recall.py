import unittest
from research_recall import terms_for,retrieve
from research_decisions import validate
from company_memory import assemble,render

class RecallTests(unittest.TestCase):
    def test_empty_case_does_not_invent_search_terms(self):
        self.assertEqual(terms_for(None),([],False))
    def test_terms_are_unique_and_bounded_with_disclosure(self):
        terms,limited=terms_for({'body':{'thesis':' '.join('driver'+str(i) for i in range(40)), 'assumptions':[{'claim':'driver0'}]}})
        self.assertEqual(len(terms),24);self.assertTrue(limited)
    def test_selection_is_bounded_and_marks_omissions(self):
        class Cur:
            def execute(self,sql,params):self.sql,self.params=sql,params
            def fetchall(self):return [dict(id=str(i),relevance=2) for i in range(13)]
        cur=Cur();rows,receipt=retrieve(cur,'ABT',{'body':{'thesis':'Margins'}},[{'id':'recent'}],'2026-09-11')
        self.assertEqual(len(rows),12);self.assertTrue(receipt['additionalMatchesOmitted'])
        self.assertEqual(cur.params[1:],('ABT',['recent'],'2026-09-11'))
        self.assertIn('LIMIT 13',cur.sql)
    def test_no_matches_are_not_presented_as_relevant(self):
        class Cur:
            def execute(self,*args):pass
            def fetchall(self):return [dict(id='irrelevant',relevance=0)]
        rows,receipt=retrieve(Cur(),'ABT',None,[],'2026-09-11')
        self.assertEqual(rows,[]);self.assertEqual(receipt['selectedIds'],[])
    def test_receipt_changes_snapshot_and_keeps_evidence_boundary(self):
        a=assemble('ABT',recall={'selectedIds':['a']})
        b=assemble('ABT',recall={'selectedIds':['b']})
        self.assertNotEqual(a['snapshotHash'],b['snapshotHash'])
        self.assertIn('not verified facts',render(a))

class IssueTests(unittest.TestCase):
    def payload(self):return dict(requestId='00000000-0000-4000-8000-000000000001',revision=0,decisionDate='2026-01-01',decision='Wait',rationale='Uncertain durability',revisitWhen='New evidence')
    def test_issue_identity_cannot_be_forged(self):
        p=self.payload();body=validate({**p,'issue':'Margins','issueId':'forged'})[2]
        self.assertEqual(body['issueId'],p['requestId']);self.assertEqual(body['disposition'],'unresolved')
    def test_invalid_review_is_rejected(self):
        for value in ({'reviewDate':'2026-10-01'},{'issue':'Margins','disposition':'ignore_forever'},{'issue':'Margins','reviewDate':'bad'}):
            with self.assertRaises(ValueError):validate({**self.payload(),**value})
