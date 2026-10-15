import unittest
from research_assignment_plan import plan
from research_purpose import context,note_prompt

BASE={'ticker':'CVS','since':'2026-07-01','until':'2026-10-01','outputs':['summary_note']}
EVENT={'kind':'event_note','event':{'title':' Q3 guidance update ','date':'2026-09-30'}}

class PurposeTests(unittest.TestCase):
    def test_legacy_is_not_reclassified(self):
        self.assertNotIn('researchPurpose',plan(BASE))
        self.assertEqual(context(None),'')
    def test_event_is_frozen_and_cannot_request_thesis_or_whole_stock(self):
        p=plan({**BASE,'researchPurpose':EVENT})
        self.assertEqual(p['researchPurpose']['event']['title'],'Q3 guidance update')
        for output in ['thesis','stock_summary','visual']:
            with self.subTest(output=output),self.assertRaisesRegex(ValueError,'Event notes'):
                plan({**BASE,'outputs':[output],'researchPurpose':EVENT})
    def test_event_needs_identity_and_note_selection(self):
        for purpose in ({'kind':'event_note','event':{'title':'Future','date':'2099-01-01'}},{'kind':'event_note'},{'kind':'event_note','event':{'title':'','date':'2026-09-30'}},{'kind':'event_note','event':{'title':'x','date':'bad'}},{'kind':'other'}):
            with self.assertRaises(ValueError):plan({**BASE,'researchPurpose':purpose})
        with self.assertRaisesRegex(ValueError,'note sources'):
            plan({**BASE,'researchPurpose':EVENT,'labSelection':{'task':'thesis','revision':1,'digest':'a'*64}})
    def test_thesis_scope_is_longitudinal_and_collection_only_spends_nothing(self):
        p=plan({**BASE,'outputs':[],'collectionOnly':True,'researchPurpose':{'kind':'thesis_review'}})
        self.assertTrue(p['collectionOnly'])
        self.assertIn('not the investment case horizon',context(p['researchPurpose']))
        prompt=note_prompt(plan({**BASE,'researchPurpose':EVENT})['researchPurpose'])
        self.assertIn('without producing or applying thesis changes',prompt)
        self.assertIn('===NOTE_START===',prompt)
