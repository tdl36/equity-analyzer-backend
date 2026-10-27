import json
import unittest
from summary_lab import generate
import summary_lab_editorial as editorial

class EditorialTests(unittest.TestCase):
    def test_draft_revision_and_review_use_separate_contracts(self):
        calls=[]
        def ask(system,prompt,tokens):
            calls.append((system,prompt))
            if 'Return JSON only:' in prompt:
                return json.dumps({'record':'Growth is a target, not assured.','passages':['Growth could improve.'],'issues':['Date unclear; verify privately.']})
            return 'The margin recovery depends on pricing and execution.'
        state={'editorialPolicy':editorial.POLICY}
        generate('Growth could improve.',state,ask,lambda s:None)
        drafting=[s for s,p in calls if 'REVIEWED SOURCE RECORDS:' in p or 'Revise only where' in p]
        self.assertEqual(len(drafting),12)
        self.assertTrue(all(s==editorial.RULES for s in drafting))
        self.assertTrue(any(editorial.REVIEW in p for s,p in calls))
        self.assertTrue(any(editorial.AUDIT in p for s,p in calls))
        self.assertEqual(len(state['completedSections']),6)
        self.assertEqual(set(state['editorialFlags']),set(state['completedSections']))
        before=len(calls);generate('Growth could improve.',state,ask,lambda s:None)
        self.assertEqual(len(calls),before)

    def test_diagnostics_do_not_mutate_numbers_or_guessed_values(self):
        note='EPS 20.50 (inferred from transcript); management said "quite well". Margin -1.7% to -1.0%.'
        flags=editorial.review_flags(note)
        self.assertTrue(any('Source/process' in f for f in flags))
        self.assertTrue(any('quotation' in f for f in flags))
        self.assertIn('-1.7%',note)
        self.assertEqual(editorial.review_flags('Margins could reach 2–4% if pricing offsets higher costs.'),[])

    def test_legacy_generation_keeps_original_contract(self):
        systems=[]
        def ask(system,prompt,tokens):
            systems.append(system)
            if 'Return JSON only:' in prompt:
                return json.dumps({'record':'Growth may improve.','passages':['Growth may improve.'],'issues':[]})
            return 'Growth may improve.'
        state={};generate('Growth may improve.',state,ask,lambda s:None)
        self.assertNotIn(editorial.RULES,systems)
        self.assertNotIn('editorialFlags',state)
