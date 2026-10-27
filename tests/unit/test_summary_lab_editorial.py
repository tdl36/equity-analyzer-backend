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


class SectionDistinctionTests(unittest.TestCase):
    def test_new_plan_checks_bullet_brief_and_resumes_without_new_calls(self):
        calls=[]
        def ask(system,prompt,tokens):
            calls.append(prompt)
            if 'Return JSON only:' in prompt:
                return json.dumps({'record':'Margin may improve.','passages':['Margin may improve.'],'issues':[]})
            return '- Margin may improve if execution holds.'
        state={'editorialPolicy':editorial.POLICY,'conciseBrief':True}
        generate('Margin may improve.',state,ask,lambda s:None)
        self.assertEqual(len(state['completedSections']),7)
        self.assertIn('bullet_brief',state['checks'])
        self.assertIn('bullet_brief',state['editorialFlags'])
        note_prompts=[p for p in calls if 'Write detailed Meeting Notes' in p]
        self.assertEqual(len(note_prompts),2)
        self.assertTrue(all('Compress syntax, not meaning' in p for p in note_prompts))
        self.assertTrue(any('Write Executive Brief · Bullets' in p for p in calls))
        before=len(calls)
        generate('Margin may improve.',state,ask,lambda s:None)
        self.assertEqual(len(calls),before)

    def test_korean_only_does_not_add_english_brief(self):
        def ask(system,prompt,tokens):
            if 'Return JSON only:' in prompt:
                return json.dumps({'record':'Margin may improve.','passages':['Margin may improve.'],'issues':[]})
            return '마진 개선 가능'
        state={'editorialPolicy':editorial.POLICY,'conciseBrief':True,'outputMode':'korean_only'}
        generate('Margin may improve.',state,ask,lambda s:None)
        self.assertEqual(set(state['sections']),{'notes','korean'})

    def test_bullet_export_preserves_list_and_excludes_private_review(self):
        from summary_lab import export_row
        row={'state':{'sections':{'bullet_brief':'- Margin may improve.\n- EPS < 5.'},'finalReview':'PRIVATE'}}
        result=export_row(row,'bullet_brief')
        self.assertIn('<ul>',result['bullet_brief'])
        self.assertIn('&lt;',result['bullet_brief'])
        self.assertNotIn('PRIVATE',str(result))
