"""Safe tests: never import the application or access its database."""
import json
import unittest
from copy import deepcopy
from summary_lab import generate, validate_review, SECTIONS

class LabTests(unittest.TestCase):
    def ask(self, system, prompt, tokens):
        self.calls.append(prompt)
        if 'Return JSON only:' in prompt:
            body=prompt.split('SOURCE PART P',1)[1].split(':\n',1)[1]
            return json.dumps({'record':body,'passages':[body[:50]],'issues':[]})
        return 'Management stated growth remains uncertain. Interpretation: insufficient evidence for a model change. [P1]'
    def setUp(self): self.calls=[]
    def test_full_source_tail_and_all_sections(self):
        source=('Revenue may improve.\n'*1400)+'FINAL QUALIFICATION: timing is uncertain.'
        state={};generate(source,state,self.ask,lambda s:None)
        self.assertEqual(set(state['sections']),set(SECTIONS))
        self.assertEqual(''.join(p['record'] for p in state['parts'].values()),source)
        for section in SECTIONS:
            checks=[p for p in self.calls if f'Review the draft against ORIGINAL' in p and 'FINAL QUALIFICATION' in p]
            self.assertGreaterEqual(len(checks),5)
        count=len(self.calls);generate(source,state,self.ask,lambda s:None)
        self.assertEqual(len(self.calls),count)
    def test_pdf_whitespace_returns_exact_original(self):
        result=validate_review({'record':'r','passages':['Revenue grew 5 percent.']},'Revenue\n grew 5  percent.')
        self.assertEqual(result['passages'],['Revenue\n grew 5  percent.'])
    def test_invalid_passage_is_excluded_not_certified(self):
        result=validate_review({'record':'r','passages':['Revenue grew 5 percent.','Revenue grew 9 percent.']},'Revenue grew 5 percent.')
        self.assertEqual(result['passages'],['Revenue grew 5 percent.'])
        self.assertTrue(result['issues'])
    def test_quote_mismatch_rejected(self):
        with self.assertRaises(ValueError):validate_review({'record':'x','passages':['fiction']},'original')
    def test_interrupted_revision_resumes(self):
        state={}; snapshots=[]
        def interrupted(system,prompt,tokens):
            if 'Revise only' in prompt: raise RuntimeError('interruption')
            return self.ask(system,prompt,tokens)
        with self.assertRaises(RuntimeError):generate('Management: uncertain.',state,interrupted,lambda s:snapshots.append(deepcopy(s)))
        self.assertNotIn('brief',state['completedSections'])
        generate('Management: uncertain.',state,self.ask,lambda s:None)
        self.assertEqual(len(state['completedSections']),5)

if __name__=='__main__':unittest.main()
