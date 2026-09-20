"""Safe pure tests: no app import, database, providers, or real notes."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from summary_comparison import DIRECT_SOURCE_LIMIT, SECTIONS, split_text, generate

class ComparisonTests(unittest.TestCase):
    def test_lossless_arbitrary_length_and_unicode(self):
        text=('Management: 12% is a target, not guidance.\n投資家の質問 🙂\n'*6000)+'FINAL MATERIAL ANSWER'
        parts=split_text(text)
        self.assertGreater(len(text),200000)
        self.assertEqual(''.join(p[2] for p in parts),text)
        self.assertEqual(parts[-1][1],len(text))
        self.assertTrue(all(end-start<=24000 for start,end,_ in parts))
        self.assertTrue(all(parts[i][1]==parts[i+1][0] for i in range(len(parts)-1)))

    def test_resume_preserves_completed_work(self):
        state={};calls=[];saved=[]
        def ask(system,message,tokens):
            calls.append(message)
            if len(calls)==2:raise RuntimeError('interruption')
            return 'Management evidence [Part 1].'
        with self.assertRaises(RuntimeError):
            generate('A'*50000,state,ask,lambda s:saved.append(copy.deepcopy(s)))
        self.assertEqual(len(state['parts']),1)
        completed=state['parts']['0']['record']
        resumed=[]
        generate('A'*50000,state,lambda s,m,t:resumed.append(m) or 'Saved result',lambda s:None)
        self.assertEqual(state['parts']['0']['record'],completed)
        self.assertFalse(any(m.startswith('SOURCE PART 1/') for m in resumed))
        self.assertEqual(state['coveredCharacters'],50000)
        self.assertEqual(len(state['sections']),len(SECTIONS))
        # Two unfinished record parts, one Q&A pass per source part, then one
        # request per section drafted from the records.
        parts=len(state['parts'])
        self.assertEqual(len(resumed),2+parts+(len(SECTIONS)-1))

    def test_late_answer_reaches_synthesis(self):
        marker='LATE ANSWER: management declined exact percentage.'
        text='A'*220000+marker
        prompts=[]
        def ask(s,m,t):
            prompts.append(m)
            return marker if marker in m else 'Evidence from earlier part.'
        state=generate(text,{},ask,lambda s:None)
        self.assertTrue(all(marker in p for p in prompts[-4:]))
        self.assertEqual(state['coveredCharacters'],len(text))

    def test_hierarchical_synthesis_retains_all_detailed_parts(self):
        # Consolidation is the fallback for a source too large to read whole;
        # below DIRECT_SOURCE_LIMIT the sections read the source instead.
        def ask(s,m,t):
            if m.startswith('SOURCE PART'):return 'Detailed source record. '*1500
            return 'Consolidated evidence'
        source='Z'*(DIRECT_SOURCE_LIMIT+50000)
        state=generate(source,{},ask,lambda s:None)
        self.assertEqual(state['synthesisBasis'],'records')
        self.assertTrue(state['hierarchicalSynthesis'])
        self.assertEqual(len(state['parts']),len(split_text(source)))
        self.assertTrue(all(len(p['record'])>30000 for p in state['parts'].values()))

    def test_a_source_that_fits_is_read_directly_without_consolidation(self):
        def ask(s,m,t):
            return 'Detailed source record.' if m.startswith('SOURCE PART') else 'Section prose'
        state=generate('Z'*100000,{},ask,lambda s:None)
        self.assertEqual(state['synthesisBasis'],'source')
        self.assertFalse(state['hierarchicalSynthesis'])

    def test_completed_sections_not_replayed(self):
        state=generate('Meeting text',{},lambda *a:'Result',lambda s:None)
        generate('Meeting text',state,lambda *a:self.fail('Completed call replayed'),lambda s:None)

if __name__=='__main__':unittest.main()
