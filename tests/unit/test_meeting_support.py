import unittest
from meeting_source_support import verify
from meeting_analysis_cache import key
class SupportTests(unittest.TestCase):
    def test_cache_identity_changes_with_source_prompt_and_model(self):
        base=key('ABT','prompt','source',['model'])
        for args in [('MDT','prompt','source',['model']),('ABT','changed','source',['model']),('ABT','prompt','new source',['model']),('ABT','prompt','source',['new model'])]:self.assertNotEqual(base,key(*args))
    def test_wrong_source_quote_cannot_pass(self):
        passage='The company reported revenue growth of 10 percent this quarter.'
        q={'source_filenames':['a.pdf'],'supporting_quotes':[{'filename':'a.pdf','quote':passage}]};topics=[{'questions':[q]}]
        evidence={'sources':[{'filename':'a.pdf','pages':[{'text':passage}]}]}
        verify(topics,evidence);self.assertIn('matched',q['source_support'])
        q['supporting_quotes'][0]['quote']=passage.replace('10','20')
        with self.assertRaises(ValueError):verify(topics,evidence)
