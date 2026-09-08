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

class PassageRegisterTests(unittest.TestCase):
    def test_verbatim_passages_attached_without_model_transcription(self):
        from meeting_source_support import passage_register,attach_passages
        text='Revenue was $10 million, with a 20% operating margin. '*60
        evidence={'sources':[{'filename':'a.pdf','pages':[{'page':2,'text':text}]}]}
        register=passage_register(evidence)
        self.assertGreater(len(register),1)
        for entry in register.values():self.assertIn(entry['quote'],text)
        topics=[{'questions':[{'source_filenames':['a.pdf'],'supporting_passage_ids':['p1']}]}]
        attach_passages(topics,register);verify(topics,evidence)
        self.assertEqual(topics[0]['questions'][0]['supporting_quotes'][0]['page'],2)
    def test_cross_source_selection_rejected(self):
        from meeting_source_support import attach_passages
        with self.assertRaises(ValueError):attach_passages([{'questions':[{'source_filenames':['b.pdf'],'supporting_passage_ids':['p1']}]}],{'p1':{'filename':'a.pdf','quote':'Original source text'}})
