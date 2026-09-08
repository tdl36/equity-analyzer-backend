import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from meeting_question_generation import generate,MODEL

class QuestionGenerationTests(unittest.TestCase):
    def client(self,stop='end_turn',filename='source.pdf'):
        topics=[{'topic':'Growth','description':'Evidence','questions':[dict(question='What changes?',context='Reported evidence',source='Report',follow_up_angle='What reverses it?',priority='high',source_filenames=[filename])]}]
        response=SimpleNamespace(stop_reason=stop,content=[SimpleNamespace(type='text',text=json.dumps({'topics':topics}))],usage=SimpleNamespace(input_tokens=100,output_tokens=50))
        client=MagicMock();client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value=response
        return client
    def test_schema_constrains_citations_and_uses_complete_stream(self):
        client=self.client();topics,tokens=generate('key','ABT','Abbott','Healthcare',{},[],['source.pdf'],client)
        self.assertEqual(tokens,150)
        params=client.messages.stream.call_args.kwargs
        self.assertEqual(params['model'],MODEL)
        self.assertEqual(params['output_config']['format']['type'],'json_schema')
        self.assertEqual(len(topics),1)
    def test_truncated_or_unknown_sources_never_saved(self):
        for client in (self.client('max_tokens'),self.client(filename='invented.pdf')):
            with self.assertRaises(ValueError):generate('key','ABT','Abbott','Healthcare',{},[],['source.pdf'],client)
