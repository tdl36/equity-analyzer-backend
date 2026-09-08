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
        self.assertEqual(params['max_tokens'],32000)
        self.assertEqual(params['output_config']['effort'],'low')
        self.assertEqual(params['output_config']['format']['type'],'json_schema')
        self.assertEqual(len(topics),1)
    def test_truncated_or_unknown_sources_never_saved(self):
        for client in (self.client('max_tokens'),self.client(filename='invented.pdf')):
            with self.assertRaises(ValueError):generate('key','ABT','Abbott','Healthcare',{},[],['source.pdf'],client)

class EvidenceRepairTests(unittest.TestCase):
    def response(self,quote):
        topics=[{'topic':'Growth','description':'Evidence','questions':[dict(question='What changes?',context='Reported evidence',source='Report',follow_up_angle='What reverses it?',priority='high',source_filenames=['source.pdf'],supporting_passage_ids=['p1' if '10 percent' in quote else 'invented'])]}]
        return SimpleNamespace(stop_reason='end_turn',content=[SimpleNamespace(type='text',text=json.dumps({'topics':topics}))],usage=SimpleNamespace(input_tokens=100,output_tokens=50))
    def test_repairs_invalid_quote_once_and_counts_both_calls(self):
        quote='The company reported revenue growth of 10 percent this quarter.'
        client=MagicMock();client.messages.stream.return_value.__enter__.return_value.get_final_message.side_effect=[self.response(quote.replace('10','20')),self.response(quote)]
        evidence={'sources':[{'filename':'source.pdf','pages':[{'text':quote}]}]}
        topics,tokens=generate('key','MDT','Medtronic','Healthcare',{},[],['source.pdf'],client,evidence)
        self.assertEqual(tokens,300);self.assertEqual(client.messages.stream.call_count,2)
        self.assertIn('Topic 1, question 1',client.messages.stream.call_args.kwargs['messages'][-1]['content'])
        self.assertEqual(topics[0]['questions'][0]['supporting_quotes'][0]['quote'],quote)
    def test_second_invalid_draft_is_rejected_without_more_calls(self):
        quote='The company reported revenue growth of 10 percent this quarter.'
        client=MagicMock();client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value=self.response(quote.replace('10','20'))
        evidence={'sources':[{'filename':'source.pdf','pages':[{'text':quote}]}]}
        with self.assertRaises(ValueError):generate('key','MDT','Medtronic','Healthcare',{},[],['source.pdf'],client,evidence)
        self.assertEqual(client.messages.stream.call_count,2)

class MeetingProfileGenerationTests(QuestionGenerationTests):
    def test_hosted_profile_reaches_final_model_without_short_pack_instruction(self):
        client=self.client()
        generate('key','ABT','Abbott','Healthcare',{},[],['source.pdf'],client,meeting_profile={'format':'hosted_pm','audience':'generalist'})
        prompt=client.messages.stream.call_args.kwargs['messages'][0]['content']
        self.assertIn('35–45',prompt)
        self.assertIn('generalist portfolio managers',prompt)
        self.assertIn('segment economics',prompt)
        self.assertNotIn('12–15',prompt)
    def test_revision_instruction_reaches_model_as_request_not_source_evidence(self):
        client=self.client()
        generate('key','ABT','Abbott','Healthcare',{},[],['source.pdf'],client,revision_context={'instruction':'Expand cash conversion','priorQuestions':[]})
        prompt=client.messages.stream.call_args.kwargs['messages'][0]['content']
        self.assertIn('USER REVISION REQUEST',prompt);self.assertIn('Expand cash conversion',prompt)
