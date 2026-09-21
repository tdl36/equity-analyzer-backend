"""Choosing a model, and recording what every paid call cost."""
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from flask import Flask

import summary_lab

OFFERED = [
    {'key': 'opus-5', 'label': 'Opus 5', 'model': 'claude-opus-5', 'provider': 'anthropic'},
    {'key': 'sonnet-5', 'label': 'Sonnet 5', 'model': 'claude-sonnet-5', 'provider': 'anthropic'},
]


class Harness(unittest.TestCase):
    def setUp(self):
        self.inserted = []
        owner = self

        class Cursor:
            def execute(self, sql, params=None):
                if 'INSERT INTO summary_lab_experiments' in sql: owner.inserted.append(params)
                self.sql = sql
            def fetchone(self):
                if 'FROM meeting_summaries' in getattr(self, 'sql', ''):
                    return {'title': 'CAH', 'raw_notes': 'Source text.', 'brief': '', 'summary': '',
                            'questions': '', 'assessment': '', 'meeting_summary': '', 'korean_takeaways': ''}
                return None
            def fetchall(self): return []

        class Conn:
            def commit(self): pass

        @contextmanager
        def get_db(commit=False): yield Conn(), Cursor()

        app = Flask(__name__)
        app.register_blueprint(summary_lab.create_blueprint(get_db, models=OFFERED))
        self.client = app.test_client()

    def start(self, **extra):
        body = {'source': 'Some source text.', 'title': 'CAH', 'apiKey': 'k', **extra}
        return self.client.post('/api/summary-lab', json=body)


class ModelChoiceTests(Harness):
    def test_the_offered_models_are_published_with_the_sources(self):
        body = self.client.get('/api/summary-lab/sources').json
        self.assertEqual([m['model'] for m in body['models']][:2], ['claude-opus-5', 'claude-sonnet-5'])
        self.assertEqual(body['defaultModel'], summary_lab.MODEL)

    def test_the_default_is_always_offered_even_if_the_host_omits_it(self):
        self.assertIn(summary_lab.MODEL, [m['model'] for m in self.client.get('/api/summary-lab/sources').json['models']])

    def test_a_chosen_model_is_what_the_experiment_records(self):
        self.assertEqual(self.start(model='claude-sonnet-5').status_code, 202)
        self.assertIn('claude-sonnet-5', self.inserted[-1])

    def test_omitting_the_model_keeps_the_default(self):
        self.start()
        self.assertIn(summary_lab.MODEL, self.inserted[-1])

    def test_a_model_not_on_the_list_is_refused_before_anything_is_created(self):
        for value in ('gpt-4o', 'claude-opus-4-1', 42, {'model': 'x'}):
            response = self.start(model=value)
            self.assertEqual(response.status_code, 400, value)
        self.assertEqual(self.inserted, [])


class UsageRecordingTests(unittest.TestCase):
    """No Lab run had ever been recorded, so no experiment had a cost."""

    def setUp(self):
        self.recorded = []

    def run_one_call(self, **kwargs):
        message = SimpleNamespace(
            stop_reason='end_turn', model='claude-opus-5',
            content=[SimpleNamespace(type='text', text='Some output.')],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=340))

        class Stream:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self): return iter(())
            def get_final_message(self): return message

        class Messages:
            def stream(self, **kw): return Stream()

        class Client:
            messages = Messages()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        return summary_lab.ask_with_recovery(
            'key', 'claude-opus-5', 'system', 'prompt', 100, {}, lambda state: None,
            client_factory=lambda **kw: Client(), sleep=lambda s: None, **kwargs)

    def test_every_call_reports_its_tokens_and_model(self):
        self.run_one_call(on_usage=lambda result, attempt: self.recorded.append((result, attempt)))
        self.assertEqual(len(self.recorded), 1)
        result, attempt = self.recorded[0]
        self.assertEqual(result['usage'], {'input_tokens': 1200, 'output_tokens': 340})
        self.assertEqual(result['model'], 'claude-opus-5')
        self.assertEqual(result['provider'], 'anthropic')
        self.assertEqual(attempt, 1)

    def test_the_shape_is_what_the_ledger_expects(self):
        # record_llm_usage reads result['usage']['input_tokens'] and result['model'].
        self.run_one_call(on_usage=lambda result, attempt: self.recorded.append(result))
        result = self.recorded[0]
        self.assertEqual(int(result['usage']['input_tokens']), 1200)
        self.assertTrue(result['model'])

    def test_a_run_without_a_recorder_still_returns_its_text(self):
        self.assertEqual(self.run_one_call(), 'Some output.')


class FailedCallAccountingTests(UsageRecordingTests):
    """A call that overran its budget still spent its tokens. Recording only
    successes hid the cost of the two Opus 5 experiments that failed."""

    def failing_call(self, stop_reason):
        from types import SimpleNamespace
        message = SimpleNamespace(
            stop_reason=stop_reason, model='claude-opus-5',
            content=[SimpleNamespace(type='text', text='partial')],
            usage=SimpleNamespace(input_tokens=30000, output_tokens=3500))

        class Stream:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self): return iter(())
            def get_final_message(self): return message

        class Client:
            messages = SimpleNamespace(stream=lambda **kw: Stream())
            def __enter__(self): return self
            def __exit__(self, *a): return False

        import summary_lab
        try:
            summary_lab.ask_with_recovery(
                'k', 'claude-opus-5', 'sys', 'p', 3500, {}, lambda st: None,
                client_factory=lambda **kw: Client(), sleep=lambda s: None,
                on_usage=lambda r, a: self.recorded.append((r, a)))
        except Exception:
            pass

    def test_a_call_that_hit_its_limit_still_reports_its_tokens(self):
        self.failing_call('max_tokens')
        self.assertTrue(self.recorded, 'tokens were spent and never recorded')
        self.assertEqual(self.recorded[0][0]['usage']['input_tokens'], 30000)

    def test_overrunning_the_budget_does_not_retry_and_bill_again(self):
        self.failing_call('max_tokens')
        # The retry loop catches provider errors, not a bad stop_reason, so an
        # overrun fails once rather than paying for the same prompt three times.
        self.assertEqual(len(self.recorded), 1)
        self.assertEqual(self.recorded[0][1], 1)

    def test_an_incomplete_response_is_recorded_too(self):
        self.failing_call('refusal')
        self.assertTrue(self.recorded)
