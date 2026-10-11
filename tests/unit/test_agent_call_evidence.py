import json
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch
from uuid import uuid4

from agent_call_evidence import append, attach, summary


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.execution = str(uuid4())
        self.call = str(uuid4())

    @contextmanager
    def db(self, commit=False):
        self.assertTrue(commit)
        cur = NS(execute=lambda sql, args: self.rows.extend(json.loads(args[0])),
                 fetchone=lambda: {'id': 'saved'})
        yield None, cur

    def test_response_allowlist_and_unresolved(self):
        append(self.db, 'saved', self.execution, self.call, 'started')
        self.assertEqual(summary(self.rows)['unresolved'], 1)
        response = NS(generations=[[NS(message=NS(id='msg_123', content='PRIVATE')),
                                    NS(message=NS(id='run--local')),
                                    NS(message=NS(id='secret/header'))]])
        append(self.db, 'saved', self.execution, self.call, 'returned', response)
        self.assertEqual(self.rows[-1]['responseIds'], ['msg_123'])
        self.assertNotIn('PRIVATE', json.dumps(self.rows))
        self.assertEqual(summary(self.rows)['unresolved'], 0)
        self.assertEqual(summary(self.rows)['returned'], 1)

    def test_errors_do_not_prove_no_charge_or_safe_retry(self):
        append(self.db, 'saved', self.execution, self.call, 'error')
        result = summary(self.rows)
        self.assertEqual(result['errors'], 1)
        self.assertNotIn('retry', result)
        self.assertEqual(summary([{'event': 'model_call_v1', 'callId': 'bad'}])['recordedCalls'], 0)

    def test_instance_scoping_and_no_prompt_capture(self):
        original_config = {'callbacks': []}
        graph = NS(propagator=NS(get_graph_args=lambda: {'config': original_config, 'stream_mode': 'values'}))
        untouched = NS(propagator=NS(get_graph_args=lambda: {}))
        with patch.dict(sys.modules, {'langchain_core.callbacks.base': NS(BaseCallbackHandler=object)}):
            attach(graph, self.db, 'saved', self.execution)
        args = graph.propagator.get_graph_args()
        self.assertEqual(original_config, {'callbacks': []})
        self.assertEqual(untouched.propagator.get_graph_args(), {})
        handler = args['config']['callbacks'][0]
        self.assertTrue(handler.raise_error)
        self.assertTrue(handler.run_inline)
        handler.on_chat_model_start({'secret': 'PRIVATE'}, ['PRIVATE'], run_id=self.call)
        handler.on_llm_error(Exception('PRIVATE credential'), run_id=self.call)
        self.assertNotIn('PRIVATE', json.dumps(self.rows))
        self.assertEqual(summary(self.rows)['recordedCalls'], 1)

    def test_write_failure_propagates_before_call(self):
        @contextmanager
        def broken(commit=False):
            raise RuntimeError('database unavailable')
            yield
        with self.assertRaises(RuntimeError):
            append(broken, 'saved', self.execution, self.call, 'started')


if __name__ == '__main__':
    unittest.main()
