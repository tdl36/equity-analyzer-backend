"""Guard the SUMMARIES audio fan-out against duplicate and lost paid work."""
import ast
import os
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from flask import Flask, jsonify

import summary_lab

SOURCE = Path('app_v3.py').read_text()
TREE = ast.parse(SOURCE)


def load(name, namespace):
    fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app_v3.py', 'exec'), namespace)
    return namespace[name]


def constant(name):
    node = next(
        n for n in TREE.body
        if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == name for t in n.targets)
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'app_v3.py', 'exec'), namespace)
    return namespace[name]


class FanOutOriginTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {'SUMMARIES_FOLDER_ORIGIN': constant('SUMMARIES_FOLDER_ORIGIN')}
        self.should_fan_out = load('_should_fan_out_summary_lab', self.namespace)

    def test_the_folder_watcher_sends_the_marker_the_backend_expects(self):
        marker = constant('SUMMARIES_FOLDER_ORIGIN')
        agent = Path('charlie_local_agent.py').read_text()
        self.assertIn(f"'origin': '{marker}'", agent)

    def test_only_summaries_folder_audio_fans_out(self):
        self.assertTrue(self.should_fan_out('summaries-folder'))
        self.assertTrue(self.should_fan_out('  SUMMARIES-Folder '))

    def test_summary_lab_and_unidentified_uploads_do_not_fan_out(self):
        # Summary Lab's own audio intake starts an experiment itself; fanning
        # out here would run the same multi-pass review twice.
        for origin in ('summary-lab', '', None, 'unknown-caller'):
            self.assertFalse(self.should_fan_out(origin), origin)


class CompletionOrderTests(unittest.TestCase):
    """The saved Summary must be durable before any follow-on queue runs."""

    def setUp(self):
        self.body = next(
            n for n in TREE.body
            if isinstance(n, ast.FunctionDef) and n.name == '_run_auto_process_audio'
        )

    def completion_line(self):
        """Line where the audio job marks itself complete in memory."""
        for node in ast.walk(self.body):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and node.value.value == 'complete':
                target = node.targets[0]
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) \
                        and target.slice.value == 'status':
                    return node.lineno
        self.fail('the audio job no longer marks itself complete')

    def calls_after_completion(self):
        after = self.completion_line()
        found = [
            (node.lineno, node.func.id)
            for node in ast.walk(self.body)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ('_mirror_transcription_state', '_queue_summary_lab', '_queue_improved_summary')
            and node.lineno > after
        ]
        return [name for _lineno, name in sorted(found)]

    def test_completion_is_mirrored_before_the_fan_out_queues(self):
        order = self.calls_after_completion()
        # A restart between 'complete' and the mirror would leave the watcher
        # re-uploading audio that is already transcribed and saved.
        self.assertEqual(order[0], '_mirror_transcription_state')
        self.assertLess(order.index('_mirror_transcription_state'), order.index('_queue_improved_summary'))
        self.assertLess(order.index('_mirror_transcription_state'), order.index('_queue_summary_lab'))
        self.assertGreater(
            len([n for n in order if n == '_mirror_transcription_state']), 1,
            'the Summary Lab id must also be persisted',
        )


class StatusStateTests(unittest.TestCase):
    def setUp(self):
        self.jobs = {}
        self.persisted = None
        owner = self

        class Cursor:
            def execute(self, *args): pass
            def fetchone(self): return owner.persisted

        @contextmanager
        def get_db(): yield None, Cursor()

        app = Flask(__name__)
        load('transcribe_audio_status', {
            'app': app, 'jsonify': jsonify, '_transcription_jobs': self.jobs,
            'get_db': get_db, 'time': time,
        })
        self.client = app.test_client()

    def test_pending_state_is_visible_while_the_fan_out_runs(self):
        self.jobs['a'] = {'status': 'complete', 'summaryId': 's1', 'summaryLabState': 'pending'}
        self.assertEqual(self.client.get('/api/transcribe-audio/a').json['summaryLabState'], 'pending')

    def test_restart_never_reports_pending_from_the_mirror(self):
        self.persisted = {'status': 'complete', 'summary_id': 's1', 'summary_lab_id': None}
        body = self.client.get('/api/transcribe-audio/a').json
        self.assertEqual(body['summaryLabState'], 'unknown')
        self.persisted = {'status': 'complete', 'summary_id': 's1', 'summary_lab_id': 'lab-1'}
        self.assertEqual(self.client.get('/api/transcribe-audio/a').json['summaryLabState'], 'started')


class RecoveryCursor:
    def __init__(self, jobs):
        self.jobs = jobs
        self.rows = []

    def execute(self, sql, args=()):
        compact = ' '.join(sql.split()).lower()
        self.rows = [{'id': jid} for jid in self.jobs] if 'where automatic and recovery_enabled' in compact else []

    def fetchone(self): return None
    def fetchall(self): return self.rows


class FakeThread:
    started = []

    def __init__(self, target=None, args=(), **_kwargs):
        self.target, self.args = target, args

    def start(self):
        FakeThread.started.append(self.args[0])


class RecoverySweepTests(unittest.TestCase):
    def test_one_recovery_thread_per_experiment_across_sweeps(self):
        @contextmanager
        def get_db(**_kwargs): yield None, RecoveryCursor(['lab-1'])

        FakeThread.started = []
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}), \
                patch('summary_lab.threading.Thread', FakeThread):
            bp = summary_lab.create_blueprint(get_db)
            bp.recover_once()
            bp.recover_once()
            bp.recover_once()
        # Only two experiments run at a time; repeated sweeps must not queue a
        # second blocked thread for an experiment already being recovered.
        self.assertEqual(FakeThread.started, ['lab-1'])
        self.assertIn('lab-1', bp.recovering)
