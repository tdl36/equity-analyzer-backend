"""Synthetic recovery only: no watched folders, real uploads or model calls."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flask import Flask, jsonify, request
import audio_intake_receipts as receipts

AGENT = ast.parse(Path('charlie_local_agent.py').read_text())
APP = ast.parse(Path('app_v3.py').read_text())


def load(tree, name, scope):
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node = __import__('copy').deepcopy(node)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<isolated>', 'exec'), scope)
    return scope[name]


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'receipts.db'
        self.history = Mock(side_effect=lambda name: {'filename': name, 'history': [], 'truncated': False})

    def prepare(self, name='source.mp3', data=b'original', reader=None):
        return receipts.prepare(name, data, '/audio', reader or self.history, self.path)

    def test_reservation_survives_crash_and_rename(self):
        first, allowed = self.prepare()
        self.assertTrue(allowed)
        again, allowed = self.prepare('renamed.mp3')
        self.assertFalse(allowed)
        self.assertEqual(first['id'], again['id'])
        self.assertEqual(self.history.call_count, 1)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_only_one_concurrent_caller_can_upload(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(lambda _: self.prepare(), range(12)))
        self.assertEqual(sum(allowed for _, allowed in rows), 1)

    def test_new_bytes_need_new_history_check(self):
        self.prepare()
        self.assertTrue(self.prepare(data=b'different')[1])
        self.assertEqual(self.history.call_count, 2)

    def test_legacy_history_and_truncation_hold_all_outcomes(self):
        for status in ('complete', 'failed', 'running', 'error'):
            history = lambda name: {'filename': name, 'history': [{'id': 'old', 'status': status}], 'truncated': False}
            row, allowed = self.prepare(data=status.encode(), reader=history)
            self.assertFalse(allowed)
            self.assertEqual(row['status'], 'attention')
        row, allowed = self.prepare(data=b'bounded', reader=lambda name: {'filename': name, 'history': [], 'truncated': True})
        self.assertFalse(allowed)

    def test_unavailable_history_retries_only_read(self):
        with self.assertRaises(TimeoutError):
            self.prepare(reader=Mock(side_effect=TimeoutError))
        self.assertTrue(self.prepare()[1])
        for payload in ({}, None, {'filename': 'wrong', 'history': [], 'truncated': False}):
            with self.assertRaises(ValueError):
                self.prepare(data=str(payload).encode(), reader=lambda _: payload)

    def test_saved_job_and_completion_are_durable(self):
        row, _ = self.prepare()
        receipts.update(row['id'], 'submitted', job_id='job-1', path=self.path)
        again, allowed = self.prepare()
        self.assertFalse(allowed)
        self.assertEqual(again['job_id'], 'job-1')
        receipts.update(row['id'], 'complete', path=self.path)
        late = receipts.update(row['id'], 'attention', reason='late failure', path=self.path)
        self.assertEqual(late['status'], 'complete')
        with self.assertRaises(ValueError):
            receipts.update(row['id'], 'submitted', job_id='job-2', path=self.path)

    def test_late_history_cannot_erase_reserved_job(self):
        def raced(name):
            row, allowed = self.prepare(reader=self.history)
            self.assertTrue(allowed)
            receipts.update(row['id'], 'submitted', job_id='job-1', path=self.path)
            return {'filename': name, 'history': [{'id': 'job-1'}], 'truncated': False}
        row, allowed = self.prepare(reader=raced)
        self.assertFalse(allowed)
        self.assertEqual(row['status'], 'submitted')

    def test_invalid_receipts_fail_closed_and_alert_once(self):
        row, _ = self.prepare()
        for bad in ('?', '', None, 'x' * 201):
            with self.assertRaises(ValueError):
                receipts.update(row['id'], 'submitted', job_id=bad, path=self.path)
        with self.assertRaises(ValueError):
            receipts.update('missing', 'attention', path=self.path)
        with self.assertRaises(ValueError):
            receipts.update(row['id'], 'complete', path=self.path)
        self.assertTrue(receipts.reserve_alert(row['id'], self.path))
        self.assertFalse(receipts.reserve_alert(row['id'], self.path))

    def test_unknown_startup_original_never_assumes_absent_history_means_new(self):
        row, allowed = receipts.prepare('source.mp3', b'original', '/audio', self.history, self.path, allow_new=False)
        self.assertFalse(allowed)
        self.assertEqual(row['status'], 'attention')
        self.history.assert_not_called()
        self.assertFalse(self.prepare()[1])

    def test_byte_identity_required_before_move(self):
        row, _ = self.prepare()
        original = Path(self.temp.name) / 'source.mp3'
        original.write_bytes(b'original')
        self.assertTrue(receipts.same_original(original, row['digest']))
        original.write_bytes(b'changed')
        self.assertFalse(receipts.same_original(original, row['digest']))


class AgentRecoveryTests(unittest.TestCase):
    setUp = ReceiptTests.setUp
    # Exercise actual nested upload and poll functions without starting the agent.
    def scope(self):
        api = SimpleNamespace(
            prepare=lambda *a, **k: receipts.prepare(*a, **k, path=self.path),
            update=lambda *a, **k: receipts.update(*a, **k, path=self.path),
            reserve_alert=lambda *a: receipts.reserve_alert(*a, path=self.path),
            same_original=receipts.same_original)
        http = Mock()
        http.get.return_value.json.return_value = {'filename': 'source.mp3', 'history': [], 'truncated': False}
        scope = {'audio_intake_receipts': api, 'requests': http, 'CHARLIE_API': 'https://invalid.test',
                 '_agent_headers': lambda: {}, 'get_secret': lambda _: '', 'log': Mock(),
                 '_queue_audio_issue': Mock(), 'notify': Mock(), '_known_audio_files': {'source.mp3'}, '_audio_in_flight': {'source.mp3'},
                 '_audio_started_notified': set(), '_audio_startup_files': set(), '_audio_failed_attempts': {}, 'TEXT_EXTS': {'.txt'},
                 'Path': Path, 'SUMMARIES_DIR': Path(self.temp.name)}
        load(AGENT, '_hold_audio_intake', scope)
        scope['_wait_for_audio_job_and_move'] = Mock()
        return scope

    def test_lost_response_is_not_reuploaded_after_restart(self):
        scope = self.scope()
        def post(url, **kwargs):
            if 'auto-process' in url: raise TimeoutError('lost response')
            return Mock()
        scope['requests'].post.side_effect = post
        upload = load(AGENT, '_audio_upload_and_wait', scope)
        upload('source.mp3', Path(self.temp.name) / 'source.mp3', b'original')
        scope['_known_audio_files'] = set()  # simulate a fresh agent process
        upload('source.mp3', Path(self.temp.name) / 'source.mp3', b'original')
        posts = [c for c in scope['requests'].post.call_args_list if 'auto-process' in c.args[0]]
        self.assertEqual(len(posts), 1)
        scope['_wait_for_audio_job_and_move'].assert_not_called()

    def test_successful_receipt_resumes_readonly(self):
        scope = self.scope()
        scope['requests'].post.return_value = Mock(ok=True)
        scope['requests'].post.return_value.json.return_value = {'jobId': 'job-1'}
        upload = load(AGENT, '_audio_upload_and_wait', scope)
        for _ in range(2):upload('source.mp3', Path(self.temp.name) / 'source.mp3', b'original')
        self.assertEqual(scope['requests'].post.call_count, 1)
        self.assertEqual(scope['_wait_for_audio_job_and_move'].call_count, 2)

    def test_startup_original_is_held_but_existing_receipt_can_resume(self):
        scope = self.scope();scope['_audio_startup_files'].add('source.mp3')
        upload = load(AGENT, '_audio_upload_and_wait', scope)
        upload('source.mp3', Path(self.temp.name) / 'source.mp3', b'original')
        self.assertTrue(all('/api/alerts' in c.args[0] for c in scope['requests'].post.call_args_list))
        scope['requests'].get.assert_not_called()

    def test_old_backend_or_unavailable_history_never_uploads(self):
        scope = self.scope()
        scope['requests'].get.return_value.json.return_value = {'jobs': {}, 'count': 0}
        upload = load(AGENT, '_audio_upload_and_wait', scope)
        upload('source.mp3', Path(self.temp.name) / 'source.mp3', b'original')
        scope['requests'].post.assert_not_called()

    def test_http_error_and_malformed_success_never_repeat_upload(self):
        for response in (Mock(ok=False, status_code=502), Mock(ok=True)):
            response.json.return_value = {}
            with TemporaryDirectory() as folder:
                self.path = Path(folder) / 'receipts.db'
                scope = self.scope()
                scope['requests'].post.return_value = response
                upload = load(AGENT, '_audio_upload_and_wait', scope)
                for _ in range(2):upload('source.mp3', Path(folder) / 'source.mp3', b'original')
                posts = [c for c in scope['requests'].post.call_args_list if 'auto-process' in c.args[0]]
                self.assertEqual(len(posts), 1)

    def test_completed_job_cannot_move_changed_or_colliding_original(self):
        for changed in (True, False):
            with TemporaryDirectory() as folder:
                self.path = Path(folder) / 'receipts.db'
                scope = self.scope()
                row, _ = receipts.prepare('source.mp3', b'original', '/audio', self.history, self.path)
                receipts.update(row['id'], 'submitted', job_id='job-1', path=self.path)
                row['job_id'] = 'job-1'
                original = Path(folder) / 'source.mp3'
                original.write_bytes(b'changed' if changed else b'original')
                scope['SUMMARIES_DIR'] = Path(folder)
                processed = Path(folder) / 'Processed';processed.mkdir()
                if not changed:(processed / original.name).write_bytes(b'older original')
                scope['requests'].get.return_value = Mock(status_code=200, ok=True)
                scope['requests'].get.return_value.json.return_value = {'status': 'complete', 'summaryId': 'saved'}
                wait = load(AGENT, '_wait_for_audio_job_and_move', scope)
                with patch('threading.Thread', side_effect=lambda target, **k: SimpleNamespace(start=target)), patch('time.sleep'):
                    wait(original.name, original, 'job-1', row)
                self.assertTrue(original.exists())
                if not changed:self.assertEqual((processed / original.name).read_bytes(), b'older original')
                scope['notify'].assert_not_called()

    def test_failed_or_missing_job_preserves_original_and_known_set(self):
        for code, status in ((404, None), (200, 'failed'), (200, 'error'), (200, 'attention')):
            scope = self.scope()
            row, _ = receipts.prepare('source.mp3', str(code).encode()+str(status).encode(), '/audio', self.history, self.path)
            receipts.update(row['id'], 'submitted', job_id='job-1', path=self.path)
            row['job_id'] = 'job-1'
            original = Path(self.temp.name) / 'source.mp3';original.write_bytes(b'original')
            scope['requests'].get.return_value = Mock(status_code=code, ok=code==200)
            scope['requests'].get.return_value.json.return_value = {'status': status}
            wait = load(AGENT, '_wait_for_audio_job_and_move', scope)
            with patch('threading.Thread', side_effect=lambda target, **k: SimpleNamespace(start=target)), patch('time.sleep'):
                wait('source.mp3', original, 'job-1', row)
            self.assertEqual(original.read_bytes(), b'original')
            self.assertIn('source.mp3', scope['_known_audio_files'])
            self.assertTrue(all('/api/alerts' in c.args[0] for c in scope['requests'].post.call_args_list))


class HistoryEndpointTests(unittest.TestCase):
    def invoke(self, rows, live=None, filename='source.mp3', broken=False):
        flask = Flask(__name__)
        cursor = Mock();cursor.fetchall.return_value = rows
        @contextmanager
        def get_db():
            if broken:raise RuntimeError('unavailable')
            yield None, cursor
        scope = {'request': request, 'jsonify': jsonify, 'get_db': get_db, '_transcription_jobs': live or {}}
        route = load(APP, 'list_transcription_jobs', scope)
        with flask.test_request_context('/?filename='+filename):
            response = flask.make_response(route())
            return response.status_code, response.get_json(), cursor, response.headers

    def test_saved_and_unmirrored_jobs_are_combined(self):
        code, body, cursor, headers = self.invoke([{'id': 'saved', 'status': 'failed', 'summary_id': 3}],
            {'live': {'filename': 'source.mp3', 'status': 'processing'}, 'other': {'filename': 'other.mp3'}})
        self.assertEqual(code, 200)
        self.assertEqual({r['id'] for r in body['history']}, {'saved', 'live'})
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(cursor.execute.call_args.args[1], ('source.mp3',))
        self.assertNotIn('source body', str(body))

    def test_history_is_bounded_and_failure_is_not_empty(self):
        code, body, _, _ = self.invoke([{'id': str(i), 'status': 'failed'} for i in range(21)])
        self.assertEqual(len(body['history']), 20)
        self.assertTrue(body['truncated'])
        self.assertEqual(self.invoke([], broken=True)[0], 503)
        self.assertEqual(self.invoke([], filename='')[0], 400)
