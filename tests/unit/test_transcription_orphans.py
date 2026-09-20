"""A backend restart used to strand a transcription job forever.

gunicorn runs one worker, so a job still recorded as running when a new
process starts belongs to a process that no longer exists. The Mac agent
cannot tell a dead job from a slow one: it polls for 90 minutes and then
keeps the file out of the scanner to avoid a duplicate upload, so the audio
is never processed and nothing reports a failure. A real file was lost this
way when a documentation push redeployed Render seven seconds after the
upload started.
"""
import ast
import unittest
from contextlib import contextmanager
from pathlib import Path

SOURCE = Path('app_v3.py').read_text()
TREE = ast.parse(SOURCE)


def load(name, namespace):
    fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app_v3.py', 'exec'), namespace)
    return namespace[name]


class Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = ''
        self.args = ()

    def execute(self, sql, args=()):
        self.sql = ' '.join(sql.split())
        self.args = args

    def fetchall(self):
        return self.rows


class OrphanSweepTests(unittest.TestCase):
    def sweep_with(self, rows):
        cursor = Cursor(rows)

        @contextmanager
        def get_db(**_kwargs):
            yield None, cursor

        fn = load('_cleanup_orphaned_transcription_jobs', {'get_db': get_db, 'print': lambda *a: None})
        return fn(), cursor

    def test_stranded_jobs_are_failed_so_the_agent_retries(self):
        count, cursor = self.sweep_with([{'id': 'e19d64bd'}])
        self.assertEqual(count, 1)
        self.assertIn("SET status='failed'", cursor.sql)
        self.assertIn("status NOT IN ('complete', 'error', 'failed')", cursor.sql)

    def test_staleness_is_measured_on_progress_not_on_age(self):
        # created_at would fail a legitimately long transcription once the
        # sweep runs on a timer rather than only at startup.
        _count, cursor = self.sweep_with([])
        self.assertIn('updated_at < NOW() - make_interval', cursor.sql)
        self.assertNotIn('created_at <', cursor.sql)
        self.assertEqual(cursor.args, (25,))

    def test_the_window_is_configurable_for_the_periodic_caller(self):
        cursor = Cursor([])

        @contextmanager
        def get_db(**_kwargs):
            yield None, cursor

        fn = load('_cleanup_orphaned_transcription_jobs', {'get_db': get_db, 'print': lambda *a: None})
        fn(stale_minutes=60)
        self.assertEqual(cursor.args, (60,))

    def test_finished_jobs_are_untouched(self):
        _count, cursor = self.sweep_with([])
        self.assertIn("NOT IN ('complete', 'error', 'failed')", cursor.sql)

    def test_a_database_failure_does_not_stop_startup(self):
        @contextmanager
        def broken(**_kwargs):
            raise RuntimeError('no database')
            yield

        fn = load('_cleanup_orphaned_transcription_jobs', {'get_db': broken, 'print': lambda *a: None})
        self.assertEqual(fn(), 0)


class ProgressFreshnessTests(unittest.TestCase):
    """The age guard only works if a running job keeps its row young."""

    def test_transcription_mirrors_progress_not_just_status_changes(self):
        body = next(n for n in TREE.body
                    if isinstance(n, ast.FunctionDef) and n.name == '_run_transcription')
        mirrors = [n for n in ast.walk(body)
                   if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == '_mirror_transcription_state']
        self.assertGreaterEqual(len(mirrors), 5,
                                'progress updates must refresh updated_at or a live job looks abandoned')

    def test_the_sweep_runs_at_startup(self):
        self.assertIn('_cleanup_orphaned_transcription_jobs()', SOURCE)
