"""Synthetic transaction/replay checks; never open the application database."""
import ast
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager

import summary_lab


class Cursor:
    def __init__(self, db):
        self.cursor = db.cursor()

    def execute(self, sql, params=()):
        self.cursor.execute(sql.replace('%s', '?').replace('::jsonb', ''), params)

    def fetchone(self):
        return self.cursor.fetchone()


class AtomicAudioLabTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE meeting_summaries(id TEXT PRIMARY KEY);
            CREATE TABLE summary_lab_experiments(
                id TEXT PRIMARY KEY,title TEXT,source TEXT,source_hash TEXT,
                baseline TEXT,focus TEXT,version TEXT,model TEXT,state TEXT,
                summary_id TEXT,output_mode TEXT,automatic BOOLEAN,
                recovery_enabled BOOLEAN,source_job_id TEXT,status TEXT DEFAULT 'queued');
            CREATE UNIQUE INDEX source_job ON summary_lab_experiments(source_job_id)
                WHERE source_job_id IS NOT NULL;
        ''')
        self.cur = Cursor(self.db)

    def prepare(self):
        return summary_lab.prepare_audio_branch(self.cur, 'summary-1', 'audio-1',
                                                'Example', 'Complete synthetic source', {'title': 'Example'})

    def test_failure_rolls_back_original_and_lab_together(self):
        with self.assertRaises(RuntimeError):
            with self.db:
                self.db.execute("INSERT INTO meeting_summaries VALUES ('summary-1')")
                self.prepare()
                raise RuntimeError('simulated interruption before commit')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM meeting_summaries').fetchone()[0], 0)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM summary_lab_experiments').fetchone()[0], 0)

    def test_committed_branch_is_recoverable_without_starting_a_model(self):
        with patch('summary_lab.threading.Thread') as thread:
            with self.db:
                self.db.execute("INSERT INTO meeting_summaries VALUES ('summary-1')")
                jid = self.prepare()
            thread.assert_not_called()
        row = self.db.execute('SELECT * FROM summary_lab_experiments').fetchone()
        self.assertEqual(row['id'], jid)
        self.assertEqual(row['status'], 'queued')
        self.assertTrue(row['automatic'])
        self.assertTrue(row['recovery_enabled'])
        self.assertEqual(row['source'], 'Complete synthetic source')

    def test_replay_does_not_restart_failed_experiment_or_change_prompt_version(self):
        with self.db:
            jid = self.prepare()
            self.db.execute("UPDATE summary_lab_experiments SET status='failed'")
        with patch.object(summary_lab, 'VERSION', 'future-version'):
            with self.db:
                self.assertEqual(self.prepare(), jid)
        rows = self.db.execute('SELECT * FROM summary_lab_experiments').fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertNotEqual(rows[0]['version'], 'future-version')

    def test_application_saves_all_three_records_in_one_transaction(self):
        tree = ast.parse(Path('app_v3.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_auto_process_audio')
        blocks = [n for n in ast.walk(fn) if isinstance(n, ast.With)]
        transactions = [ast.unparse(n) for n in blocks if 'INSERT INTO meeting_summaries' in ast.unparse(n)]
        self.assertEqual(len(transactions), 1)
        self.assertIn('prepare_audio_branch', transactions[0])
        self.assertIn('INSERT INTO transcription_jobs', transactions[0])
        self.assertIn('if fan_out_lab:', transactions[0])
        self.assertNotIn('start_prepared', transactions[0])

    def test_starting_committed_intent_never_retries_finished_or_failed_work(self):
        class Connection:
            def commit(self): pass
        class ResultCursor:
            def execute(self, sql, params=()): self.sql = sql
            def fetchone(self):
                if 'pg_try_advisory_lock' in self.sql: return {'ok': True}
                return {'status': status, 'archived_at': archived}
        @contextmanager
        def get_db(**kwargs): yield Connection(), ResultCursor()
        class ImmediateThread:
            def __init__(self, target, args, **kwargs): self.target, self.args = target, args
            def start(self): self.target(*self.args)
        bp = summary_lab.create_blueprint(get_db)
        with patch('summary_lab.threading.Thread', ImmediateThread), \
                patch('summary_lab.generate') as generate:
            for status, archived in [('failed', None), ('running', None),
                                     ('cancelled', None), ('complete', None), ('queued', 'archived')]:
                with self.subTest(status=status, archived=archived):
                    self.assertEqual(bp.start_prepared('lab-id', 'synthetic-key'), 'lab-id')
            generate.assert_not_called()
