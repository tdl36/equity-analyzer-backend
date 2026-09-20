"""Archiving hides an experiment; it never destroys one, and never hides live work."""
import unittest
from contextlib import contextmanager

from flask import Flask

import summary_lab


class FakeCursor:
    def __init__(self, owner):
        self.owner = owner
        self._rows = []

    def execute(self, sql, params=None):
        self.owner.sql.append((' '.join(sql.split()), params))
        if 'pg_try_advisory_lock' in sql:
            # True means the lock was free, i.e. no worker holds the row.
            self._rows = [{'ok': not self.owner.live}]
        elif 'RETURNING id' in sql:
            ids = (params or [[]])[0]
            self._rows = [{'id': i} for i in ids]
        elif 'FROM summary_lab_experiments' in sql:
            self._rows = list(self.owner.listing)
        else:
            self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class Harness(unittest.TestCase):
    def setUp(self):
        self.sql = []
        self.live = False
        self.listing = []
        owner = self

        class Conn:
            def commit(self): pass

        @contextmanager
        def get_db(commit=False): yield Conn(), FakeCursor(owner)

        app = Flask(__name__)
        app.register_blueprint(summary_lab.create_blueprint(get_db))
        self.client = app.test_client()

    def post(self, body):
        return self.client.post('/api/summary-lab/archive', json=body)

    def statements(self, needle):
        return [sql for sql, _ in self.sql if needle in sql]


class ArchiveTests(Harness):
    def test_archiving_stamps_a_date_and_never_deletes(self):
        response = self.post({'ids': ['a', 'b']})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['archived'], ['a', 'b'])
        self.assertTrue(self.statements('SET archived_at=NOW()'))
        self.assertFalse([s for s in self.statements('') if 'DELETE' in s])

    def test_a_live_worker_blocks_the_archive_so_paid_work_cannot_be_hidden(self):
        self.live = True
        response = self.post({'ids': ['a']})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json['running'], ['a'])
        self.assertIn('Stop the running experiment', response.json['error'])
        self.assertFalse(self.statements('SET archived_at=NOW()'))

    def test_restore_brings_an_archived_experiment_back(self):
        response = self.post({'ids': ['a'], 'restore': True})
        self.assertEqual(response.json['restored'], ['a'])
        self.assertTrue(self.statements('SET archived_at=NULL'))
        # Restoring does not need the row to be idle, so it never takes the lock.
        self.assertFalse(self.statements('pg_try_advisory_lock'))

    def test_repeated_ids_are_collapsed_before_the_lock_is_taken(self):
        self.post({'ids': ['a', 'a', 'b']})
        self.assertEqual(len(self.statements('pg_try_advisory_lock')), 2)

    def test_malformed_requests_are_refused_without_touching_the_database(self):
        for body in ({}, {'ids': []}, {'ids': 'a'}, {'ids': [1]}, {'ids': ['']},
                     {'ids': ['x' * 101]}, {'ids': ['a'] * 101}):
            self.assertEqual(self.post(body).status_code, 400, body)
        self.assertFalse(self.statements('summary_lab_experiments SET'))
        self.assertFalse(self.statements('pg_try_advisory_lock'))


class ListingTests(Harness):
    def test_the_list_hides_archived_experiments_by_default(self):
        body = self.client.get('/api/summary-lab').json
        self.assertFalse(body['archived'])
        self.assertTrue(self.statements('WHERE archived_at IS NULL'))

    def test_the_archived_view_asks_for_exactly_the_opposite_rows(self):
        body = self.client.get('/api/summary-lab?archived=1').json
        self.assertTrue(body['archived'])
        self.assertTrue(self.statements('WHERE archived_at IS NOT NULL'))

    def test_recovery_never_resumes_an_archived_experiment(self):
        # An archived row left at 'running' must not quietly start spending again.
        source = summary_lab.__file__
        with open(source) as handle:
            text = handle.read()
        sweep = text[text.index('def recover_once'):text.index('def start_recovery')]
        self.assertIn('archived_at IS NULL', sweep)
