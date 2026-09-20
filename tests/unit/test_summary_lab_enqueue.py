"""Safe Summary Lab auto-fanout tests with an in-memory cursor."""
from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import patch

from summary_lab import create_blueprint


class Cursor:
    def __init__(self, store):
        self.store = store
        self.rowcount = 0
        self.one = None

    def execute(self, sql, args=()):
        compact = ' '.join(sql.split()).lower()
        self.rowcount = 0
        self.one = None
        if 'select title,raw_notes' in compact and 'from meeting_summaries' in compact:
            self.one = {
                'title': 'CAH CFO meeting', 'raw_notes': 'Complete transcript.',
                'brief': 'Original brief', 'summary': 'Original takeaways',
                'questions': 'Original questions', 'assessment': 'Original assessment',
                'meeting_summary': 'Original meeting', 'korean_takeaways': '',
            }
        elif compact.startswith('insert into summary_lab_experiments') and 'true,true' in compact:
            key = (args[9], args[3], args[6], args[10])
            if key not in self.store:
                self.store[key] = {'id': args[0], 'status': 'queued'}
                self.rowcount = 1
        elif 'select id,status from summary_lab_experiments' in compact:
            key = (args[0], args[1], args[2], args[3])
            self.one = self.store[key]
        elif compact.startswith('update summary_lab_experiments'):
            self.rowcount = 1
        elif compact.startswith('insert into summary_lab_experiments'):
            self.rowcount = 1

    def fetchone(self):
        return self.one

    def fetchall(self):
        return []


class FakeThread:
    starts = []

    def __init__(self, target=None, args=(), **_kwargs):
        self.target = target
        self.args = args

    def start(self):
        self.starts.append(self.args)


class SummaryLabEnqueueTests(TestCase):
    def test_automatic_fanout_is_idempotent_and_launches_once(self):
        store = {}

        @contextmanager
        def get_db(**_kwargs):
            yield None, Cursor(store)

        FakeThread.starts = []
        with patch('summary_lab.threading.Thread', FakeThread):
            bp = create_blueprint(get_db)
            first = bp.enqueue('summary-1', 'test-key', automatic=True)
            second = bp.enqueue('summary-1', 'test-key', automatic=True)
        self.assertEqual(first, second)
        self.assertEqual(len(FakeThread.starts), 1)
