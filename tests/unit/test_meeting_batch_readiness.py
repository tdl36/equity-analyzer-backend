import copy
from datetime import datetime, timezone
import unittest
from meeting_batch_readiness import summarize


class BatchReadinessTests(unittest.TestCase):
    def ready(self):
        return {'command': {'id': 'one', 'ticker': 'ABT', 'created_at': '2026-09-08T12:00:00Z'},
                'preparation': {'status': 'done'}, 'meetingId': 1, 'documents': [{'filename': 'release.pdf'}],
                'versions': [{'created_at': '2026-09-08T12:10:00Z'}, {'created_at': '2026-09-08T12:20:00Z'}]}

    def test_one_failed_company_prevents_batch_ready(self):
        one = self.ready(); two = copy.deepcopy(one)
        two['command'].update(id='two', ticker='MDT')
        two['preparation'] = {'status': 'failed', 'error': 'Source support failed'}
        report = summarize([one, two])
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['ready'], 1)
        self.assertEqual(report['companies'][1]['issue'], 'Source support failed')

    def test_first_saved_pack_excludes_later_revision_delay(self):
        report = summarize([self.ready()], datetime(2026, 9, 8, 13, tzinfo=timezone.utc))
        self.assertEqual(report['companies'][0]['secondsToFirstSavedPack'], 600)
        self.assertEqual(report['companies'][0]['secondsSinceSubmitted'], 3600)

    def test_missing_saved_output_and_duplicate_command_are_not_success(self):
        one = self.ready(); one['versions'] = []
        self.assertEqual(summarize([one])['status'], 'incomplete')
        self.assertEqual(summarize([])['status'], 'incomplete')
        with self.assertRaises(ValueError): summarize([one, one])
