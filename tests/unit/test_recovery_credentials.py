"""No database or provider calls: credential fallback and recovery dispatch."""
import os
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from recovery_credentials import anthropic_recovery_key
import summary_lab
import summary_comparison


class RecoveryCredentialTests(unittest.TestCase):
    def database(self, value):
        cursor = Mock()
        cursor.fetchone.return_value = value
        @contextmanager
        def get_db(**kwargs):
            yield None, cursor
        return get_db, cursor

    def test_server_key_has_precedence_without_database_access(self):
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ' server-test '}):
            self.assertEqual(anthropic_recovery_key(Mock(side_effect=AssertionError)), 'server-test')

    def test_saved_key_is_read_again_after_rotation_and_removal(self):
        get_db, cur = self.database({'value': ' saved-test '})
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ' '}):
            self.assertEqual(anthropic_recovery_key(get_db), 'saved-test')
            cur.fetchone.return_value = {'value': 'rotated-test'}
            self.assertEqual(anthropic_recovery_key(get_db), 'rotated-test')
            cur.fetchone.return_value = None
            self.assertEqual(anthropic_recovery_key(get_db), '')
        cur.execute.assert_called_with('SELECT value FROM app_settings WHERE key=%s', ('apiKey',))

    def test_missing_or_invalid_settings_do_not_become_credentials(self):
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}):
            for row in (None, {}, {'value': None}, {'value': 123}, {'value': {}}, {'value': ' '}):
                get_db, _ = self.database(row)
                self.assertEqual(anthropic_recovery_key(get_db), '')

    def test_lookup_failure_does_not_start_jobs(self):
        for module in (summary_lab, summary_comparison):
            db = Mock(side_effect=RuntimeError('database unavailable'))
            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}), patch.object(module.threading, 'Thread') as thread:
                bp = module.create_blueprint(db)
                with self.assertRaises(RuntimeError):
                    bp.recover_once()
                thread.assert_not_called()

    def test_lab_dispatches_saved_key_once_without_provider_calls(self):
        cur = Mock()
        cur.fetchone.return_value = {'value': 'saved-test'}
        cur.fetchall.return_value = [{'id': 'lab-test'}]
        @contextmanager
        def get_db(**kwargs): yield None, cur
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}), patch.object(summary_lab.threading, 'Thread') as thread:
            bp = summary_lab.create_blueprint(get_db)
            bp.recover_once()
            bp.recover_once()
            thread.assert_called_once()
            self.assertEqual(thread.call_args.kwargs['args'], ('lab-test', 'saved-test'))

    def test_comparison_uses_saved_key_and_existing_recovery_path(self):
        get_db, cur = self.database({'value': 'saved-test'})
        cur.fetchall.return_value = [{'id': 'comparison-test'}]
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}):
            bp = summary_comparison.create_blueprint(get_db)
            # Replace the closed-over launch, preventing any model/SQL execution.
            recover = bp.recover_once
            cells = dict(zip(recover.__code__.co_freevars, recover.__closure__))
            launch = Mock()
            cells['launch'].cell_contents = launch
            recover()
            launch.assert_called_once_with('comparison-test', 'saved-test', recovery=True)


if __name__ == '__main__': unittest.main()
