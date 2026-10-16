"""Isolated agent scheduling; no watched folders, APIs, commands or paid jobs."""
import ast
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock

TREE = ast.parse(Path('charlie_local_agent.py').read_text())


def loop(sync, log=None):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'collection_control_loop')
    scope = {'sync_collection_control': sync, 'log': log or Mock()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<isolated-control-loop>', 'exec'), scope)
    return scope['collection_control_loop']


class CollectionControlLoopTests(unittest.TestCase):
    def test_slow_file_lane_does_not_block_control_and_shutdown_wakes_wait(self):
        stopped = threading.Event(); synced = threading.Event(); file_release = threading.Event()
        file_started = threading.Event()
        def file_lane():
            file_started.set();file_release.wait(2)
        files = threading.Thread(target=file_lane)
        control = threading.Thread(target=loop(synced.set), args=(stopped, 60))
        try:
            files.start();self.assertTrue(file_started.wait(1));control.start()
            self.assertTrue(synced.wait(1))
            self.assertTrue(files.is_alive())
            stopped.set();control.join(1)
            self.assertFalse(control.is_alive())
        finally:
            stopped.set();file_release.set();files.join(2)
            if control.ident:control.join(2)

    def test_stopped_agent_never_starts_sync(self):
        stop = threading.Event();stop.set();sync = Mock()
        loop(sync)(stop)
        sync.assert_not_called()

    def test_failed_sync_waits_then_retries_without_logging_private_exception(self):
        stop = Mock();stop.is_set.return_value = False
        stop.wait.side_effect = [False, True]
        sync = Mock(side_effect=[ValueError('private token or source body'), None]);log = Mock()
        loop(sync, log)(stop, 60)
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(stop.wait.call_args_list, [((60,),), ((60,),)])
        self.assertEqual(log.warning.call_count, 1)
        self.assertEqual(log.warning.call_args.args[1], 'ValueError')
        self.assertNotIn('private', str(log.warning.call_args))

    def test_slow_sync_never_overlaps_itself_or_catches_up_in_burst(self):
        stop = threading.Event();entered = threading.Event();release = threading.Event()
        calls = []
        def sync():
            calls.append(1);entered.set();release.wait(2)
        thread = threading.Thread(target=loop(sync), args=(stop, .01))
        try:
            thread.start();self.assertTrue(entered.wait(1))
            self.assertEqual(len(calls), 1)
            stop.set();release.set();thread.join(1)
            self.assertFalse(thread.is_alive());self.assertEqual(len(calls), 1)
        finally:
            stop.set();release.set();thread.join(2)

    def test_single_sync_site_and_lane_starts_before_startup_inventory(self):
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'sync_collection_control']
        self.assertEqual(len(calls), 1)
        main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        starts = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == 'collection_thread' and n.func.attr == 'start']
        inventory = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id in ('import_existing_notes', 'import_existing_documents')]
        self.assertEqual(len(starts), 1)
        self.assertTrue(all(starts[0].lineno < n.lineno for n in inventory))
