"""Exercise the production scanner with temporary sources and no live API calls."""
import ast
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import Mock


class ManifestTests(unittest.TestCase):
    def test_public_html_and_nested_sources_reach_cloud_manifest(self):
        tree = ast.parse(Path('charlie_local_agent.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'push_file_manifest')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stocks, catalysts = root/'STOCKS', root/'CATALYSTS'
            (stocks/'ABT').mkdir(parents=True)
            event = catalysts/'ABT'/'Meeting'
            (event/'Exhibits').mkdir(parents=True)
            for name in ('filing.htm', 'release.html', 'report.pdf', '.pending', 'download.icloud'):
                (event/name).write_text('source')
            (event/'Exhibits'/'table.tsv').write_text('a\tb')
            network = Mock()
            env = dict(Path=Path, Optional=Optional, datetime=datetime, requests=network,
                       STOCKS_DIR=stocks, CATALYSTS_DIR=catalysts, CHARLIE_API='https://test.invalid',
                       _agent_headers=lambda: {}, log=logging.getLogger(__name__))
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'manifest', 'exec'), env)
            env['push_file_manifest']()
            files = network.post.call_args.kwargs['json']['manifest']['ABT']
            self.assertEqual({f['filename'] for f in files}, {'filing.htm','release.html','report.pdf','table.tsv'})
            self.assertEqual(next(f['folder'] for f in files if f['filename']=='filing.htm'), 'Catalysts/Meeting')
            self.assertEqual(next(f['folder'] for f in files if f['filename']=='table.tsv'), 'Catalysts/Meeting/Exhibits')

    def test_watcher_does_not_dispatch_a_managed_source_revision(self):
        import time
        from catalyst_sources import inventory, fingerprint, record_dispatch
        tree=ast.parse(Path('charlie_local_agent.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_for_catalyst_auto_synth')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);event=root/'ABT'/'Meeting';event.mkdir(parents=True)
            (event/'source.txt').write_text('Evidence')
            record_dispatch(event,fingerprint(inventory(event)[0]),'managed-assignment')
            network=Mock();save=Mock()
            env=dict(CATALYSTS_DIR=root,time=time,_load_catalyst_auto_state=lambda:{},
                     _save_catalyst_auto_state=save,catalyst_inventory=inventory,catalyst_fingerprint=fingerprint,
                     requests=network,log=logging.getLogger(__name__))
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'watcher','exec'),env)
            env['check_for_catalyst_auto_synth']()
            network.post.assert_not_called()
            save.assert_called_once()

    def test_changed_folder_after_prior_attempt_only_proposes(self):
        import time
        from catalyst_sources import inventory, fingerprint
        tree=ast.parse(Path('charlie_local_agent.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_for_catalyst_auto_synth')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);event=root/'UNH'/'Earnings';event.mkdir(parents=True)
            (event/'source.txt').write_text('New evidence')
            # Old source revision already started; new files must not spend again.
            state={'UNH::Earnings':{'last_fired_at':1,'fingerprint':'old'}}
            network=Mock();network.post.return_value.ok=True
            network.post.return_value.json.return_value={'jobId':'proposal','count':1}
            env=dict(CATALYSTS_DIR=root,time=time,_load_catalyst_auto_state=lambda:state,
                     _save_catalyst_auto_state=lambda value:None,catalyst_inventory=inventory,
                     catalyst_fingerprint=fingerprint,requests=network,log=logging.getLogger(__name__),
                     _fetch_catalyst_auto_mode=lambda:{'enabled':True},CATALYST_AUTO_DEBOUNCE_S=0,
                     CHARLIE_API='https://test.invalid',_agent_headers=lambda:{})
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'watcher','exec'),env)
            env['check_for_catalyst_auto_synth']()
            self.assertTrue(state['UNH::Earnings']['automatic_attempted'])
            self.assertTrue(network.post.call_args_list[0].args[0].endswith('/propose-synth'))
            self.assertFalse(any(c.args[0].endswith('/synthesize') for c in network.post.call_args_list))

    def test_corrupt_watcher_state_fails_closed_and_atomic_save_roundtrips(self):
        import json
        tree=ast.parse(Path('charlie_local_agent.py').read_text())
        names={'_load_catalyst_auto_state','_save_catalyst_auto_state'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json'
            env=dict(CATALYST_AUTO_STATE_FILE=path,json=json,log=logging.getLogger(__name__))
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'state','exec'),env)
            for malformed in ('{broken','[]','null'):
                path.write_text(malformed)
                with self.assertRaises(RuntimeError):
                    env['_load_catalyst_auto_state']()
            state={'UNH::Earnings':{'automatic_attempted':True}}
            env['_save_catalyst_auto_state'](state)
            self.assertEqual(env['_load_catalyst_auto_state'](),state)
            self.assertFalse(path.with_suffix('.json.part').exists())

    def test_automatic_timeout_is_not_replayed_and_failed_reservation_never_posts(self):
        import copy
        import time
        from catalyst_sources import inventory, fingerprint
        tree=ast.parse(Path('charlie_local_agent.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_for_catalyst_auto_synth')
        for save_works in (True,False):
            with self.subTest(save_works=save_works), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);event=root/'UNH'/'Earnings';event.mkdir(parents=True)
                (event/'source.txt').write_text('Evidence')
                disk={}
                def save(state):
                    if save_works:
                        disk.clear();disk.update(copy.deepcopy(state))
                network=Mock();network.post.side_effect=TimeoutError('Unknown outcome')
                env=dict(CATALYSTS_DIR=root,time=time,_load_catalyst_auto_state=lambda:copy.deepcopy(disk),
                         _save_catalyst_auto_state=save,catalyst_inventory=inventory,catalyst_fingerprint=fingerprint,
                         requests=network,log=logging.getLogger(__name__),_fetch_catalyst_auto_mode=lambda:{'enabled':True},
                         CATALYST_AUTO_DEBOUNCE_S=0,CHARLIE_API='https://test.invalid',_agent_headers=lambda:{})
                exec(compile(ast.Module(body=[fn],type_ignores=[]),'watcher','exec'),env)
                env['check_for_catalyst_auto_synth']()
                env['check_for_catalyst_auto_synth']()
                urls=[c.args[0] for c in network.post.call_args_list]
                if save_works:
                    self.assertEqual(sum(u.endswith('/synthesize') for u in urls),1)
                    self.assertTrue(urls[-1].endswith('/propose-synth'))
                else:
                    network.post.assert_not_called()
