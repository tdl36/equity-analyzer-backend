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
