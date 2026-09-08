import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from catalyst_delivery import save_result, deliver
from pipeline_delivery import digest


class DeliveryTests(unittest.TestCase):
    def test_failed_delivery_retains_output_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = save_result('11111111-1111-1111-1111-111111111111', {'markdown':'paid output'}, tmp)
            self.assertFalse(deliver(p, 'https://test.invalid', {}, post=lambda *a, **k: SimpleNamespace(ok=False)))
            self.assertIn('paid output', p.read_text())
            self.assertFalse(deliver(p, 'https://test.invalid', {}, post=lambda *a, **k: SimpleNamespace(ok=True,json=lambda:{'received':True})))
            self.assertTrue(deliver(p, 'https://test.invalid', {}, post=lambda *a, **k: SimpleNamespace(ok=True,json=lambda:{'received':True,'jobId':p.stem,'resultHash':digest({'markdown':'paid output'})})))
            self.assertFalse(p.exists())

    def test_outbox_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): save_result('../escape', {}, tmp)
            self.assertFalse(list(Path(tmp).iterdir()))
