import tempfile
import unittest
from pathlib import Path
from recap_checkpoint import Checkpoint
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_resume_only_identical_inputs_and_remove_after_delivery_save(self):
        c=Checkpoint({'sourceHash':'a','instructions':'x'},self.root);c.save(2,'Complete merged draft')
        self.assertEqual(c.load(3)['completed'],2)
        self.assertIsNone(Checkpoint({'sourceHash':'b','instructions':'x'},self.root).load(3))
        self.assertIsNone(Checkpoint({'sourceHash':'a','instructions':'y'},self.root).load(3))
        c.finish();self.assertIsNone(c.load(3))
    def test_corrupt_or_out_of_range_checkpoint_is_not_reused(self):
        c=Checkpoint({},self.root);c.save(3,'Draft');self.assertIsNone(c.load(2));c.path.write_text('{broken');self.assertIsNone(c.load(3))
    def test_atomic_private_files_and_no_empty_result(self):
        c=Checkpoint({},self.root);c.save(1,'Old');c.save(2,'New');self.assertEqual(c.load(2)['markdown'],'New')
        self.assertEqual(c.path.stat().st_mode&0o777,0o600)
        with self.assertRaises(ValueError):c.save(2,'')
        self.assertEqual(c.load(2)['markdown'],'New')
