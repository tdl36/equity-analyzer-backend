import base64,subprocess,unittest
from unittest.mock import patch
from meeting_pdf_worker import extract_original
class BoundedPdfTests(unittest.TestCase):
    def test_timeout_becomes_actionable_failure(self):
        with patch('meeting_pdf_worker.subprocess.run',side_effect=subprocess.TimeoutExpired('extract',40)):
            with self.assertRaisesRegex(ValueError,'time/resource limit'):extract_original(base64.b64encode(b'fixture').decode())
    def test_excessive_size_rejected_without_process(self):
        class Large:
            def __len__(self):return 80_000_001
        with patch('meeting_pdf_worker.subprocess.run') as run:
            with self.assertRaises(ValueError):extract_original(Large())
            run.assert_not_called()
