"""Run directly: .venv/bin/python tests/unit/test_charlie_collector.py."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PyPDF2 import PdfWriter
from charlie_collector import Collector, originals, source_url


def pdf(width=72):
    writer = PdfWriter()
    writer.add_blank_page(width=width, height=72)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stocks = self.root / "STOCKS"
        (self.stocks / "DE").mkdir(parents=True)
        self.collector = Collector(self.root / "state", self.stocks)
        self.run = self.collector.create(["DE"], "2026-08-07", "2026-09-06")["id"]
        self.file = self.root / "earnings.pdf"
        self.file.write_bytes(pdf())

    def tearDown(self):
        self.collector.db.close()
        self.temp.cleanup()

    def stage(self):
        return self.collector.stage(self.run, "DE", "transcript", self.file,
                                    "https://research.alpha-sense.com/doc/example")["documents"][0]

    def test_resume_and_idempotent_handoff(self):
        doc = self.stage()
        self.collector.db.close()
        self.collector = Collector(self.root / "state", self.stocks)
        first = self.collector.handoff(doc["id"])
        second = self.collector.handoff(doc["id"])
        self.assertEqual(first["destination"], second["destination"])
        self.assertEqual(Path(first["destination"]).parent, self.stocks / "DE")
        self.assertEqual(Path(first["destination"]).read_bytes(), self.file.read_bytes())
        self.assertEqual(len(list(self.stocks.rglob("*.pdf"))), 1)
        self.collector.finish(self.run, "DE", "transcript", 1)

    def test_verified_document_deduplicates_download_timestamps_but_not_revisions(self):
        from reportlab.pdfgen.canvas import Canvas
        def report(stamp, revision='original'):
            stream = io.BytesIO(); canvas = Canvas(stream)
            canvas.drawString(30, 750, ('Research contents ' * 12) + revision)
            canvas.drawString(30, 700, 'reader@example.com - Research Team - ' + stamp)
            canvas.save(); return stream.getvalue()
        url = 'https://research.alpha-sense.com/company/TK1/summary?docid=ASR-example'
        self.file.write_bytes(report('07/09/2026 01:09 AM UTC'))
        self.collector.stage(self.run, 'DE', 'broker-report', self.file, url)
        self.file.write_bytes(report('07/09/2026 01:36 AM UTC'))
        self.collector.stage(self.run, 'DE', 'broker-report', self.file, url)
        self.assertEqual(len(self.collector.status(self.run)['documents']), 1)
        self.file.write_bytes(report('07/09/2026 01:36 AM UTC', 'revised'))
        with self.assertRaises(ValueError):
            self.collector.stage(self.run, 'DE', 'broker-report', self.file, url)
        other = url.replace('ASR-example', 'ASR-other')
        self.collector.stage(self.run, 'DE', 'broker-report', self.file, other)
        self.assertEqual(len(self.collector.status(self.run)['documents']), 2)

    def test_existing_library_duplicate(self):
        old = self.stocks / "DE" / "existing.pdf"
        old.write_bytes(self.file.read_bytes())
        doc = self.stage()
        result = self.collector.handoff(doc["id"])
        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(result["destination"], str(old))

    def test_filename_collision_preserves_both(self):
        first = self.collector.handoff(self.stage()["id"])
        self.file.write_bytes(pdf(99))
        self.collector.stage(self.run, "DE", "transcript", self.file,
                             "https://research.alpha-sense.com/doc/second")
        second = self.collector.status(self.run)["documents"][-1]
        self.collector.handoff(second["id"])
        self.assertEqual(len(list(self.stocks.rglob("*.pdf"))), 2)
        self.assertEqual(Path(first["destination"]).read_bytes(), pdf())

    def test_no_completion_with_unprocessed_original(self):
        self.stage()
        with self.assertRaises(ValueError):
            self.collector.finish(self.run, "DE", "transcript", 1)
        with self.assertRaises(ValueError):
            self.collector.finish(self.run, "DE", "transcript", 0)

    def test_auth_does_not_discard_review_progress(self):
        self.stage()
        self.collector.auth(self.run, True)
        tasks = self.collector.status(self.run)["tasks"]
        self.assertEqual({r["status"] for r in tasks}, {"needs_auth", "review"})
        with self.assertRaises(ValueError):
            self.collector.finish(self.run, "DE", "broker-report", 0)
        self.collector.auth(self.run, False)
        self.collector.finish(self.run, "DE", "broker-report", 0)

    def test_tampered_staged_file_stops_handoff(self):
        doc = self.stage()
        Path(doc["staged"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.collector.handoff(doc["id"])
        self.assertFalse(list(self.stocks.rglob("*.pdf")))

    def test_zip_paths_and_invalid_pdf_rejected_before_staging(self):
        archive = self.root / "download.zip"
        for name, content in [("../escape.pdf", pdf()), ("fake.pdf", b"<html>Login</html>")]:
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("good.pdf", pdf())
                z.writestr(name, content)
            with self.assertRaises(ValueError):
                self.collector.stage(self.run, "DE", "transcript", archive,
                                     "https://research.alpha-sense.com/search")
            self.assertFalse(self.collector.status(self.run)["documents"])

    def test_valid_nested_zip_and_repeated_stage(self):
        archive = self.root / "download.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("originals/call.pdf", pdf())
        self.assertEqual(len(originals(archive)), 1)
        self.stage()
        self.stage()
        self.assertEqual(len(self.collector.status(self.run)["documents"]), 1)

    def test_reject_credential_urls_and_unknown_company(self):
        for url in ["https://research.alpha-sense.com/login", "https://research.alpha-sense.com/?token=secret",
                    "https://evil.test/document"]:
            with self.assertRaises(ValueError):
                source_url(url)
        with self.assertRaises(ValueError):
            self.collector.create(["../DE"], "2026-08-07", "2026-09-06")

    def test_notifications_deduplicate_and_incomplete_digest_is_blocked(self):
        messages = []
        self.collector.notify(self.run, "auth", sender=messages.append)
        self.collector.notify(self.run, "auth", sender=messages.append)
        self.assertEqual(len(messages), 1)
        self.assertIn("Do not reply with", messages[0])
        with self.assertRaises(ValueError):
            self.collector.notify(self.run, "digest", sender=messages.append)
        for kind in ("transcript", "broker-report"):
            self.collector.finish(self.run, "DE", kind, 0)
        self.collector.notify(self.run, "digest", sender=messages.append)
        self.assertEqual(len(messages), 2)

    def test_reference_only_cannot_enter_ai_pipeline(self):
        result = self.collector.stage(self.run, 'DE', 'broker-report', self.file,
                                      'https://research.alpha-sense.com/doc/restricted', usage='reference_only')
        doc = result['documents'][0]
        with self.assertRaises(ValueError):
            self.collector.handoff(doc['id'])
        self.assertFalse(list(self.stocks.rglob('*.pdf')))
        result = self.collector.finish(self.run, 'DE', 'broker-report', 1)
        self.assertEqual(result['tasks'][0]['status'],'complete_with_exceptions')

    def test_observation_does_not_mark_search_complete(self):
        result = self.collector.observe(self.run,'DE','broker-report',
                                       'https://research.alpha-sense.com/search',45,'Source restrictions visible')
        self.assertEqual(result['observations'][0]['result_count'],45)
        self.assertEqual(result['tasks'][0]['status'],'pending')
        from collector_status import snapshot
        self.stage()
        status = snapshot(self.root / 'state')
        self.assertNotIn('staged',status['runs'][0]['documents'][0])


if __name__ == "__main__":
    unittest.main()
