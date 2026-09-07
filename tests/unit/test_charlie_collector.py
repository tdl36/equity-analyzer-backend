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

    def observe(self, kind, count):
        return self.collector.observe(self.run, 'DE', kind,
            'https://research.alpha-sense.com/search', count, 'Reviewed unique originals in the fixed window')

    def test_full_universe_plan_is_idempotent_and_bounded(self):
        tickers = ['DE'] + ['T' + str(i) for i in range(25)]
        first = self.collector.plan(tickers, '2026-08-07', '2026-09-06')
        self.assertEqual(len(first['createdRuns']), 3)
        self.assertEqual(len(first['newTickers']), 25)
        self.assertEqual(self.collector.plan(tickers, '2026-08-07', '2026-09-06')['createdRuns'], [])
        for run in first['createdRuns']:
            self.assertLessEqual(len(self.collector.status(run)['tasks']), 24)

    def test_event_handoff_uses_catalyst_folder_and_manifest(self):
        folder = self.root / 'CATALYSTS' / 'DE' / 'DE F3Q26 Earnings'
        folder.mkdir(parents=True)
        run = self.collector.create(['DE'], '2026-08-07', '2026-09-06', folder.name)['id']
        doc = self.collector.stage(run, 'DE', 'transcript', self.file, 'https://research.alpha-sense.com/doc/event')['documents'][0]
        dest = self.collector.handoff(doc['id'])['destination']
        self.assertEqual(Path(dest).parent, folder)
        result = self.collector.verify(run, lambda ticker: {'files': [{'filename':Path(dest).name,'folder':'Catalysts/' + folder.name}]})
        self.assertEqual(result['verifications'][0]['visible'], 1)

    def test_event_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.collector.create(['DE'], '2026-08-07', '2026-09-06', '../wrong')

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
        self.observe("transcript", 1)
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
        self.observe("transcript", 1)
        with self.assertRaises(ValueError):
            self.collector.finish(self.run, "DE", "transcript", 1)
        with self.assertRaises(ValueError):
            self.collector.finish(self.run, "DE", "transcript", 0)

    def test_auth_pause_survives_restart_and_requires_fresh_evidence(self):
        doc = self.stage()
        self.observe('transcript', 1)
        self.collector.auth(self.run, True)
        self.collector.auth(self.run, True)  # Repeated pause must preserve review state.
        self.assertEqual({t['status'] for t in self.collector.status(self.run)['tasks']}, {'needs_auth'})
        with self.assertRaises(ValueError): self.stage()
        with self.assertRaises(ValueError): self.observe('transcript', 1)
        self.collector.handoff(doc['id'])  # Local validated handoff needs no browser session.
        self.collector.db.close()
        self.collector = Collector(self.root / 'state', self.stocks)
        self.collector.auth(self.run, False)
        statuses = {t['kind']:t['status'] for t in self.collector.status(self.run)['tasks']}
        self.assertEqual(statuses, {'broker-report':'pending', 'transcript':'review'})
        with self.assertRaises(ValueError): self.collector.finish(self.run,'DE','transcript',1)
        self.observe('transcript',1)
        self.collector.finish(self.run,'DE','transcript',1)
        self.collector.auth(self.run, True)
        self.assertEqual(self.collector.status(self.run)['tasks'][1]['status'],'complete')

    def test_completion_requires_matching_evidence_even_for_empty_results(self):
        with self.assertRaises(ValueError): self.collector.finish(self.run,'DE','broker-report',0)
        self.observe('broker-report',1)
        with self.assertRaises(ValueError): self.collector.finish(self.run,'DE','broker-report',0)
        self.observe('broker-report',0)
        self.collector.finish(self.run,'DE','broker-report',0)
        self.collector.finish(self.run,'DE','broker-report',0)  # Safe retry.
        with self.assertRaises(ValueError): self.observe('broker-report',1)
        with self.assertRaises(ValueError): self.collector.finish(self.run,'DE','broker-report',1)

    def test_interrupted_archive_never_stages_partial_contents(self):
        archive = self.root / 'partial.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('first.pdf',pdf())
            z.writestr('second.pdf',pdf(99))
        intact=archive.read_bytes();archive.write_bytes(intact[:-30])
        with self.assertRaises(zipfile.BadZipFile):
            self.collector.stage(self.run,'DE','transcript',archive,'https://research.alpha-sense.com/search')
        self.assertEqual(self.collector.status(self.run)['documents'],[])
        archive.write_bytes(intact)
        self.collector.stage(self.run,'DE','transcript',archive,'https://research.alpha-sense.com/search')
        self.assertEqual(len(self.collector.status(self.run)['documents']),2)

    def test_interrupted_publish_is_retryable_without_partial_handoff(self):
        from unittest.mock import patch
        doc=self.stage()
        with patch('charlie_collector.os.link',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):self.collector.handoff(doc['id'])
        self.assertEqual(list((self.stocks/'DE').iterdir()),[])
        self.assertEqual(self.collector.status(self.run)['documents'][0]['status'],'staged')
        self.collector.handoff(doc['id'])
        self.assertEqual(len(list((self.stocks/'DE').iterdir())),1)

    def test_late_restriction_and_out_of_window_metadata_stop_for_review(self):
        doc=self.collector.handoff(self.stage()['id'])
        with self.assertRaises(ValueError):
            self.collector.stage(self.run,'DE','transcript',self.file,
                'https://research.alpha-sense.com/search',usage='reference_only')
        self.assertEqual(self.collector.status(self.run)['documents'][0]['usage'],'research')
        self.assertTrue(Path(doc['destination']).exists())
        with self.assertRaises(ValueError):
            self.collector.stage(self.run,'DE','broker-report',self.file,
                'https://research.alpha-sense.com/search',published='2026-07-01')

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
            self.observe(kind,0)
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
        self.observe('broker-report',1)
        result = self.collector.finish(self.run, 'DE', 'broker-report', 1)
        self.assertEqual(result['tasks'][0]['status'],'complete_with_exceptions')

    def test_manifest_verification_binds_to_handoff_and_redacts_paths(self):
        from collector_status import snapshot
        doc = self.collector.handoff(self.stage()['id'])
        manifest = {'files': [{'filename': Path(doc['destination']).name, 'folder': 'main'}]}
        result = self.collector.verify(self.run, lambda ticker: manifest)
        self.assertEqual(result['verifications'][0]['visible'], 1)
        view = snapshot(self.root / 'state')['runs'][0]
        self.assertTrue(view['verifications'][0]['current'])
        self.assertEqual(view['documents'][0]['destinationFolder'], 'STOCKS/DE/')
        self.assertNotIn(str(self.root), str(view))
        self.file.write_bytes(pdf(99))
        self.collector.stage(self.run, 'DE', 'broker-report', self.file,
                             'https://research.alpha-sense.com/search')
        self.assertFalse(snapshot(self.root / 'state')['runs'][0]['verifications'][0]['current'])
        result = self.collector.verify(self.run, lambda ticker: manifest)
        self.assertEqual(result['verifications'][0]['expected'], 2)
        self.assertEqual(result['verifications'][0]['visible'], 1)
        Path(doc['destination']).write_bytes(pdf(99))
        self.assertEqual(self.collector.verify(self.run, lambda ticker: manifest)['verifications'][0]['visible'], 0)

    def test_failed_manifest_check_keeps_previous_evidence_and_hides_error(self):
        self.collector.verify(self.run, lambda ticker: {'files': []})
        def failure(ticker):
            raise RuntimeError('sensitive credentials must not be returned')
        result = self.collector.verify(self.run, failure)
        self.assertNotIn('sensitive', str(result))
        from collector_status import snapshot
        self.assertEqual(len(snapshot(self.root / 'state')['runs'][0]['verifications']), 1)

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
