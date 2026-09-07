import tempfile
import unittest
from pathlib import Path
from catalyst_sources import inventory, fingerprint, read_sources


class CatalystSourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, name, text='source evidence'):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def test_nested_sources_are_read_with_relative_names(self):
        self.put('MDT F1Q27 Earnings/broker.txt')
        self.assertEqual(read_sources(self.root)[0]['name'], 'MDT F1Q27 Earnings/broker.txt')

    def test_outputs_hidden_processed_and_external_links_excluded(self):
        self.put('RECAP_MDT.md')
        self.put('nested/SYNTHESIS_MDT.md')
        self.put('Processed/old.txt')
        self.put('.secret.txt')
        self.put('ok.txt')
        (self.root / 'linked.txt').symlink_to('/etc/hosts')
        self.assertEqual([f['name'] for f in inventory(self.root)[0]], ['ok.txt'])

    def test_partial_read_failure_blocks_generation(self):
        self.put('ok.txt')
        self.put('broken.pdf', 'not a pdf')
        with self.assertRaisesRegex(ValueError, 'Source preflight stopped'):
            read_sources(self.root)

    def test_pending_cloud_file_is_actionable(self):
        self.put('.transcript.pdf.icloud')
        with self.assertRaisesRegex(ValueError, 'iCloud download pending'):
            read_sources(self.root)

    def test_no_sources_explains_supported_inputs(self):
        with self.assertRaisesRegex(ValueError, 'subfolders'):
            read_sources(self.root)

    def test_exclusions_support_nested_identity(self):
        self.put('a/broker.txt')
        self.put('b/broker.txt')
        self.assertEqual([f['name'] for f in inventory(self.root, ['a/broker.txt'])[0]], ['b/broker.txt'])

    def test_fingerprint_changes_on_rename_without_count_or_mtime_change(self):
        p = self.put('old.txt')
        before = fingerprint(inventory(self.root)[0])
        p.rename(self.root / 'new.txt')
        self.assertNotEqual(before, fingerprint(inventory(self.root)[0]))

    def test_legacy_formats_explicitly_block(self):
        self.put('old.xls')
        self.assertIn('convert', inventory(self.root)[1][0])

    def test_empty_text_is_not_evidence(self):
        self.put('blank.txt', '   ')
        with self.assertRaisesRegex(ValueError, 'no extractable text'):
            read_sources(self.root)
