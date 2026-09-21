"""The transcriber comparison must detect the defect it exists to measure."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('compare_transcription', 'scripts/compare_transcription.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

# The MCK pattern: one id holds management's voice and the analyst's questions.
MERGED = """Speaker 1: Uh
Speaker 2: just as a reminder, we are not disclosing any new material information.
Speaker 1: I would first say that we have generally positive macro trends this year.
Speaker 1: Do you feel as confident about your retail capability as you do on specialty?
Speaker 1: We think the 101 implementation lands fine, and we know the 71 states matter.
Speaker 3: Sure, and we expect roughly 300 bps of pressure.
"""

SEPARATED = """Analyst: Do you feel as confident about your retail capability as you do on specialty?
Management: I would first say that we have generally positive macro trends this year.
Management: We think the 101 implementation lands fine, and we know the 71 states matter.
Analyst: So how do you see the 71 states playing out?
Management: We expect roughly 300 bps of pressure, um, you know, across the book.
"""


class RoleSeparationTests(unittest.TestCase):
    def test_a_speaker_holding_both_roles_is_reported(self):
        result = compare.score(MERGED, 'gemini')
        self.assertIn('Speaker 1', result['role_collisions'])

    def test_properly_separated_speakers_report_no_collision(self):
        self.assertEqual(compare.score(SEPARATED, 'openai')['role_collisions'], [])

    def test_each_speaker_is_counted_separately(self):
        per = compare.score(MERGED, 'gemini')['per_speaker']
        self.assertEqual(per['Speaker 1']['questions'], 1)
        self.assertGreaterEqual(per['Speaker 1']['management'], 2)


class FidelityTests(unittest.TestCase):
    """A transcriber that tidies its output would look 'cleaner' and be worse."""

    def test_bare_digit_dates_are_counted_not_normalised_away(self):
        self.assertGreaterEqual(compare.score(MERGED, 'x')['digit_dates'], 2)

    def test_disfluencies_are_measured(self):
        self.assertGreater(compare.score(SEPARATED, 'x')['disfluencies_per_1k'], 0)
        self.assertEqual(compare.score('Management: Clean prose only.', 'x')['disfluencies_per_1k'], 0)

    def test_figures_are_counted(self):
        self.assertGreaterEqual(compare.score(MERGED, 'x')['figures'], 1)


class ParsingTests(unittest.TestCase):
    def test_turns_and_speakers_are_recovered(self):
        result = compare.score(MERGED, 'x')
        self.assertEqual(result['turns'], 6)
        self.assertEqual(result['speakers'], 3)

    def test_an_unlabelled_transcript_still_scores(self):
        result = compare.score('Just a wall of text with no speaker labels at all.', 'x')
        self.assertEqual(result['speakers'], 1)
        self.assertEqual(result['turns'], 1)

    def test_turn_quality_flags_mid_sentence_boundaries(self):
        result = compare.score('Speaker 1: Uh\nSpeaker 2: just as a reminder, we are not\n', 'x')
        self.assertEqual(result['turns_starting_lowercase'], 1)
        self.assertGreaterEqual(result['fragment_turns'], 1)

    def test_it_never_writes_anything_without_being_asked(self):
        source = Path('scripts/compare_transcription.py').read_text()
        self.assertNotIn('INSERT', source)
        self.assertNotIn('UPDATE', source)
        # The only write is the transcript the caller names with --out.
        self.assertEqual(source.count("open(out_path, 'w')"), 1)


class NamedEntityTests(unittest.TestCase):
    """The metrics passed a transcript that had turned "no new material
    information" into "no team internal plans or mission". Counting words
    cannot see a name replaced by another name, so measure the names."""

    BASE = 'Speaker 1: We think ClarusOne and Keytruda help, and Optum agrees.'

    def test_a_dropped_name_is_reported(self):
        kept, lost = compare.retention(self.BASE, 'A: We think it helps, and they agree.')
        self.assertIn('ClarusOne', lost)
        self.assertIn('Keytruda', lost)
        self.assertLess(kept, 50)

    def test_a_faithful_transcript_keeps_its_names(self):
        kept, lost = compare.retention(self.BASE, 'A: We think ClarusOne and Keytruda help, Optum agrees.')
        self.assertEqual(lost, [])
        self.assertEqual(kept, 100.0)

    def test_case_differences_do_not_count_as_loss(self):
        _, lost = compare.retention(self.BASE, 'A: we think clarusone and keytruda help, optum agrees.')
        self.assertEqual(lost, [])

    def test_speaker_labels_are_not_mistaken_for_names(self):
        kept, lost = compare.retention('Speaker 1: Hello there.', 'A: Hello there.')
        self.assertNotIn('Speaker', lost or [])

    def test_a_baseline_without_names_reports_nothing_rather_than_zero(self):
        kept, lost = compare.retention('a: all lowercase here, no names at all.', 'b: same again.')
        self.assertIsNone(kept)
