"""A different provider covers the case where every Gemini attempt fails."""
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app_v3

SOURCE = Path('app_v3.py').read_text()


def segments(*pairs):
    return SimpleNamespace(
        segments=[SimpleNamespace(speaker=who, text=said) for who, said in pairs],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50))


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.audio = Path(self.id().replace('.', '_') + '.mp3')
        self.audio.write_bytes(b'x' * 1024)
        self.addCleanup(lambda: self.audio.exists() and self.audio.unlink())

    def transcribe(self, result, size_override=None, reject=()):
        created = {}

        def create(**kw):
            # The pinned 1.x SDK rejects diarized_json client-side, before any
            # request is sent. Simulate that rather than assuming it works.
            if kw.get('response_format') in reject:
                raise TypeError(f"Unexpected audio response format: {kw['response_format']}")
            created.update(kw)
            return result

        class Audio:
            transcriptions = SimpleNamespace(create=create)

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'k'}), \
             patch.object(app_v3.openai, 'OpenAI', lambda **kw: SimpleNamespace(audio=Audio())), \
             patch.object(app_v3, 'record_llm_usage', lambda *a, **k: None):
            text = app_v3._transcribe_with_openai(str(self.audio), 'job1')
        return text, created

    def test_consecutive_turns_by_one_speaker_are_joined(self):
        text, _ = self.transcribe(segments(('A', 'We grew five percent.'), ('A', 'Mostly volume.'),
                                           ('B', 'On which segment?')))
        self.assertEqual(text, 'A: We grew five percent. Mostly volume.\n\nB: On which segment?')

    def test_it_asks_for_speaker_labels_the_gemini_path_cannot_give(self):
        _, created = self.transcribe(segments(('A', 'Hello.')))
        self.assertEqual(created['model'], 'gpt-4o-transcribe-diarize')
        self.assertEqual(created['response_format'], 'diarized_json')
        self.assertEqual(created['chunking_strategy'], 'auto')

    def test_plain_text_is_accepted_when_no_segments_come_back(self):
        text, _ = self.transcribe(SimpleNamespace(segments=[], text='Flat transcript.', usage=None))
        self.assertEqual(text, 'Flat transcript.')

    def test_empty_speech_is_an_error_not_an_empty_transcript(self):
        with self.assertRaises(RuntimeError):
            self.transcribe(segments(('A', '   ')))

    def test_a_file_over_the_documented_cap_is_refused_before_upload(self):
        self.audio.write_bytes(b'x' * 16)
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'k'}), \
             patch.object(app_v3, 'OPENAI_TRANSCRIBE_MAX_BYTES', 8):
            with self.assertRaises(RuntimeError) as caught:
                app_v3._transcribe_with_openai(str(self.audio), 'job1')
        self.assertIn('25MB', str(caught.exception))

    def test_a_missing_key_or_file_fails_clearly(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            with self.assertRaises(RuntimeError):
                app_v3._transcribe_with_openai(str(self.audio), 'job1')
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'k'}):
            with self.assertRaises(RuntimeError):
                app_v3._transcribe_with_openai('/no/such/file.mp3', 'job1')


class WiringTests(unittest.TestCase):
    """The chunk used to be deleted from disk the moment it was read, which
    would have left the fallback with nothing to send."""

    def test_the_chunk_survives_until_its_transcription_finishes(self):
        body = SOURCE[SOURCE.index('# Read chunk from disk'):SOURCE.index('transcript_text = ')]
        read_at = body.index("cf.read()")
        call_at = body.index('_transcribe_audio_content(')
        remove_at = body.index('os.remove(chunk_path)')
        self.assertLess(read_at, call_at)
        self.assertLess(call_at, remove_at, 'the chunk is removed before it can be resent')

    def test_both_transcription_paths_pass_a_path_for_the_fallback(self):
        calls = [c for c in re.findall(r'(?<!def )_transcribe_audio_content\(client[^)]*\)', SOURCE)
                 if 'label=""' not in c]   # drop the definition itself
        self.assertEqual(len(calls), 2, calls)
        for call in calls:
            self.assertIn('audio_path=', call)

    def test_the_fallback_runs_only_after_every_gemini_attempt(self):
        body = SOURCE[SOURCE.index('def _transcribe_audio_content'):]
        body = body[:body.index('All models failed')]
        self.assertIn('_transcribe_with_openai', body)
        self.assertLess(body.index('for model_name in models_to_try'),
                        body.index('_transcribe_with_openai'))

    def test_a_failed_fallback_still_reports_the_original_failure(self):
        body = SOURCE[SOURCE.index('def _transcribe_audio_content'):]
        tail = body[body.index('_transcribe_with_openai'):body.index('All models failed') + 40]
        self.assertIn('except Exception', tail)
        self.assertIn('raise Exception', tail)


class SdkCompatibilityTests(FallbackTests):
    """requirements.txt pins openai<2, which does not know diarized_json. The
    fallback exists for resilience, so an old SDK must cost the speaker labels,
    not the transcript."""

    def test_an_sdk_without_diarization_still_returns_a_transcript(self):
        plain = SimpleNamespace(segments=[], text='A plain transcript.', usage=None)
        text, created = self.transcribe(plain, reject=('diarized_json',))
        self.assertEqual(text, 'A plain transcript.')
        self.assertEqual(created['response_format'], 'json')
        self.assertEqual(created['model'], app_v3.OPENAI_TRANSCRIBE_FALLBACK_MODEL)

    def test_diarization_is_preferred_when_the_sdk_supports_it(self):
        _, created = self.transcribe(segments(('A', 'Hello.')))
        self.assertEqual(created['response_format'], 'diarized_json')
        self.assertEqual(created['model'], app_v3.OPENAI_TRANSCRIBE_MODEL)

    def test_both_attempts_failing_raises_rather_than_returning_nothing(self):
        with self.assertRaises(RuntimeError):
            self.transcribe(segments(('A', 'Hello.')), reject=('diarized_json', 'json'))
