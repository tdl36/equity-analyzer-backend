"""Summary sections run concurrently.

Measured on a real meeting: transcription took about four minutes, the five
sections about sixteen, run one after another. The sections do not read each
other, so that wait was pure serialisation.
"""
import ast
import time
import unittest
from pathlib import Path

SOURCE = Path('app_v3.py').read_text()
TREE = ast.parse(SOURCE)


def load(name, namespace):
    fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app_v3.py', 'exec'), namespace)
    return namespace[name]


class Harness:
    """Stubs the model call so the helper can be exercised without providers."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.jobs = {'job-1': {}}
        self.progress = []

    def build(self):
        def call(**kwargs):
            label = kwargs['label']
            delay, result = self.behaviour[label]
            time.sleep(delay)
            if isinstance(result, Exception):
                raise result
            return {'text': result}

        def mirror(job_id):
            self.progress.append(self.jobs[job_id].get('progress'))

        return load('_generate_sections_concurrently', {
            '_call_llm_stream_with_retry': call,
            '_transcription_jobs': self.jobs,
            '_mirror_transcription_state': mirror,
            'print': lambda *a, **k: None,
        })


def spec(key, fatal, label):
    return (key, fatal, dict(label=label, messages=[], system='', tier='standard', max_tokens=10))


class ConcurrencyTests(unittest.TestCase):
    def test_sections_run_at_the_same_time_not_one_after_another(self):
        harness = Harness({f'l{i}': (0.30, f'text{i}') for i in range(4)})
        run = harness.build()
        started = time.monotonic()
        run('job-1', [spec(f'k{i}', False, f'l{i}') for i in range(4)])
        elapsed = time.monotonic() - started
        # Serial would be ~1.2s; concurrent is bounded by the slowest section.
        self.assertLess(elapsed, 0.75, f'sections appear to run serially ({elapsed:.2f}s)')

    def test_results_are_keyed_so_finishing_order_cannot_scramble_them(self):
        # Deliberately finish in reverse order of submission.
        harness = Harness({'slow': (0.35, 'SUMMARY'), 'mid': (0.20, 'ASSESSMENT'), 'fast': (0.02, 'QUESTIONS')})
        run = harness.build()
        out = run('job-1', [spec('summary', True, 'slow'),
                            spec('assessment', False, 'mid'),
                            spec('questions', False, 'fast')])
        self.assertEqual(out['summary'], 'SUMMARY')
        self.assertEqual(out['assessment'], 'ASSESSMENT')
        self.assertEqual(out['questions'], 'QUESTIONS')

    def test_a_non_fatal_section_yields_empty_without_failing_the_job(self):
        harness = Harness({'ok': (0.01, 'kept'), 'bad': (0.01, RuntimeError('provider down'))})
        run = harness.build()
        out = run('job-1', [spec('summary', True, 'ok'), spec('questions', False, 'bad')])
        self.assertEqual(out['summary'], 'kept')
        self.assertEqual(out['questions'], '')

    def test_a_fatal_section_still_fails_the_job(self):
        harness = Harness({'bad': (0.01, RuntimeError('provider down')), 'ok': (0.01, 'kept')})
        run = harness.build()
        with self.assertRaises(Exception) as caught:
            run('job-1', [spec('summary', True, 'bad'), spec('questions', False, 'ok')])
        self.assertIn('summary LLM failed', str(caught.exception))

    def test_one_failure_does_not_discard_the_sections_that_worked(self):
        harness = Harness({'ok': (0.01, 'kept'), 'bad': (0.01, RuntimeError('x'))})
        run = harness.build()
        out = run('job-1', [spec('assessment', False, 'ok'), spec('questions', False, 'bad')])
        self.assertEqual(out['assessment'], 'kept')

    def test_progress_counts_completions_since_order_varies(self):
        harness = Harness({f'l{i}': (0.01, 'x') for i in range(3)})
        run = harness.build()
        run('job-1', [spec(f'k{i}', False, f'l{i}') for i in range(3)])
        self.assertEqual(harness.progress, ['Generated 1 of 3 sections',
                                            'Generated 2 of 3 sections',
                                            'Generated 3 of 3 sections'])


class AudioJobWiringTests(unittest.TestCase):
    def body(self):
        return next(n for n in TREE.body
                    if isinstance(n, ast.FunctionDef) and n.name == '_run_auto_process_audio')

    def test_the_four_independent_sections_go_through_the_concurrent_helper(self):
        calls = [n for n in ast.walk(self.body())
                 if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == '_generate_sections_concurrently']
        self.assertEqual(len(calls), 1)
        keys = {c.value for c in ast.walk(calls[0]) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        for section in ('summary', 'questions', 'assessment', 'meeting_summary'):
            self.assertIn(section, keys, section)

    def test_only_key_takeaways_is_fatal(self):
        src = ast.unparse(self.body())
        self.assertIn("('summary', True,", src)
        for section in ('questions', 'assessment', 'meeting_summary'):
            self.assertIn(f"('{section}', False,", src)

    def test_the_brief_still_runs_after_key_takeaways(self):
        src = ast.unparse(self.body())
        # Since T97 the Brief reuses the classification Key Takeaways makes,
        # so it must not be folded into the concurrent batch.
        self.assertLess(src.index('_generate_sections_concurrently'),
                        src.index('_classified_source_type'))
        self.assertLess(src.index('_classified_source_type'), src.index('audio brief ('))
        self.assertNotIn("'brief'", src.split('_classified_source_type')[0])

    def test_every_section_keeps_its_own_budget_and_tier(self):
        src = ast.unparse(self.body())
        for tokens in ('24576', '2048', '4096', '8192'):
            self.assertIn(tokens, src, tokens)
        self.assertIn("tier='fast'", src)
