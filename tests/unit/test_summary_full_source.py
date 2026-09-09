"""Safe prompt regression: no app import, database, or provider requests."""
import ast
from pathlib import Path
import unittest

APP = Path(__file__).resolve().parents[2] / 'app_v3.py'
FUNCTIONS = {'_run_podcast_fullsummary_job', '_run_auto_process_text', '_run_auto_process_audio'}

class FullSourceTests(unittest.TestCase):
    def test_every_source_prompt_preserves_long_unicode_tail(self):
        tree = ast.parse(APP.read_text())
        source = 'Management: 매출 growth remains uncertain.\n' * 15000 + 'FINAL_QA_SENTINEL'
        counts = {}
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name not in FUNCTIONS:
                continue
            prompts = []
            for node in ast.walk(fn):
                if not isinstance(node, ast.JoinedStr):
                    continue
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                if any(isinstance(v, ast.FormattedValue) and isinstance(v.value, (ast.Name, ast.Subscript)) and any(isinstance(n, ast.Name) and n.id in {'transcript', 'text'} for n in ast.walk(v.value)) for v in node.values):
                    env = {name: source if name in {'transcript', 'text'} else '' for name in names}
                    rendered = eval(compile(ast.Expression(node), '<prompt-test>', 'eval'), {'__builtins__': {}}, env)
                    self.assertIn(source, rendered, fn.name)
                    prompts.append(node)
            counts[fn.name] = len(prompts)
        self.assertEqual(counts, {'_run_podcast_fullsummary_job': 4, '_run_auto_process_text': 6, '_run_auto_process_audio': 5})

if __name__ == '__main__':
    unittest.main()
