"""No app import, scheduler execution, credentials or database connection."""
import ast
from pathlib import Path
import unittest


class SchedulerDriverTests(unittest.TestCase):
    def test_postgres_uses_installed_driver_and_preserves_connection_options(self):
        tree = ast.parse(Path('scheduler.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'jobstore_url')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'scheduler-driver-test', 'exec'), namespace)
        normalize = namespace['jobstore_url']
        for scheme in ('postgres', 'postgresql'):
            url = normalize(scheme + '://synthetic:p%40ss@example.invalid:5432/research?sslmode=require')
            self.assertEqual(url.drivername, 'postgresql+psycopg2')
            self.assertEqual(url.password, 'p@ss')
            self.assertEqual(url.database, 'research')
            self.assertEqual(url.query['sslmode'], 'require')
            self.assertEqual(url.get_dialect().driver, 'psycopg2')
        self.assertEqual(normalize('sqlite:///synthetic.db').drivername, 'sqlite')
        self.assertEqual(normalize('postgresql+psycopg2://example.invalid/research').drivername, 'postgresql+psycopg2')
