"""Per-section Word export: the Lab reuses the Summaries exporter, safely."""
import unittest
from contextlib import contextmanager

from flask import Flask

import summary_lab

FINISHED = {
    'id': 'e1', 'title': 'MCK Mgmt Meeting', 'summary_id': 's1', 'created_at': None,
    'state': {'sections': {'brief': 'Bottom line.\n\nSecond point.', 'takeaways': 'A takeaway.'}},
}


class ProjectionTests(unittest.TestCase):
    def test_sections_land_in_the_columns_the_exporter_reads(self):
        row = summary_lab.export_row(dict(FINISHED), 'all')
        self.assertIn('Bottom line.', row['brief'])
        self.assertIn('A takeaway.', row['summary'])          # takeaways -> summary
        self.assertEqual(row['raw_notes'], '')                # never ship the transcript
        self.assertEqual(row['source_type'], 'summary-lab')

    def test_blank_paragraphs_do_not_become_empty_boxes(self):
        row = summary_lab.export_row({**FINISHED, 'state': {'sections': {'brief': 'One.\n\n\n\nTwo.'}}}, 'brief')
        self.assertEqual(row['brief'].count('<p>'), 2)

    def test_model_text_is_escaped_rather_than_interpreted(self):
        row = summary_lab.export_row({**FINISHED, 'state': {'sections': {'brief': '<script>x</script>'}}}, 'brief')
        self.assertNotIn('<script>', row['brief'])
        self.assertIn('&lt;script&gt;', row['brief'])

    def test_an_unfinished_section_is_refused_instead_of_exported_empty(self):
        with self.assertRaises(ValueError):
            summary_lab.export_row(dict(FINISHED), 'assessment')
        with self.assertRaises(ValueError):
            summary_lab.export_row({**FINISHED, 'state': {'sections': {}}}, 'all')

    def test_an_unknown_section_is_refused(self):
        with self.assertRaises(ValueError):
            summary_lab.export_row(dict(FINISHED), 'transcript')


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.inserted = []
        self.rendered = []
        self.row = dict(FINISHED)
        owner = self

        class Cursor:
            def execute(self, sql, params=None):
                if 'INSERT INTO icloud_export_tasks' in sql: owner.inserted.append(params)
                self.sql = sql
            def fetchone(self):
                return owner.row if 'FROM summary_lab_experiments' in getattr(self, 'sql', '') else None
            def fetchall(self): return []

        class Conn:
            def commit(self): pass

        @contextmanager
        def get_db(commit=False): yield Conn(), Cursor()

        def render(row, sections=None):
            owner.rendered.append(sections)
            return b'PK-docx'

        app = Flask(__name__)
        app.register_blueprint(summary_lab.create_blueprint(get_db, render, lambda t, n=80: t))
        self.client = app.test_client()

    def post(self, section):
        return self.client.post('/api/summary-lab/e1/save-to-icloud', json={'section': section})

    def test_one_section_is_queued_under_a_readable_filename(self):
        body = self.post('takeaways').json
        self.assertEqual(body['filename'], 'MCK Mgmt Meeting - Key Takeaways.docx')
        self.assertEqual(self.rendered, [['takeaways']])
        self.assertEqual(len(self.inserted), 1)
        self.assertEqual(self.inserted[0][1], 's1')       # stays linked to its source Summary
        self.assertEqual(self.inserted[0][5] if len(self.inserted[0]) > 5 else 'queued', 'queued')

    def test_an_unfinished_section_is_a_conflict_and_queues_nothing(self):
        response = self.post('assessment')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.inserted, [])

    def test_a_missing_experiment_is_not_found(self):
        self.row = None
        self.assertEqual(self.post('brief').status_code, 404)
        self.assertEqual(self.inserted, [])

    def test_a_server_without_the_exporter_says_so_instead_of_failing(self):
        app = Flask(__name__)
        app.register_blueprint(summary_lab.create_blueprint(lambda *a, **k: None))
        response = app.test_client().post('/api/summary-lab/e1/save-to-icloud', json={'section': 'brief'})
        self.assertEqual(response.status_code, 503)
