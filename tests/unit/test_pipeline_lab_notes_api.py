import unittest
import uuid
from flask import Flask, request
from tests.unit import test_pipeline_lab_notes as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from pipeline_lab_notes_api import create_blueprint


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class DraftHTTP(unittest.TestCase):
    setUpClass = classmethod(fixtures.DraftVersions.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.DraftVersions.tearDownClass.__func__)

    def setUp(self):
        fixtures.DraftVersions.setUp(self)
        fixtures.DraftVersions.save(self)
        app = Flask('lab-draft-review')
        app.register_blueprint(create_blueprint(
            self.db, lambda: request.headers.get('X-Test-Identity') == 'yes'))
        self.client = app.test_client()
        self.headers = {'X-Test-Identity': 'yes'}
        self.url = '/api/pipeline-lab/notes/' + self.note
        self.change = {'eventId': str(uuid.uuid4()), 'expectedRevision': 1,
                       'title': 'Edited draft', 'markdown': 'New interpretation [1].'}

    def test_authentication_required_before_read_or_write(self):
        for path in (self.url, self.url + '/history'):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.post(self.url + '/revisions', json=self.change).status_code, 401)
        self.assertEqual(self.client.get(self.url, headers=self.headers).json['revision'], 1)

    def test_edit_history_and_retry_after_later_revision(self):
        route = self.url + '/revisions'
        r = self.client.post(route, headers=self.headers, json=self.change)
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(r.json['revision'], 2)
        later = {**self.change, 'eventId': str(uuid.uuid4()), 'expectedRevision': 2, 'markdown': 'Later'}
        self.assertEqual(self.client.post(route, headers=self.headers, json=later).status_code, 200)
        self.assertTrue(self.client.post(route, headers=self.headers, json=self.change).json['replayed'])
        page = self.client.get(self.url + '/history?limit=2', headers=self.headers)
        self.assertEqual([v['revision'] for v in page.json['versions']], [3, 2])
        self.assertEqual(page.json['nextBefore'], 2)
        self.assertNotIn('markdown', str(page.json))
        self.assertIn('no-store', page.headers['Cache-Control'])
        original = self.client.get(self.url + '?revision=1', headers=self.headers).json
        self.assertEqual(original['markdown'], self.payload['markdown'])
        latest = self.client.get(self.url, headers=self.headers).json
        self.assertEqual(latest['sourcePack'], original['sourcePack'])
        self.assertEqual(latest['qualityIssues'], original['qualityIssues'])
        self.assertFalse(latest['investorApproved'])

    def test_conflicts_validation_and_no_approval_surface(self):
        route = self.url + '/revisions'
        self.client.post(route, headers=self.headers, json=self.change)
        stale = {**self.change, 'eventId': str(uuid.uuid4())}
        self.assertEqual(self.client.post(route, headers=self.headers, json=stale).status_code, 409)
        for extra in ('sourcePack', 'qualityIssues', 'investorApproved'):
            self.assertEqual(self.client.post(route, headers=self.headers,
                                             json={**self.change, extra: []}).status_code, 400)
        for suffix in ('?revision=0', '?revision=1&revision=2', '/history?limit=51', '/history?unknown=x'):
            self.assertEqual(self.client.get(self.url + suffix, headers=self.headers).status_code, 400)
        self.assertEqual(self.client.get(self.url + '?revision=99', headers=self.headers).status_code, 404)
        self.assertEqual(self.client.post(self.url + '/accept', headers=self.headers, json={}).status_code, 404)
        self.assertEqual(self.client.post(route, headers=self.headers,
                                         json={**self.change, 'markdown': 'x' * 2_010_000}).status_code, 413)
        self.assertEqual(self.client.post(route, headers=self.headers, data='{',
                                         content_type='application/json').status_code, 400)
