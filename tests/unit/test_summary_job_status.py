"""Exercise the production poll route without server startup or database fixtures."""
import ast
import time
import unittest
from pathlib import Path
from contextlib import contextmanager
from flask import Flask, jsonify

class SummaryJobStatusTests(unittest.TestCase):
    def setUp(self):
        self.jobs={};self.persisted=None
        owner=self
        class Cursor:
            def execute(self,*args): pass
            def fetchone(self): return owner.persisted
        @contextmanager
        def get_db(): yield None,Cursor()
        app=Flask(__name__)
        tree=ast.parse(Path('app_v3.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transcribe_audio_status')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'app_v3.py','exec'),{'app':app,'jsonify':jsonify,'_transcription_jobs':self.jobs,'get_db':get_db,'time':time})
        self.client=app.test_client()
    def test_completed_text_job_returns_saved_summary_id(self):
        self.jobs['yt']={'status':'complete','summaryId':'saved-summary'}
        r=self.client.get('/api/transcribe-audio/yt')
        self.assertEqual(r.json['summaryId'],'saved-summary')
        self.assertEqual(r.json['status'],'complete')
    def test_failed_pipeline_returns_actual_failure(self):
        self.jobs['yt']={'status':'failed','error':'Transcript unavailable'}
        self.assertEqual(self.client.get('/api/transcribe-audio/yt').json['error'],'Transcript unavailable')
    def test_restart_reads_completed_mirror(self):
        self.persisted={'status':'complete','summary_id':'saved-summary'}
        r=self.client.get('/api/transcribe-audio/yt')
        self.assertEqual(r.json['summaryId'],'saved-summary')
        self.assertTrue(r.json['persisted'])
    def test_legacy_audio_and_running_states_remain_supported(self):
        self.jobs['audio']={'status':'done','text':'transcript'}
        self.assertEqual(self.client.get('/api/transcribe-audio/audio').json['text'],'transcript')
        self.jobs['yt']={'status':'summarizing','progress':'Generating assessment'}
        self.assertEqual(self.client.get('/api/transcribe-audio/yt').json['progress'],'Generating assessment')
