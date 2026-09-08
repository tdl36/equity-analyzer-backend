import unittest,base64
from contextlib import contextmanager
from unittest.mock import Mock
from manual_meeting_sources import freeze
class SourceTests(unittest.TestCase):
    def db(self,rows):
        cur=Mock();cur.fetchall.return_value=rows
        @contextmanager
        def db(**kwargs):yield None,cur
        return db
    def test_uses_server_text_and_hash_not_client_substitution(self):
        row={'id':1,'filename':'original.txt','doc_type':'report','file_data':None,'extracted_text':'Actual saved text'}
        doc=freeze(self.db([row]),5,[{'id':1,'extractedText':'Injected text','filename':'fake.txt'}])[0]
        self.assertEqual(doc['filename'],'original.txt');self.assertEqual(doc['extractedText'],'Actual saved text');self.assertTrue(doc['textSha256'])
    def test_foreign_or_missing_document_is_rejected(self):
        with self.assertRaises(ValueError):freeze(self.db([]),5,[{'id':1}])
    def test_duplicate_names_rejected_before_model_work(self):
        rows=[{'id':i,'filename':'same.txt','doc_type':'report','file_data':None,'extracted_text':'Saved source'} for i in (1,2)]
        with self.assertRaises(ValueError):freeze(self.db(rows),5,[{'id':1},{'id':2}])
