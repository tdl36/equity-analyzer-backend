import base64
import hashlib
import unittest
from research_originals import verified_original
class OriginalTests(unittest.TestCase):
    def setUp(self):
        self.raw=b'%PDF-1.4 Synthetic exact original\x00\xff'
        self.sha=hashlib.sha256(self.raw).hexdigest()
        self.source={'filename':'original.pdf','originalHash':self.sha}
        self.doc={'filename':'original.pdf','file_data':base64.b64encode(self.raw).decode()}
    def test_exact_bytes_and_frozen_identity_required(self):
        self.assertEqual(verified_original(self.source,self.doc,self.sha),self.raw)
        for source,doc,frozen in [(self.source,self.doc,'a'*64),({**self.source,'originalHash':None},self.doc,self.sha),(self.source,None,self.sha),(self.source,{**self.doc,'filename':'another.pdf'},self.sha),(self.source,{**self.doc,'file_data':'not base64'},self.sha),(self.source,{**self.doc,'file_data':base64.b64encode(b'new revision').decode()},self.sha)]:
            with self.subTest(source=source,document=doc is not None),self.assertRaises(ValueError):verified_original(source,doc,frozen)
