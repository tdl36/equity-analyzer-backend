import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_lab_local import identity,read_original

class ExactLocalOriginal(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name).resolve();self.stocks=root/'STOCKS';self.catalysts=root/'CATALYSTS'
        (self.stocks/'ABT'/'Reports').mkdir(parents=True);(self.catalysts/'ABT'/'Meeting'/'Exhibits').mkdir(parents=True)
        (self.stocks/'ABT'/'report.txt').write_bytes(b'main original')
        (self.stocks/'ABT'/'Reports'/'report.txt').write_bytes(b'other original')
        (self.catalysts/'ABT'/'Meeting'/'Exhibits'/'report.txt').write_bytes(b'event original')
    def read(self,folder='main',name='report.txt',**kwargs):
        return read_original(self.stocks,self.catalysts,'ABT','icloud:'+json.dumps([folder,name]),**kwargs)
    def test_exact_folders_and_hash_bound_replay(self):
        for folder,raw in [('main',b'main original'),('Reports',b'other original'),('Catalysts/Meeting/Exhibits',b'event original')]:
            result=self.read(folder);self.assertEqual(base64.b64decode(result['fileData']),raw)
            self.assertEqual(self.read(folder,expected_sha256=result['sha256']),result)
        with self.assertRaisesRegex(ValueError,'frozen'):self.read('Reports',expected_sha256=self.read()['sha256'])
        with self.assertRaisesRegex(ValueError,'unavailable'):self.read('Absent')
    def test_paths_cannot_escape_or_hide_placeholders(self):
        for folder,name in [('../ABT','report.txt'),('main','../report.txt'),('/tmp','report.txt'),('main','.report.txt.icloud'),('Reports//x','report.txt'),('main','report\\x.txt'),('main','~$report.txt')]:
            with self.subTest(folder=folder,name=name),self.assertRaises(ValueError):self.read(folder,name)
        for value in ['icloud:null','icloud:{}','icloud:["main"]','charlie:thesis_1']:
            with self.assertRaises(ValueError):identity(value)
    def test_symlink_files_directories_and_roots_rejected(self):
        (self.stocks/'ABT'/'link.txt').symlink_to(self.stocks/'ABT'/'report.txt')
        (self.stocks/'ABT'/'Alias').symlink_to(self.stocks/'ABT'/'Reports')
        for folder,name in [('main','link.txt'),('Alias','report.txt')]:
            with self.assertRaises(ValueError):self.read(folder,name)
        alias=self.stocks.parent/'alias';alias.symlink_to(self.stocks)
        with self.assertRaises(ValueError):read_original(alias,self.catalysts,'ABT','icloud:["main","report.txt"]')
    def test_mutation_while_reading_rejected_and_original_preserved(self):
        original_read=os.read;changed=False;path=self.stocks/'ABT'/'report.txt'
        def changing(fd,size):
            nonlocal changed
            block=original_read(fd,size)
            if not changed:changed=True;path.write_bytes(b'changed original')
            return block
        with patch('pipeline_lab_local.os.read',side_effect=changing):
            with self.assertRaisesRegex(ValueError,'changed'):self.read()
        self.assertEqual(path.read_bytes(),b'changed original')
    def test_empty_oversized_and_unhydrated_files_rejected(self):
        path=self.stocks/'ABT'/'report.txt';path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'empty'):self.read()
        path.write_bytes(b'abcd')
        with patch('pipeline_lab_local.MAX_FILE',3):
            with self.assertRaisesRegex(ValueError,'exceeds'):self.read()
        original_stat=os.stat
        def dataless(*a,**kw):
            s=original_stat(*a,**kw)
            class Info:
                st_mode=s.st_mode;st_flags=0x40000000
            return Info() if kw.get('dir_fd') is not None else s
        with patch('pipeline_lab_local.os.stat',side_effect=dataless):
            with self.assertRaisesRegex(ValueError,'not downloaded'):self.read()
