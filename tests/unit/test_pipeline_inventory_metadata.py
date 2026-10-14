"""Exercise the real bounded inventory handler without booting the application."""
import ast
import json
import unittest
from contextlib import contextmanager
from datetime import date,datetime
from pathlib import Path
from flask import Flask,jsonify

class InventoryDates(unittest.TestCase):
    def test_text_date_and_missing_publication_dates(self):
        tree=ast.parse(Path('app_v3.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pipeline_documents')
        node.decorator_list=[]
        for publication,expected in [('2026-10-01','2026-10-01'),(date(2026,10,1),'2026-10-01'),(None,None),('unverified',None)]:
            with self.subTest(publication=publication):
                class Cursor:
                    def execute(self,sql,args):self.research='research_document_files' in sql
                    def fetchall(self):
                        if self.research:return [dict(id=2,filename='report.pdf',created_at=datetime(2026,10,1),published_date=publication)]
                        return []
                    def fetchone(self):return None
                @contextmanager
                def db():yield None,Cursor()
                ns={'get_db':db,'jsonify':jsonify,'json':json}
                exec(compile(ast.Module(body=[node],type_ignores=[]),'inventory-handler','exec'),ns)
                with Flask('inventory-test').app_context():
                    response=ns['pipeline_documents']('ABT')
                    self.assertEqual(response.status_code,200)
                    self.assertEqual(response.json['documents'][0]['publishedDate'],expected)
