"""Isolated export/email regression tests; no app startup, database or email delivery."""
import ast
import base64
import io
import re
import unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from flask import Flask, request, jsonify
from summary_bulk import ordered_rows, sections_html, pooled_html

SOURCE = Path(__file__).resolve().parents[2] / 'app_v3.py'
NAMES = {'_html_to_docx_elements', '_generate_summary_docx_bytes', '_generate_summary_pdf_bytes', 'email_summary_section', 'summaries_bulk_export'}
ns = dict(io=io, re=re, base64=base64, request=request, jsonify=jsonify)
for node in ast.parse(SOURCE.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in NAMES:
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)

class Cursor:
    def execute(self, *args): pass
    def fetchall(self): return list(reversed(ROWS))
ROWS = [dict(id=str(i), title=f'Entry {i}', brief='<p>BRIEF_MARKER</p>', meeting_summary='<p>MEETING_MARKER</p>', summary='<p>TAKEAWAY_MARKER</p>', questions='<p>QUESTION_MARKER</p>', assessment='<p>ASSESSMENT_MARKER</p>', source_type='audio', raw_notes='TRANSCRIPT_MARKER', korean_takeaways='KOREAN_MARKER') for i in range(2)]
@contextmanager
def fake_db(): yield None, Cursor()
ns['get_db'] = fake_db
app = Flask(__name__)
app.add_url_rule('/email', view_func=ns['email_summary_section'], methods=['POST'])
app.add_url_rule('/export', view_func=ns['summaries_bulk_export'], methods=['POST'])

class BulkTests(unittest.TestCase):
    def test_order_missing_and_invalid(self):
        self.assertEqual([r['id'] for r in ordered_rows(Cursor(), ['0','1'])], ['0','1'])
        for ids in [['missing'], ['0','0'], '0', []]:
            with self.assertRaises(ValueError): ordered_rows(Cursor(), ids)
    def test_no_truncation_and_escaping(self):
        text = sections_html(dict(source_type='audio', raw_notes='x'*100001+'TAIL<&>'))
        self.assertIn('x'*100001+'TAIL&lt;&amp;&gt;', text)
        self.assertIn('&lt;script&gt;', pooled_html([dict(title='<script>')]))
    def test_documents(self):
        from docx import Document
        from pypdf import PdfReader
        row=ROWS[0]
        doc=Document(io.BytesIO(ns['_generate_summary_docx_bytes'](row, ['brief','meeting'])))
        word='\n'.join(p.text for p in doc.paragraphs)
        pdf=PdfReader(io.BytesIO(ns['_generate_summary_pdf_bytes'](row, ['brief','meeting'])))
        text='\n'.join(p.extract_text() for p in pdf.pages)
        for output in [word,text]:
            self.assertIn('BRIEF_MARKER',output)
            self.assertIn('MEETING_MARKER',output)
            self.assertNotIn('TAKEAWAY_MARKER',output)
    def test_one_combined_email(self):
        payload=dict(summaryIds=['0','1'],email='test@example.com',smtpConfig=dict(gmail_user='test@example.com',gmail_app_password='fake',use_gmail=True))
        with patch('smtplib.SMTP') as smtp:
            response=app.test_client().post('/email',json=payload)
            self.assertEqual(response.status_code,200)
            server=smtp.return_value.__enter__.return_value
            server.send_message.assert_called_once()
            message=server.send_message.call_args.args[0]
            html=message.get_payload()[1].get_payload(decode=True).decode()
            self.assertLess(html.index('Entry 0'),html.index('Entry 1'))
            for field in ['BRIEF','MEETING','TAKEAWAY','QUESTION','ASSESSMENT','TRANSCRIPT','KOREAN']:
                self.assertEqual(html.count(field+'_MARKER'),2)
            payload['summaryIds']=['missing']
            self.assertEqual(app.test_client().post('/email',json=payload).status_code,400)
            server.send_message.assert_called_once()
    def test_export_zip_and_validation(self):
        import zipfile
        response=app.test_client().post('/export',json=dict(summaryIds=['1','0'],format='docx',sections=['brief','meeting']))
        self.assertEqual(response.status_code,200)
        archive=zipfile.ZipFile(io.BytesIO(base64.b64decode(response.json['fileData'])))
        self.assertEqual(archive.namelist(),['Entry_1.docx','Entry_0.docx'])
        for sections in [[],['bogus']]:
            self.assertEqual(app.test_client().post('/export',json=dict(summaryIds=['0'],sections=sections)).status_code,400)

if __name__ == '__main__': unittest.main()
