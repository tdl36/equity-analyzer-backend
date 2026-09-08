import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from PyPDF2 import PdfWriter
from pdf_text import extract,render


def blank(count=1):
    out=io.BytesIO();writer=PdfWriter()
    for _ in range(count):writer.add_blank_page(width=612,height=792)
    writer.write(out);return out.getvalue()


class OCRTests(unittest.TestCase):
    def test_page_numbers_source_hash_and_cached_retry(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            def ocr(data,page):calls.append(page);return {'text':f'Page {page} revenue was $120 million. Guidance increased to $130 million.','confidence':95}
            result=extract(blank(2),root=root,ocr=ocr,engine_id='test')
            self.assertEqual(result['ocrPages'],[1,2]);self.assertIn('[Page 2 · OCR transcription]',render(result));self.assertTrue(result['limitations'])
            self.assertEqual(extract(blank(2),root=root,ocr=ocr,engine_id='test'),result);self.assertEqual(calls,[1,2])
    def test_partial_checkpoint_survives_later_failure(self):
        with tempfile.TemporaryDirectory() as root:
            def fail(data,page):
                if page==2:raise ValueError('interruption')
                return {'text':'Completed page one evidence','confidence':95}
            with self.assertRaises(ValueError):extract(blank(2),root=root,ocr=fail,engine_id='test')
            calls=[]
            def resume(data,page):calls.append(page);return {'text':'Completed page two evidence','confidence':96}
            result=extract(blank(2),root=root,ocr=resume,engine_id='test')
            self.assertEqual(calls,[2]);self.assertEqual(len(result['pages']),2)
    def test_low_confidence_stops_text_only_input(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'low-confidence'):
                extract(blank(),root=root,ocr=lambda *args:{'text':'123 uncertain','confidence':45},engine_id='test')
            self.assertEqual(list(Path(root).rglob('*.json')),[])
    def test_native_review_retains_explicit_unreadable_page(self):
        with tempfile.TemporaryDirectory() as root:
            result=extract(blank(),strict=False,root=root,ocr=lambda *args:{'text':'','confidence':0},engine_id='test')
            self.assertEqual(result['pages'][0]['page'],1);self.assertTrue(result['limitations']);self.assertEqual(result['ocrPages'],[])
    def test_source_and_engine_changes_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            def ocr(data,page):calls.append(page);return {'text':'Evidence','confidence':99}
            extract(blank(),root=root,ocr=ocr,engine_id='v1');extract(blank(),root=root,ocr=ocr,engine_id='v2')
            self.assertEqual(len(calls),2)
    def test_excessive_scanned_pages_fail_before_ocr(self):
        with patch('pdf_text.MAX_OCR_PAGES',1):
            with self.assertRaisesRegex(ValueError,'OCR limit'):extract(blank(2),ocr=lambda *a:self.fail('must not OCR'))
    def test_short_native_text_does_not_require_ocr(self):
        from reportlab.pdfgen import canvas
        out=io.BytesIO();c=canvas.Canvas(out);c.drawString(30,700,'Short native text.');c.save()
        result=extract(out.getvalue(),ocr=lambda *args:self.fail('native text should not be OCRed'))
        self.assertEqual(result['ocrPages'],[]);self.assertIn('Short native text',render(result))
