"""No provider calls or OCR; synthetic PDF containers only."""
import base64
import io
import unittest
from unittest.mock import patch
from PyPDF2 import PdfWriter
from processing_proposal_note import check_pdf


class PdfReadiness(unittest.TestCase):
    def document(self, pages=1, encrypted=False):
        writer = PdfWriter()
        for _ in range(pages): writer.add_blank_page(width=100, height=100)
        if encrypted: writer.encrypt('synthetic')
        stream = io.BytesIO();writer.write(stream)
        return {'filename': 'fixture.pdf', 'file_data': base64.b64encode(stream.getvalue()).decode()}

    def test_native_pdf_pages_are_allowed_without_text_extraction(self):
        with patch('pdf_text.extract', side_effect=AssertionError('Unneeded extraction')):
            check_pdf(self.document(), False)

    def test_encrypted_pdf_is_blocked(self):
        with self.assertRaisesRegex(ValueError, 'encrypted'):
            check_pdf(self.document(encrypted=True), True)

    def test_oversized_pdf_requires_policy_and_never_runs_ocr(self):
        document = self.document(86)
        with self.assertRaisesRegex(ValueError, 'text-only conversion'):
            check_pdf(document, False)
        with patch('pdf_text.engine', side_effect=AssertionError('OCR engine called')):
            with self.assertRaisesRegex(ValueError, 'needs OCR'):
                check_pdf(document, True)
