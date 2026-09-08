import io
import shutil
import unittest
from unittest.mock import patch
from pdf_export_identity import same_rendered_original


def original(title, number='10', shade=0):
    from reportlab.pdfgen.canvas import Canvas
    stream = io.BytesIO()
    canvas = Canvas(stream)
    canvas.setTitle(title)
    canvas.drawString(30, 750, 'Revenue grew ' + number + ' percent. ' + 'Original research evidence. ' * 5)
    canvas.setFillGray(shade)
    canvas.rect(30, 600, 40, 50, fill=1)
    canvas.save()
    return stream.getvalue()


class ExportIdentityTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('pdftoppm'), 'Local Poppler required')
    def test_metadata_changes_can_reuse_but_numbers_and_graphics_cannot(self):
        saved = original('old export')
        self.assertTrue(same_rendered_original(saved, original('new export')))
        self.assertFalse(same_rendered_original(saved, original('new export', number='20')))
        self.assertFalse(same_rendered_original(saved, original('new export', shade=.6)))

    def test_missing_renderer_and_invalid_pdf_fail_closed(self):
        with patch('pdf_export_identity.shutil.which', return_value=None):
            self.assertFalse(same_rendered_original(original('old'), original('new')))
        self.assertFalse(same_rendered_original(b'not a pdf', b'not a pdf'))
