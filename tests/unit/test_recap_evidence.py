import unittest
from recap_evidence import snapshot, text_prompt

class RecapEvidenceTests(unittest.TestCase):
    def test_identity_changes_with_source_content(self):
        a={'type':'text','name':'note.txt','content':'Original source'}
        b=dict(a,content='Revised source')
        self.assertNotEqual(snapshot([a],'anthropic')['sources'][0]['sha256'],snapshot([b],'anthropic')['sources'][0]['sha256'])
    def test_long_text_is_preserved_not_clipped_at_8000(self):
        text='A'*9000+'END OF SOURCE'
        self.assertIn('END OF SOURCE',text_prompt([{'type':'text','name':'long.txt','content':text}],''))
    def test_oversize_source_is_actionable_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,'no sources were silently truncated'):
            text_prompt([{'type':'text','name':'long.txt','content':'x'*100}], '',char_cap=20)
    def test_binary_placeholder_or_empty_selection_cannot_claim_readable_inputs(self):
        with self.assertRaises(ValueError):snapshot([],'backend_text')
        with self.assertRaises(ValueError):snapshot([{'type':'text','name':'file.pdf','content':''}],'backend_text')
    def test_blank_pdf_page_requires_ocr_for_text_models(self):
        import io,base64
        from PyPDF2 import PdfWriter
        writer=PdfWriter();writer.add_blank_page(width=100,height=100);buf=io.BytesIO();writer.write(buf)
        part={'type':'pdf','name':'scan.pdf','data':base64.b64encode(buf.getvalue()).decode(),'pageCount':1}
        self.assertEqual(snapshot([part],'anthropic')['sources'][0]['inputMode'],'native_pdf')
        with self.assertRaisesRegex(ValueError,'OCR or use a native PDF model'):text_prompt([part],'')

    def test_actual_native_pdf_builder_keeps_full_text_sources(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path('charlie_local_agent.py').read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_content_blocks')
        env={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'charlie_local_agent.py','exec'),env)
        blocks=env['_build_content_blocks']([{'type':'text','name':'long.txt','content':'a'*9000+'FINAL SOURCE PASSAGE'}],'Task')
        self.assertIn('FINAL SOURCE PASSAGE',blocks[0]['text'])

    def test_text_model_partition_preserves_every_source_character(self):
        from recap_evidence import text_batches
        source='First source. '*20000+'FINAL SENTENCE'
        parts=[{'type':'text','name':'large.txt','content':source}]
        batches=text_batches(parts)
        recovered=''.join(p['content'] for batch in batches for p in batch)
        self.assertIn(source,recovered)
        self.assertGreater(len(batches),1)
        for batch in batches:self.assertLess(len(text_prompt(batch,'')),120000)
