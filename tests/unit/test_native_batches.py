import base64
import io
import unittest
from PyPDF2 import PdfReader, PdfWriter
from recap_evidence import native_batches, snapshot


def pdf(pages):
    writer=PdfWriter()
    for i in range(pages):writer.add_blank_page(width=100+i,height=100)
    out=io.BytesIO();writer.write(out)
    return {'type':'pdf','name':'Original.pdf','data':base64.b64encode(out.getvalue()).decode(),'pageCount':pages}


class NativeBatchesTests(unittest.TestCase):
    def test_split_preserves_every_original_page_and_identity(self):
        part=pdf(71);before=snapshot([part],'anthropic')
        batches=native_batches([part])
        self.assertEqual([b[0]['pageCount'] for b in batches],[30,30,11])
        self.assertEqual([b[0]['pageStart'] for b in batches],[1,31,61])
        widths=[]
        for batch in batches:
            for p in batch:
                widths.extend(float(page.mediabox.width) for page in PdfReader(io.BytesIO(base64.b64decode(p['data']))).pages)
        self.assertEqual(widths,list(range(100,171)))
        self.assertEqual(before['sources'],snapshot([part],'anthropic')['sources'])

    def test_multiple_files_share_page_budget_and_small_original_bytes_unchanged(self):
        a=pdf(20);b=pdf(20)
        batches=native_batches([a,b]);self.assertEqual(len(batches),2)
        self.assertEqual(batches[0][0]['data'],a['data'])

    def test_text_tail_is_retained(self):
        text='x'*230001+'END';batches=native_batches([{'type':'text','name':'Large.txt','content':text}])
        self.assertEqual(''.join(p['content'] for b in batches for p in b),text)
        self.assertEqual(len(batches),3)

    def test_oversized_single_page_fails_before_any_batch_is_returned(self):
        with self.assertRaisesRegex(ValueError,'No source pages were dropped'):
            native_batches([pdf(2)],byte_budget=100)

    def test_prompt_maps_split_page_to_original_page(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path('charlie_local_agent.py').read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_content_blocks')
        env={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'agent','exec'),env)
        blocks=env['_build_content_blocks'](native_batches([pdf(40)])[1],'Task')
        self.assertIn('PDF page 1 corresponds to original page 31',blocks[1]['text'])
