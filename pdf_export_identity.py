"""Conservative local comparison of metadata-only re-exports of one document.

Callers must establish the same provider document ID first. This is deliberately
bounded and fails closed without Poppler; text equality alone is insufficient.
"""
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile


def same_rendered_original(left, right):
    try:
        from PyPDF2 import PdfReader
        readers = [PdfReader(io.BytesIO(data), strict=True) for data in (left, right)]
        count = len(readers[0].pages)
        if not 1 <= count <= 60 or len(readers[1].pages) != count:
            return False
        texts = [[page.extract_text() or '' for page in reader.pages] for reader in readers]
        if texts[0] != texts[1] or sum(len(t.strip()) for t in texts[0]) < 100:
            return False
        renderer = shutil.which('pdftoppm')
        if not renderer:
            return False
        with tempfile.TemporaryDirectory(prefix='charlie-export-compare-') as directory:
            hashes = []
            for index, data in enumerate((left, right)):
                root = Path(directory) / str(index)
                root.mkdir()
                source = root / 'original.pdf'
                source.write_bytes(data)
                subprocess.run([renderer, '-r', '96', '-png', str(source), str(root / 'page')],
                               check=True, timeout=30, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                pages = sorted(root.glob('page-*.png'))
                if len(pages) != count:
                    return False
                hashes.append([hashlib.sha256(p.read_bytes()).digest() for p in pages])
            return hashes[0] == hashes[1]
    except Exception:
        return False
