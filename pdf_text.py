"""Page-preserving text extraction with bounded local OCR and private caches.

Original PDFs are never rewritten. OCR targets text-poor pages, not every
image region in a text-rich page; confidence is an engine score, not accuracy.
"""
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from functools import lru_cache

ROOT=Path.home()/'Library/Application Support/Charlie/OCR'
SLOTS=threading.BoundedSemaphore(1)
MAX_OCR_PAGES=80


def executable(name):
    return shutil.which(name) or next((str(p) for p in (Path('/opt/homebrew/bin')/name,Path('/usr/local/bin')/name) if p.is_file()),None)


@lru_cache(maxsize=1)
def engine():
    tess=executable('tesseract');render=executable('pdftoppm')
    if not tess or not render:raise ValueError('Local OCR requires Tesseract and Poppler; use a native PDF model until available')
    version=subprocess.run([tess,'--version'],capture_output=True,text=True,timeout=10,check=True).stdout.splitlines()[0]
    return tess,render,hashlib.sha256((version+'|eng|2400px|psm3|v1').encode()).hexdigest()[:16]


def recognize(data,page):
    tess,render,_=engine()
    with SLOTS,tempfile.TemporaryDirectory(prefix='charlie-ocr-') as tmp:
        source=Path(tmp)/'source.pdf';source.write_bytes(data);image=Path(tmp)/'page'
        subprocess.run([render,'-f',str(page),'-l',str(page),'-singlefile','-scale-to','2400','-png',str(source),str(image)],capture_output=True,timeout=45,check=True)
        result=subprocess.run([tess,str(image)+'.png','stdout','-l','eng','--psm','3','tsv'],capture_output=True,text=True,timeout=45,check=True)
    import csv
    rows=list(csv.DictReader(io.StringIO(result.stdout),delimiter='\t'));lines={};weighted=chars=0
    for row in rows:
        word=(row.get('text') or '').strip()
        if not word:continue
        confidence=float(row.get('conf',-1))
        if confidence<0:continue
        key=tuple(row.get(k) for k in ('block_num','par_num','line_num'))
        lines.setdefault(key,[]).append(word);weighted+=confidence*len(word);chars+=len(word)
    return {'text':'\n'.join(' '.join(words) for words in lines.values()),'confidence':round(weighted/chars,1) if chars else 0}


def extract(data,strict=True,root=ROOT,ocr=recognize,engine_id=None):
    from PyPDF2 import PdfReader
    if len(data)>60_000_000:raise ValueError('PDF exceeds the 60 MB text-extraction limit')
    reader=PdfReader(io.BytesIO(data))
    if reader.is_encrypted or not len(reader.pages):raise ValueError('PDF is encrypted or has no pages')
    pages=[];issues=[];candidates=[]
    for i,page in enumerate(reader.pages,1):
        try:text=page.extract_text() or ''
        except Exception:text=''
        pages.append({'page':i,'text':text,'mode':'native_text','confidence':None})
        try:
            resources=page.get('/Resources',{}).get_object();objects=resources.get('/XObject',{}).get_object()
            images=any(o.get_object().get('/Subtype')=='/Image' for o in objects.values())
        except Exception:images=False
        if not text.strip() or (images and sum(c.isalnum() for c in text)<80):candidates.append(i)
    if len(candidates)>MAX_OCR_PAGES:
        issues.append(f'{len(candidates)} text-poor pages exceeds the {MAX_OCR_PAGES}-page OCR limit; split the source or use a native PDF model')
        if strict:raise ValueError(issues[-1])
        candidates=[]
    if candidates:
        try:version=engine_id or engine()[2]
        except Exception as exc:
            if strict:raise ValueError(str(exc)) from exc
            issues.append(str(exc));candidates=[]
    key=hashlib.sha256(data).hexdigest();done=[]
    for number in candidates:
        row=pages[number-1]
        try:
            directory=Path(root)/key/version;path=directory/(str(number)+'.json');cached=None
            if path.is_file() and not path.is_symlink() and path.stat().st_size<1_000_000:
                try:
                    saved=json.loads(path.read_text())
                    if saved.get('sourceHash')==key and saved.get('page')==number and isinstance(saved.get('text'),str) and isinstance(saved.get('confidence'),(int,float)):cached=saved
                except (ValueError,OSError):pass
            result=cached or ocr(data,number)
            if not isinstance(result.get('text'),str) or not result['text'].strip() or not isinstance(result.get('confidence'),(int,float)) or not math.isfinite(result['confidence']) or not 70<=result['confidence']<=100:
                raise ValueError('OCR returned empty or low-confidence text; inspect the original page')
            if not cached:
                directory.mkdir(parents=True,exist_ok=True,mode=0o700)
                with tempfile.NamedTemporaryFile('w',dir=directory,delete=False) as stream:
                    json.dump({**result,'sourceHash':key,'page':number},stream);stream.flush();os.fsync(stream.fileno());temporary=stream.name
                os.replace(temporary,path)
            row.update(text=result['text'],mode='ocr',confidence=result['confidence']);done.append(number)
        except Exception as exc:
            issue=f'Page {number}: {type(exc).__name__}: {exc}'
            if strict:raise ValueError(issue) from exc
            issues.append(issue)
    for row in pages:
        if not row['text'].strip():
            issue=f"Page {row['page']}: no readable text"
            if strict:raise ValueError(issue)
            if issue not in issues:issues.append(issue)
    if done:issues.append('OCR transcription may misread numbers, symbols, columns and tables; verify material figures against original pages. OCR targets text-poor pages, not all embedded image regions.')
    return {'pages':pages,'ocrPages':done,'limitations':issues,'sourceHash':key}


def render(result):
    return '\n\n'.join(f"[Page {p['page']}"+(' · OCR transcription' if p['mode']=='ocr' else '')+']\n'+p['text'] for p in result['pages'])
