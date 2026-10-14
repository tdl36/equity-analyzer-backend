"""Bounded exact-path reader for future Lab imports; no scanning, uploads or dispatch.

Not connected to the live agent yet. Caller must persist the returned identity before
upload and use the same expected hash on retries. Folder names match the agent catalog.
"""
import base64
import hashlib
import json
import os
import re
import stat
from contextlib import ExitStack
from pathlib import Path

MAX_FILE = 15_000_000
EXTENSIONS = {'.pdf','.txt','.md','.html','.htm','.csv','.tsv','.docx','.pptx','.xlsx','.xls'}


def identity(reference):
    if not isinstance(reference,str) or not reference.startswith('icloud:') or len(reference)>1200:
        raise ValueError('An exact iCloud folder and filename are required.')
    try:parts=json.loads(reference[7:])
    except ValueError:raise ValueError('Invalid iCloud source identity.')
    if not isinstance(parts,list) or len(parts)!=2 or not all(isinstance(s,str) and s for s in parts):
        raise ValueError('Invalid iCloud source identity.')
    folder,name=parts
    chunks=folder.split('/')
    if any(s in ('','.','..') or s.startswith('.') or '\\' in s or any(ord(c)<32 for c in s) for s in chunks+[name]):
        raise ValueError('Invalid iCloud source path.')
    if '/' in name or name.startswith('~$') or Path(name).suffix.lower() not in EXTENSIONS:
        raise ValueError('Unsupported iCloud original.')
    return folder,name


def read_original(stocks,catalysts,ticker,reference,expected_sha256=None):
    """Read only the selected original, using no-follow directory descriptors.

    Never hydrate placeholders, traverse links, search another folder or substitute
    a same-named file. Missing/unavailable sources remain actionable failures.
    """
    if not isinstance(ticker,str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,19}',ticker):
        raise ValueError('Confirmed ticker required.')
    if expected_sha256 is not None and (not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256)):
        raise ValueError('Invalid expected original hash.')
    folder,name=identity(reference)
    if folder.startswith('Catalysts/'):
        root=Path(catalysts);relative=[ticker]+folder.split('/')[1:]
    elif folder=='Catalysts':
        raise ValueError('Select an exact catalyst event folder.')
    else:
        root=Path(stocks);relative=[ticker]+([] if folder=='main' else folder.split('/'))
    if not root.is_absolute():raise ValueError('Configured source roots must be absolute.')
    flags=os.O_RDONLY|os.O_NOFOLLOW
    try:
        with ExitStack() as stack:
            def directory(part,parent=None):
                fd=os.open(part,flags|os.O_DIRECTORY,dir_fd=parent)
                stack.callback(os.close,fd);return fd
            # Walk from / so symlinked ancestors cannot redirect the configured root.
            parent=directory('/')
            for part in list(root.parts[1:])+relative:parent=directory(part,parent)
            before=os.stat(name,dir_fd=parent,follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):raise ValueError('Select a regular original, not a link or directory.')
            if getattr(before,'st_flags',0)&getattr(stat,'SF_DATALESS',0x40000000):
                raise ValueError('Original is not downloaded on this Mac. Download it in iCloud first.')
            if not 0<before.st_size<=MAX_FILE:raise ValueError('Original is empty or exceeds the 15 MB Lab limit.')
            fd=os.open(name,flags|os.O_NONBLOCK,dir_fd=parent);stack.callback(os.close,fd)
            opened=os.fstat(fd)
            fingerprint=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            if not stat.S_ISREG(opened.st_mode) or fingerprint(before)!=fingerprint(opened):
                raise ValueError('Original changed before reading. Reload the source inventory.')
            chunks=[];size=0
            while True:
                block=os.read(fd,min(1024*1024,MAX_FILE+1-size))
                if not block:break
                chunks.append(block);size+=len(block)
                if size>MAX_FILE:raise ValueError('Original grew beyond the Lab limit.')
            after=os.fstat(fd);path_after=os.stat(name,dir_fd=parent,follow_symlinks=False)
            if fingerprint(opened)!=fingerprint(after) or fingerprint(after)!=fingerprint(path_after) or size!=opened.st_size:
                raise ValueError('Original changed during reading. Reload before importing.')
            raw=b''.join(chunks);sha=hashlib.sha256(raw).hexdigest()
            if expected_sha256 is not None and sha!=expected_sha256:
                raise ValueError('Original differs from the frozen import identity; no substitution is allowed.')
            return {'reference':reference,'ticker':ticker,'folder':folder,'filename':name,
                    'sha256':sha,'bytes':size,'fileData':base64.b64encode(raw).decode('ascii')}
    except OSError:
        raise ValueError('Original unavailable at its exact iCloud path. Check Mac access, download status and links.') from None
