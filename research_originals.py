"""Read-only access to the exact stored bytes behind a frozen research source."""
import base64
import binascii
import hashlib
import re


def verified_original(source, document, frozen_hash):
    expected=source.get('originalHash')
    if not isinstance(expected,str) or not re.fullmatch(r'[a-f0-9]{64}',expected) or expected!=frozen_hash:
        raise ValueError('The frozen original identity is unavailable. Inspect the saved source record.')
    if not document or document.get('filename')!=source.get('filename'):
        raise ValueError('This original is no longer stored for the company. Restore the exact original before downloading.')
    encoded=document.get('file_data')
    if not isinstance(encoded,(str,bytes)) or not encoded or len(encoded)>80_000_000:
        raise ValueError('Stored original bytes are unavailable or exceed the verification bound.')
    try:raw=base64.b64decode(encoded,validate=True)
    except (ValueError,binascii.Error):raise ValueError('Stored original bytes could not be decoded.') from None
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('The stored file changed after this research. It is not the frozen original; no replacement was downloaded.')
    return raw
