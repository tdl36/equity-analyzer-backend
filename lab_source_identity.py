"""Optional exact-storage identity contract for Lab handoff to shared research."""
import re


def validate(value, filenames):
    if value is None:
        return None
    if not isinstance(value,list) or not 1<=len(value)<=8:
        raise ValueError('Invalid selected-original identity pack.')
    for item in value:
        if (not isinstance(item,dict) or set(item)!={'id','filename','sha256'}
                or not isinstance(item['id'],str) or not 1<=len(item['id'])<=100
                or not isinstance(item['filename'],str) or item['filename'] not in filenames
                or not isinstance(item['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',item['sha256'])):
            raise ValueError('Invalid selected-original identity.')
    if len({r['id'] for r in value})!=len(value) or sorted(r['filename'] for r in value)!=sorted(filenames):
        raise ValueError('Selected-original identities do not match the source list.')
    return sorted(value,key=lambda r:r['filename'])


def verify(expected, documents):
    if expected is None:return
    from command_thesis_bridge import file_hash
    actual=sorted([{'id':str(d.get('id')),'filename':d['filename'],'sha256':file_hash(d)} for d in documents],key=lambda r:r['filename'])
    if expected!=actual:
        raise ValueError('A selected original changed identity or bytes. Check the Lab selection and start a new revision.')
