"""Freeze server-owned meeting sources; never trust client-provided source text."""
import hashlib
from command_thesis_bridge import file_hash


def freeze(get_db,meeting_id,selected):
    if not isinstance(selected,list) or not 1<=len(selected)<=50:
        raise ValueError('Choose 1–50 meeting documents.')
    ids=[d.get('id') for d in selected if isinstance(d,dict)]
    if len(ids)!=len(selected) or any(type(i) is not int for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Choose distinct saved meeting documents.')
    with get_db() as (_,cur):
        cur.execute("SELECT id,filename,(file_data IS NOT NULL AND file_data<>'') AS has_file_data,extracted_text,doc_type FROM mp_documents WHERE meeting_id=%s AND id=ANY(%s)",(meeting_id,ids))
        rows={r['id']:r for r in cur.fetchall()}
    docs=[]
    for ident in ids:
        r=rows.get(ident)
        if not r:raise ValueError('A selected document no longer belongs to this meeting. Reload the source list.')
        d={'id':ident,'filename':r['filename'],'docType':r['doc_type'],'extractedText':r['extracted_text'] or ''}
        if r.get('has_file_data'):
            with get_db() as (_,cur):
                cur.execute('SELECT file_data FROM mp_documents WHERE meeting_id=%s AND id=%s',(meeting_id,ident))
                original=cur.fetchone()
                if not original:raise ValueError('Selected original was removed during source verification.')
                d['sha256']=file_hash(original)
                del original
            if d['filename'].lower().endswith(('.png','.jpg','.jpeg','.gif','.webp')):
                d['textSha256']=hashlib.sha256(d['extractedText'].encode()).hexdigest()
        elif d['extractedText']:
            d['textSha256']=hashlib.sha256(d['extractedText'].encode()).hexdigest()
        else:raise ValueError('No readable source retained for '+r['filename'])
        docs.append(d)
    if len({d['filename'] for d in docs})!=len(docs):
        raise ValueError('Selected documents have duplicate filenames. Rename or remove duplicates before generating.')
    return docs
