"""Original-passage support for new managed meeting questions."""
import base64,json,re
from command_thesis_bridge import file_hash

def excerpts(get_db,docs,budget=120000):
    from recap_validation import catalog,audit_excerpts
    parts=[]
    with get_db() as (_,cur):
        for d in docs:
            cur.execute('SELECT file_data FROM mp_documents WHERE id=%s',(d['id'],));row=cur.fetchone()
            if not row or file_hash(row)!=d['sha256']:raise ValueError('Meeting source changed before passage verification.')
            raw=row['file_data']
            if d['filename'].lower().endswith('.pdf'):parts.append({'name':d['filename'],'type':'pdf','data':raw})
            else:
                text=base64.b64decode(raw).decode('utf-8',errors='strict')
                if d['filename'].lower().endswith(('.htm','.html')):
                    from bs4 import BeautifulSoup
                    text=BeautifulSoup(text,'html.parser').get_text(' ',strip=True)
                parts.append({'name':d['filename'],'type':'text','content':text})
    sources,issues=catalog(parts);selected,limits=audit_excerpts(sources,budget)
    return {'sources':selected,'limitations':issues+limits}

def verify(topics,evidence):
    norm=lambda x:' '.join(x.split())
    sources={s['filename']:s for s in evidence['sources']}
    for ti,topic in enumerate(topics):
        for qi,q in enumerate(topic['questions']):
            quotes=q.get('supporting_quotes')
            if not isinstance(quotes,list) or not quotes:raise ValueError('Question lacks an original supporting passage.')
            for item in quotes:
                if not isinstance(item,dict):raise ValueError('Invalid supporting passage.')
                source=sources.get(item.get('filename'));quote=item.get('quote','')
                if not source or item['filename'] not in q['source_filenames'] or not isinstance(quote,str) or len(norm(quote))<30 or not any(norm(quote) in norm(p['text']) for p in source['pages']):
                    raise ValueError(f'Topic {ti+1}, question {qi+1}: supporting passage was not found in its cited original. Use an exact contiguous passage from that filename; do not paraphrase or join passages. Question stage retained for retry.')
            q['source_support']='Original quote matched; factual interpretation still requires review.'
    return topics
