"""Original-passage support for new managed meeting questions."""
import base64,json,re,hashlib
from command_thesis_bridge import file_hash

def excerpts(get_db,docs,budget=120000):
    from recap_validation import catalog,audit_excerpts
    sources=[];issues=[];text_limits=[]
    def add(part):
        selected,limitations=catalog([part])
        for source in selected:source['id']='s'+str(len(sources)+1);sources.append(source)
        issues.extend(limitations)
    with get_db() as (_,cur):
        for d in docs:
            cur.execute('SELECT file_data,extracted_text FROM mp_documents WHERE id=%s',(d['id'],));row=cur.fetchone()
            if d.get('textSha256'):
                text=(row or {}).get('extracted_text') or ''
                if hashlib.sha256(text.encode()).hexdigest()!=d['textSha256']:raise ValueError('Saved meeting text changed before passage verification.')
                add({'name':d['filename'],'type':'text','content':text})
                text_limits.append(d['filename']+': verified against saved extracted text; original-file passage verification unavailable.')
                continue
            if not row or file_hash(row)!=d['sha256']:raise ValueError('Meeting source changed before passage verification.')
            raw=row['file_data']
            if d['filename'].lower().endswith('.pdf'):add({'name':d['filename'],'type':'pdf','data':raw})
            elif d['filename'].lower().endswith(('.png','.jpg','.jpeg','.gif','.webp')):
                text=row['extracted_text'] or ''
                add({'name':d['filename'],'type':'text','content':text})
                text_limits.append(d['filename']+': image passage support is limited to saved extracted text.')
            else:
                text=base64.b64decode(raw).decode('utf-8',errors='strict')
                if d['filename'].lower().endswith(('.htm','.html')):
                    from bs4 import BeautifulSoup
                    text=BeautifulSoup(text,'html.parser').get_text(' ',strip=True)
                add({'name':d['filename'],'type':'text','content':text})
    selected,limits=audit_excerpts(sources,budget)
    text_names={d['filename'] for d in docs if d.get('textSha256')}
    for source in selected:source['supportKind']='saved_text' if source['filename'] in text_names else 'original'
    return {'sources':selected,'limitations':issues+limits+text_limits}

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
            text_only=any(sources.get(item['filename'],{}).get('supportKind')=='saved_text' for item in quotes)
            q['source_support']=('Saved extracted text matched; original-file passage verification unavailable for one or more citations.' if text_only else 'Original quote matched; factual interpretation still requires review.')
    return topics


def passage_register(evidence):
    """Stable, verbatim bounded passages; models select IDs, never transcribe quotes."""
    register={}
    for source in evidence['sources']:
        for page in source['pages']:
            text=page['text'];offset=0
            while offset<len(text):
                end=min(offset+1000,len(text))
                if end<len(text):
                    boundary=text.rfind(' ',offset+500,end)
                    if boundary>offset:end=boundary
                quote=text[offset:end];offset=end
                if len(' '.join(quote.split()))<30:continue
                ident='p'+str(len(register)+1)
                register[ident]={'filename':source['filename'],'page':page.get('page'),'quote':quote}
    if not register:raise ValueError('No readable original passages for meeting questions.')
    return register


def attach_passages(topics,register):
    for ti,topic in enumerate(topics):
        for qi,q in enumerate(topic['questions']):
            refs=q.pop('supporting_passage_ids',None)
            if not isinstance(refs,list) or not refs or len(refs)>6:
                raise ValueError(f'Topic {ti+1}, question {qi+1}: choose 1–6 original passage IDs.')
            if any(not isinstance(ref,str) or ref not in register or register[ref]['filename'] not in q['source_filenames'] for ref in refs):
                raise ValueError(f'Topic {ti+1}, question {qi+1}: passage ID must belong to its cited filename.')
            q['supporting_quotes']=[dict(register[ref]) for ref in dict.fromkeys(refs)]
    return topics
