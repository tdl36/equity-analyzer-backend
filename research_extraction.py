"""Record extraction coverage before a source can reach paid research."""
import copy
import hashlib
from pathlib import Path

PER_SOURCE_LIMIT=160000


def frozen_source(document, original_hash, text, ident):
    if not isinstance(text,str) or not text.strip():
        raise ValueError('A selected original has no readable text. Inspect its extraction before starting research.')
    if len(text)>PER_SOURCE_LIMIT or 'middle of document omitted to fit context' in text:
        raise ValueError('A selected original exceeds the 160,000-character extraction bound. Supply a scoped extract as a separate source; the original is retained. Research has not started.')
    extraction_hash=hashlib.sha256(text.encode()).hexdigest()
    metadata=document.get('metadata') or {}
    result={'id':ident,'filename':document['filename'],'originalHash':original_hash,'extractionHash':extraction_hash,
            'sourceUrl':metadata.get('sourceUrl','') if isinstance(metadata,dict) else '', 'text':text,
            'coverage':{'mode':'full_returned_extraction','extractedCharacters':len(text),'includedRanges':[[0,len(text)]],
                        'omittedCharacters':0,'fullExtractionHash':extraction_hash,
                        'limitation':''}}
    details=copy.deepcopy(document.get('textExtraction') or {})
    limitations=list(details.get('limitations') or [])
    suffix=Path(document['filename']).suffix.lower()
    if suffix in ('.docx','.doc'):
        limitations.append('Paragraph and table text only; headers, footers, images, tracked changes and layout are not analyzed. Tables may appear after body paragraphs.')
    if suffix in ('.xlsx','.xlsm','.xls'):
        limitations.append('Cached worksheet values only; formulas are not recalculated. Charts, formatting and uncached formula results are not analyzed.')
    if '\ufffd' in text:
        limitations.append('Replacement characters occur in the extracted text; inspect the original for decoding loss.')
    details['limitations']=list(dict.fromkeys(limitations))
    if details:result['textExtraction']=details
    return result


def has_omissions(sources):
    """Metadata existence alone does not mean text was omitted."""
    for source in sources:
        coverage=source.get('coverage') or {}
        count=coverage.get('omittedCharacters')
        if coverage.get('mode')=='bounded_excerpts' or (isinstance(count,(int,float)) and not isinstance(count,bool) and count>0):return True
    return False
