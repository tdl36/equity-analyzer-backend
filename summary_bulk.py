"""Ordered summary sections shared by pooled email and PDF export."""
from html import escape
SECTIONS=(('brief','brief','Brief'),('takeaways','summary','Key Takeaways'),('meeting','meeting_summary','Meeting Summary'),('questions','questions','Follow-up Questions'),('assessment','assessment','Assessment'),('korean','korean_takeaways','Korean Key Takeaways'),('transcript','raw_notes','Full Transcript'))

def validate_ids(ids):
    if not isinstance(ids,list) or not ids or any(not isinstance(i,str) or not i or len(i)>100 for i in ids) or len(set(ids))!=len(ids):raise ValueError('Choose distinct saved summaries.')
    return ids

def ordered_rows(cur,ids):
    validate_ids(ids)
    cur.execute('SELECT * FROM meeting_summaries WHERE id=ANY(%s)',(ids,))
    rows={r['id']:dict(r) for r in cur.fetchall()}
    if any(i not in rows for i in ids):raise ValueError('A selected summary is missing. Refresh the list before retrying.')
    return [rows[i] for i in ids]

def validate_sections(selected):
    if selected is not None and (not isinstance(selected,list) or not selected or any(k not in {s[0] for s in SECTIONS} for k in selected)):raise ValueError('Choose valid summary sections.')

def sections_html(row,selected=None, *, pdf=False):
    validate_sections(selected)
    out=[]
    for key,field,label in SECTIONS:
        value=row.get(field) or ''
        if not value or (selected is not None and key not in selected) or (key=='transcript' and row.get('source_type')!='audio'):continue
        if key=='transcript':value='<p>'+escape(value).replace('\n','<br/>')+'</p>'
        elif key=='korean':
            import markdown
            value=markdown.markdown(value,extensions=['extra','sane_lists','nl2br'])
            if pdf:
                value='<div style="font-family: HYSMyeongJo-Medium">'+value+'</div>'
        out.append('<h2>'+label+'</h2>'+value)
    return ''.join(out)

def pooled_html(rows):
    parts=[]
    for row in rows:
        content=sections_html(row)
        title=escape(row.get('title') or 'Summary')
        date=escape(str(row.get('created_at') or '')[:10])
        parts.append(f'<article><h1>{title}</h1><p>{date}</p>{content or "<p>No saved sections available.</p>"}</article>')
    return '<hr style="margin:32px 0;">'.join(parts)
