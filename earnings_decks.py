"""Editable, deterministic earnings decks from a frozen saved analyst recap.

No new research calls, invented citations, or changes to the underlying activity.
"""
import copy
import hashlib
import io
import json
import re
import textwrap
import uuid
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, send_file

THEMES = {
    'paper': ('F6F4EE', '172A29', '17695B', 'FFFFFF'),
    'midnight': ('142329', 'F5F3EB', 'DFB967', '21343B'),
    'sage': ('EDF2EB', '203D32', '457459', 'FFFFFF'),
}

def obj(v):
    if isinstance(v, str):
        try: v = json.loads(v)
        except ValueError: return {}
    return v if isinstance(v, dict) else {}

def plain(s):
    # Preserve links as readable text and URL; do not execute embedded markup.
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', str(s))
    s = re.sub(r'<[^>]*>', '', s)
    return re.sub(r'[*`]', '', s).strip()

def chunks(s, width=190):
    return textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=False) or ['']

def sections(markdown):
    out, title, lines = [], 'Executive overview', []
    for line in markdown.splitlines():
        m = re.match(r'^#{1,6}\s+(.+)', line)
        if m:
            if lines: out.append((title, lines))
            title, lines = plain(m[1]), []
        elif line.strip() and not re.fullmatch(r'[\s|:\-]+', line):
            lines.append(plain(re.sub(r'^\s*(?:[-+•]|\d+[.)])\s+', '', line)))
    if lines: out.append((title, lines))
    return out

def build(row, mode='full', theme='paper'):
    if mode not in ('full', 'brief') or theme not in THEMES:
        raise ValueError('Choose a valid deck format and theme.')
    output, inputs = obj(row.get('output')), obj(row.get('input'))
    markdown = output.get('synthesisMarkdown')
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError('This event has no saved recap yet. Generate the recap first.')
    if row.get('status') in ('running', 'queued'):
        raise ValueError('Wait for the current recap to finish before building a deck.')
    source_rows = obj(output.get('evidenceSnapshot')).get('sources', [])
    source_rows = source_rows if isinstance(source_rows, list) else []
    names = output.get('sourceFiles', [])
    names = names if isinstance(names, list) else []
    sources = []
    for name in dict.fromkeys([n for n in names if isinstance(n, str)] +
                              [s['filename'] for s in source_rows if isinstance(s, dict) and isinstance(s.get('filename'), str)]):
        record = next((s for s in source_rows if isinstance(s, dict) and s.get('filename') == name), {})
        sources.append({'filename': name, 'sha256': record.get('sha256'), 'pages': record.get('pages')})
    title = str(inputs.get('topic') or 'Earnings recap')
    slides = []
    omitted = 0
    def add(heading, items, notes, kind='recap'):
        slides.append({'id': str(uuid.uuid4()), 'title': heading, 'items': items,
                       'notes': notes, 'kind': kind, 'edited': False})
    add(f"{row.get('ticker') or 'Company'} · Earnings review", chunks(title)[:3],
        title + '\nDraft from a saved research recap. Not a fresh source search or an independent verification.', 'cover')
    for heading, lines in sections(markdown):
        selected = lines[:2] if mode == 'brief' else lines
        omitted += len(lines) - len(selected)
        fragments = [part for line in selected for part in chunks(line)]
        # Split content into additional slides rather than shrinking typography.
        for start in range(0, len(fragments), 3):
            label = heading if start == 0 else heading + ' · continued'
            add(label, fragments[start:start+3], '\n'.join(lines) if start == 0 else '\n'.join(fragments[start:start+3]))
    review = obj(output.get('claimReview'))
    comparisons = review.get('numericComparisons', [])
    for n in comparisons if isinstance(comparisons, list) else []:
        if not isinstance(n, dict): continue
        actual, benchmark = obj(n.get('actual')), obj(n.get('benchmark'))
        def metric(value):
            return ' · '.join(str(value.get(k)) for k in ('value','unit','period','basis') if value.get(k) is not None) or 'Not recorded'
        points = ['Actual / updated: ' + metric(actual),
                  str(n.get('benchmarkType') or 'Benchmark').replace('_',' ') + ': ' + metric(benchmark),
                  'Saved comparison status: ' + ('arithmetic checked' if n.get('status') == 'arithmetic_checked' else 'needs review') + '. Review period, units and accounting basis before use.']
        fragments = [part for point in points for part in chunks(point)]
        notes = json.dumps(n, ensure_ascii=False) + '\nSaved recap comparison; not reverified during slide formatting.'
        for start in range(0,len(fragments),3):
            add(str(n.get('metric') or 'Numerical comparison') + (' · continued' if start else ''), fragments[start:start+3], notes, 'comparison')
    claims = review.get('claims', [])
    checked_claims = [c for c in claims if isinstance(c,dict)] if isinstance(claims,list) else []
    if checked_claims:
        passed = sum(c.get('passageMatched') is True and c.get('reviewPassed') is True for c in checked_claims)
        add('Evidence review · checks and open issues',
            [f'{passed} of {len(checked_claims)} selected claims passed saved passage and model checks.',
             f'{len(checked_claims)-passed} selected claims still need evidence review. Checks apply to these selected claims, not every sentence in the recap.',
             'Open the earnings event to inspect the supporting passages and unresolved issues before presenting.'],
            json.dumps(checked_claims,ensure_ascii=False), 'checks')
    if not sources:
        add('Source register unavailable', ['The saved recap has no recorded source filenames. Review the original documents before relying on this deck.'], '', 'sources')
    for start in range(0, len(sources), 3):
        add('Source register' + (' · continued' if start else ''),
            [f"[{i+1}] {s['filename']}" for i, s in enumerate(sources[start:start+3], start)],
            'Recap input register, not claim-level citations.\n' + json.dumps(sources[start:start+3], ensure_ascii=False), 'sources')
    return {'schema': 1, 'ticker': row.get('ticker'), 'title': title, 'mode': mode, 'theme': theme,
            'createdAt': datetime.now(timezone.utc).isoformat(),
            'activityId': str(row['id']), 'activityStatus': row.get('status'),
            'recapHash': hashlib.sha256(markdown.encode()).hexdigest(),
            'sources': sources, 'slides': slides, 'omittedParagraphs': omitted,
            'scope': 'Formatted from a frozen saved recap. Source register is not claim-level verification. No new source search or thesis approval.',
            'warnings': (['Brief includes the first two paragraphs per section; complete section text is retained in speaker notes.'] if omitted else []) +
                        (['The latest recap attempt failed; this deck uses the retained previous draft.'] if row.get('status') == 'failed' else [])}

def revise(current, edits):
    if not isinstance(edits, dict) or not isinstance(edits.get('slides'), list):
        raise ValueError('Slide edits are required.')
    original = {s['id']: s for s in current['slides']}
    if len(edits['slides']) != len(original) or {s.get('id') for s in edits['slides'] if isinstance(s, dict)} != set(original):
        raise ValueError('Reload the deck: slide identities do not match.')
    result = copy.deepcopy(current)
    result['theme'] = edits.get('theme', current['theme'])
    if result['theme'] not in THEMES: raise ValueError('Choose a valid theme.')
    result['slides'] = []
    for s in edits['slides']:
        old = original[s['id']]
        title, items = s.get('title'), s.get('items')
        if not isinstance(title, str) or not title.strip() or (len(title) > 180 and title != old['title']):
            raise ValueError('Use a slide title of 1–180 characters.')
        if not isinstance(items, list) or not 1 <= len(items) <= 3 or any(not isinstance(x, str) or not x.strip() or (len(x) > 240 and items != old['items']) for x in items):
            raise ValueError('Use one to three concise points per slide, up to 240 characters each.')
        if old['kind'] == 'sources' and (title != old['title'] or items != old['items']):
            raise ValueError('Source registers are frozen. Rebuild from an updated recap to change sources.')
        changed = title != old['title'] or items != old['items']
        result['slides'].append({**old, 'title': title, 'items': items, 'edited': old['edited'] or changed})
    return result

def render_pptx(deck):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.xmlchemy import OxmlElement
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    bg, ink, accent, card = THEMES[deck['theme']]
    def rgb(v): return RGBColor.from_string(v)
    def box(slide, text, x, y, w, h, size, color, bold=False, fill=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)) if fill else slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        if fill:
            shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill); shape.line.fill.background()
            shape._element.spPr.append(OxmlElement('a:effectLst'))
            for effect in shape._element.xpath('.//a:effectRef'): effect.set('idx', '0')
        tf = shape.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(.16 if fill else 0)
        tf.margin_top = Inches(.12 if fill else 0); tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.TOP
        p = tf.paragraphs[0]; p.text = text; p.alignment = PP_ALIGN.LEFT
        p.font.name = 'Arial'; p.font.size = Pt(size); p.font.color.rgb = rgb(color); p.font.bold = bold
        p.space_after = Pt(0)
    # Render overflow as explicit continuation slides, including source filenames.
    expanded = []
    for slide in deck['slides']:
        items = [part for item in slide['items'] for part in chunks(item, 190)]
        for start in range(0, len(items), 3):
            expanded.append({**slide, 'title': slide['title'] + (' · continued' if start else ''), 'items': items[start:start+3]})
    for i, data in enumerate(expanded):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid(); slide.background.fill.fore_color.rgb = rgb(bg)
        box(slide, f"CHARLIE  /  {deck['ticker']}  /  EARNINGS REVIEW", .6, .35, 12, .3, 11, accent, True)
        heading = data['title']
        # Explicit wrapped title; preserve full wording in notes for long titles.
        title_lines = textwrap.wrap(heading, 66, break_long_words=True)
        title = '\n'.join(title_lines[:2])
        if len(title_lines)>2: title = '\n'.join(title_lines[:2]) + '…'
        box(slide, title, .6, .95, 12.1, 1.25, 28, ink, True)
        for j, item in enumerate(data['items']):
            box(slide, f'{j+1:02}', .6, 2.55+j*1.15, .5, .4, 14, accent, True)
            box(slide, item, 1.3, 2.35+j*1.15, 11.4, 1.02, 18, ink, fill=card)
        status = 'Analyst edited · review draft' if data.get('edited') else 'Saved recap · review draft'
        box(slide, status+'  |  Source register in appendix', .6, 6.7, 11.1, .3, 12, ink)
        box(slide, f'{i+1} / {len(expanded)}', 11.8, 6.7, .9, .3, 12, ink)
        slide.notes_slide.notes_text_frame.text = (data['title']+'\n\n'+data.get('notes','')+
            '\n\n'+deck['scope']+'\nRecap SHA-256: '+deck['recapHash'])
    output = io.BytesIO(); prs.save(output); return output.getvalue()


def create_blueprint(get_db):
    bp = Blueprint('earnings_decks', __name__)
    def ensure(cur):
        cur.execute('''CREATE TABLE IF NOT EXISTS earnings_decks (
            id UUID NOT NULL, revision INTEGER NOT NULL, activity_id UUID NOT NULL,
            body JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (id, revision))''')
    def ident(value): return str(uuid.UUID(str(value)))
    @bp.route('/api/earnings/decks', methods=['GET', 'POST'])
    def decks():
        try:
            with get_db(commit=True) as (_, cur):
                ensure(cur)
                if request.method == 'GET':
                    activity = ident(request.args.get('activityId'))
                    cur.execute('SELECT DISTINCT ON (id) id,revision,body FROM earnings_decks WHERE activity_id=%s ORDER BY id,revision DESC', (activity,))
                    return jsonify(decks=[{'id':str(r['id']), 'revision':r['revision'], 'body':obj(r['body'])} for r in cur.fetchall()])
                data = request.get_json() or {}; key = ident(data.get('requestId')); activity = ident(data.get('activityId'))
                cur.execute('SELECT id,revision,body,activity_id FROM earnings_decks WHERE id=%s ORDER BY revision DESC LIMIT 1', (key,))
                old = cur.fetchone()
                if old:
                    if str(old['activity_id']) != activity: raise ValueError('Request belongs to another event.')
                    return jsonify(id=str(old['id']),revision=old['revision'],body=obj(old['body']))
                cur.execute('SELECT id,ticker,input,output,status FROM analyst_activities WHERE id=%s', (activity,))
                row = cur.fetchone()
                if not row: return jsonify(error='Saved event not found.'),404
                body = build(dict(row), data.get('mode','full'), data.get('theme','paper'))
                cur.execute('INSERT INTO earnings_decks(id,revision,activity_id,body) VALUES(%s,1,%s,%s) ON CONFLICT DO NOTHING', (key,activity,json.dumps(body)))
                # Return the winning snapshot if a retry raced this request.
                cur.execute('SELECT revision,body,activity_id FROM earnings_decks WHERE id=%s ORDER BY revision DESC LIMIT 1',(key,))
                saved=cur.fetchone()
                if str(saved['activity_id']) != activity: raise ValueError('Request belongs to another event.')
                return jsonify(id=key,revision=saved['revision'],body=obj(saved['body']))
        except (ValueError,TypeError,AttributeError) as e: return jsonify(error=str(e)),400
    @bp.route('/api/earnings/decks/<deck_id>', methods=['GET', 'PUT'])
    def save(deck_id):
        try:
            key=ident(deck_id)
            if request.method == 'GET':
                with get_db(commit=True) as (_,cur):
                    ensure(cur)
                    cur.execute('SELECT revision,created_at FROM earnings_decks WHERE id=%s ORDER BY revision DESC',(key,))
                    history=cur.fetchall()
                    if not history:return jsonify(error='Deck not found.'),404
                    revision=int(request.args.get('revision',history[0]['revision']))
                    cur.execute('SELECT body FROM earnings_decks WHERE id=%s AND revision=%s',(key,revision))
                    saved=cur.fetchone()
                    if not saved:return jsonify(error='Deck revision not found.'),404
                    return jsonify(id=key,revision=revision,body=obj(saved['body']),history=[{'revision':h['revision'],'createdAt':str(h['created_at'])} for h in history])
            data=request.get_json() or {}
            if type(data.get('revision')) is not int:raise ValueError('Reload the deck revision.')
            with get_db(commit=True) as (_,cur):
                ensure(cur)
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(key,))
                cur.execute('SELECT revision,activity_id,body FROM earnings_decks WHERE id=%s ORDER BY revision DESC LIMIT 1',(key,))
                old=cur.fetchone()
                if not old:return jsonify(error='Deck not found.'),404
                if old['revision']!=data['revision']:return jsonify(error='A newer revision exists. Reload before editing.'),409
                body=revise(obj(old['body']), data.get('body'))
                rev=old['revision']+1
                cur.execute('INSERT INTO earnings_decks(id,revision,activity_id,body) VALUES(%s,%s,%s,%s)',(key,rev,old['activity_id'],json.dumps(body)))
                return jsonify(id=key,revision=rev,body=body)
        except (ValueError,TypeError,AttributeError) as e:return jsonify(error=str(e)),400
    @bp.route('/api/earnings/decks/<deck_id>/export')
    def export(deck_id):
        try:
            key=ident(deck_id);revision=int(request.args.get('revision','0'))
            with get_db(commit=True) as (_,cur):
                ensure(cur)
                cur.execute('SELECT body FROM earnings_decks WHERE id=%s AND revision=%s',(key,revision))
                row=cur.fetchone()
            if not row:return jsonify(error='Saved deck revision not found.'),404
            body=obj(row['body'])
            return send_file(io.BytesIO(render_pptx(body)),mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',as_attachment=True,download_name=f"{body['ticker']}_earnings_review_v{revision}.pptx")
        except (ValueError,TypeError) as e:return jsonify(error=str(e)),400
    return bp
