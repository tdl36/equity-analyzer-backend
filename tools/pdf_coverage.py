"""Check that content in the data actually reaches the page.

Overflow that clips is invisible to every check we had: the page count is
right, nothing overlaps, and the preflight passes, while the last bullet of a
list or the tail of a paragraph is simply not printed. Shipped reports lost the
fifth Key Opportunity's body, the last two bull/bear items and the end of the
variant view this way.

For each field the renderers print, assert its opening words appear somewhere
in the extracted PDF text. Comparison is on alphanumerics only, because
extraction drops hyphens at line breaks and re-spaces runs.
"""
import json, re, subprocess, sys


def norm(t):
    return re.sub(r'[^a-z0-9]', '', str(t or '').lower())


def pdf_text(pdf):
    return subprocess.run(['pdftotext', pdf, '-'], capture_output=True, text=True).stdout


def head_words(text, n=7):
    ws = [w for w in re.split(r'\s+', str(text or '').strip()) if w]
    return ' '.join(ws[:n]) if len(ws) >= 3 else None


def expected_fields(fx, view):
    """(label, text) pairs the given view is supposed to render."""
    m, o = fx.get('master', {}), fx.get('onepager', {})
    out = []
    if view == 'memo':
        t = m.get('investment_thesis', {})
        out += [('thesis.summary', t.get('summary')),
                ('thesis.variant_view', t.get('variant_view')),
                ('final_takeaway', m.get('final_takeaway')),
                ('bottom_line', m.get('bottom_line')),
                ('overview.summary', (m.get('company_overview') or {}).get('summary'))]
        for i, x in enumerate(m.get('opportunities') or []):
            out.append((f'opportunity[{i}].detail', x.get('detail')))
        for i, x in enumerate(m.get('signposts') or []):
            out.append((f'signpost[{i}].why', x.get('why_it_matters')))
        for i, x in enumerate(m.get('thesis_threats') or []):
            out.append((f'threat[{i}].watch_for', x.get('watch_for')))
        for i, x in enumerate(m.get('catalysts') or []):
            out.append((f'catalyst[{i}].why_it_matters', x.get('why_it_matters')))
        for i, x in enumerate(m.get('valuation_scenarios') or []):
            out.append((f'scenario[{i}].logic', x.get('logic')))
    else:
        out += [('final_takeaway', o.get('final_takeaway')),
                ('bottom_line', o.get('bottom_line')),
                ('overview_summary', o.get('overview_summary'))]
        for i, x in enumerate(o.get('bull_case') or []):
            out.append((f'bull_case[{i}]', x))
        for i, x in enumerate(o.get('bear_case') or []):
            out.append((f'bear_case[{i}]', x))
        for i, x in enumerate(o.get('opportunities') or []):
            out.append((f'opportunity[{i}].detail', x.get('detail')))
        if view == 'twopager':
            for i, x in enumerate(o.get('signposts') or []):
                out.append((f'signpost[{i}].why', x.get('why')))
            for i, x in enumerate(o.get('threats') or []):
                out.append((f'threat[{i}].watch_for', x.get('watch_for')))
    return [(k, v) for k, v in out if v]


if __name__ == '__main__':
    pdf, fixture_path, view = sys.argv[1], sys.argv[2], sys.argv[3]
    fx = json.load(open(fixture_path))
    body = norm(pdf_text(pdf))
    missing = []
    for label, text in expected_fields(fx, view):
        h = head_words(text)
        if h and norm(h) not in body:
            missing.append((label, str(text)[:64]))
    for label, snippet in missing:
        print(f'  MISSING {label}: "{snippet}..."')
    print(f'{pdf}: {len(missing)} field(s) not printed')
    sys.exit(1 if missing else 0)
