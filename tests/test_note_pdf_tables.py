"""Table rendering in the note PDF.

Column widths used to be positional -- first column 14%, last 26%, the rest
split evenly -- which assumed the first column held a short label. A real CVS
note put "Scenario-weighted (bull/base/bear)" there, and because xhtml2pdf
honours a fixed width rather than expanding it, the text printed straight over
the next column: the PDF read "Consensus-based$93".

These tests render the actual table through the same code the PDF route uses
and read the text back out, because the defect is only visible in the artifact.
"""
import io
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PRICE_TARGET_TABLE = """| Source | Price Target | Methodology | Implied Upside from $94.72 |
|---|---|---|---|
| Scenario-weighted (bull/base/bear) | $123 | 12.5x 2028E EPS ($10.65), discounted 1/2 year | +29.9% |
| Peer-relative | $111 | 13.0x 2027E EPS ($8.51) | +17.2% |
| Consensus-based | $93 | 11.0x 2027E EPS ($8.45) | -1.8% |
"""


def _table_renderer():
    """Lift _md_table_to_html out of the PDF route so the test uses the real one."""
    src = (ROOT / 'app_v3.py').read_text(encoding='utf-8')
    start = src.index('    def _md_table_to_html(match):')
    end = src.index("    html_body = re.sub(r'(\\|.+\\|[\\n\\r]*)+'")
    ns = {}
    exec('import re\n' + '\n'.join(l[4:] for l in src[start:end].splitlines()), ns)
    return ns['_md_table_to_html']


def _render(md):
    html_table = re.sub(r'(\|.+\|[\n\r]*)+', _table_renderer(), md)
    html = ('<html><head><style>@page { size: letter portrait; margin: 0.6in; }'
            'body { font-family: Helvetica; font-size:10pt; }</style></head>'
            f'<body>{html_table}</body></html>')
    pisa = pytest.importorskip('xhtml2pdf.pisa')
    out = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=out)
    return out.getvalue()


def _text(pdf_bytes, tmp_path):
    p = tmp_path / 'table.pdf'
    p.write_bytes(pdf_bytes)
    try:
        proc = subprocess.run(['pdftotext', '-layout', str(p), '-'],
                              capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        pytest.skip('pdftotext not available')
    return proc.stdout


def test_a_long_first_column_does_not_overprint_its_neighbour(tmp_path):
    """The exact failure: a label running into the next column's value."""
    text = _text(_render(PRICE_TARGET_TABLE), tmp_path)
    for label, value in (('Scenario-weighted', '$123'),
                         ('Peer-relative', '$111'),
                         ('Consensus-based', '$93')):
        assert f'{label}{value}' not in text, (
            f'"{label}" printed on top of "{value}" -- the column is too narrow')
        assert label in text, f'{label} missing from the rendered table'


def test_column_widths_follow_content_and_total_one_hundred(tmp_path):
    html = re.sub(r'(\|.+\|[\n\r]*)+', _table_renderer(), PRICE_TARGET_TABLE)
    widths = [float(w) for w in re.findall(r'width="([\d.]+)%"', html)[:4]]
    assert abs(sum(widths) - 100.0) < 0.5, f'widths sum to {sum(widths)}'
    # "Source" holds the longest labels; "Price Target" holds "$123".
    assert widths[0] > widths[1], (
        f'the long label column ({widths[0]:.1f}%) is narrower than the short '
        f'value column ({widths[1]:.1f}%)')
    assert widths[0] >= 20, f'first column only {widths[0]:.1f}% for a 34-char label'


def test_every_cell_survives_rendering(tmp_path):
    """No column may be squeezed until its content vanishes."""
    text = _text(_render(PRICE_TARGET_TABLE), tmp_path)
    for fragment in ('12.5x 2028E EPS', '+29.9%', '-1.8%', '13.0x 2027E EPS'):
        assert fragment.split()[0] in text, f'{fragment!r} did not render'


def test_a_stored_note_renders_its_charts_into_the_pdf(tmp_path):
    """The join between storage and rendering, which is where charts were lost.

    Both halves passed their own tests while notes shipped chartless: the insert
    dropped the base64 'data' key, and the renderer skips any chart without it.
    Neither test looked across the seam. This walks the real route -- seed a note
    with real PNG bytes, request the PDF, and require the images to be in it.
    """
    import base64
    import json as _json
    import os
    import uuid
    os.environ.setdefault('DATABASE_URL', 'postgresql://localhost/charlie_test')
    import app_v3
    import segment_charts

    pytest.importorskip('xhtml2pdf.pisa')

    # Shares must differ between the series, or is_duplicate_series correctly
    # refuses to draw a profit chart that merely repeats revenue.
    charts = segment_charts.render_pair(
        'ZZZ',
        [{'segment': 'Alpha', 'revenue': 190000}, {'segment': 'Beta', 'revenue': 143000}],
        [{'segment': 'Alpha', 'profit': 2000}, {'segment': 'Beta', 'profit': 8000}])
    assert len(charts) == 2, f'fixture produced {len(charts)} chart(s), need 2'

    stored = [{'type': c['type'], 'filename': c['filename'],
               'data': base64.b64encode(c['png']).decode('ascii')} for c in charts]

    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM research_notes WHERE ticker = 'ZZZ'")
        cur.execute(
            """INSERT INTO research_notes
               (id, ticker, version, note_markdown, sources_markdown,
                changelog_markdown, note_docx, charts, metadata, status)
               VALUES (%s,'ZZZ',1,'# Zzz Inc (ZZZ)',' ',' ',' ',%s,%s,'published')""",
            (str(uuid.uuid4()), _json.dumps(stored), _json.dumps({})))
    try:
        resp = app_v3.app.test_client().get('/api/notes/ZZZ/pdf')
        assert resp.status_code == 200, resp.status_code
        pdf = base64.b64decode(resp.get_json()['fileData'])
        assert pdf[:5] == b'%PDF-', 'not a PDF'
        # Two embedded donut PNGs make this far larger than a text-only render,
        # which comes out under 2KB.
        assert len(pdf) > 60_000, (
            f'PDF is only {len(pdf)}b -- the charts did not make it onto the page')
    finally:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute("DELETE FROM research_notes WHERE ticker = 'ZZZ'")
