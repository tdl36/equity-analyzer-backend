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


TWO_TABLES_BACK_TO_BACK = """| Line | FY2025A |
|---|---|
| Health Care Benefits revenue | ~143,700 (E) |
| Gross segment revenue | ~473,400 |
| Consolidated total revenue (actual) | 402,067 |
| Adjusted operating income | FY2025A ($mn) | FY2026E ($mn) | Basis |
|---|---|---|---|
| Health Care Benefits | ~2,935 (E) | 5,200 | FY26 = midpoint of guide |
| Health Services | ~7,286 (E) | 7,388 (E) | FY26 guide reiterated |
| Enterprise adjusted operating income | 14,443 (A) | 16,750 | FY26 = midpoint |
"""


def test_two_tables_written_back_to_back_stay_separate():
    """A narrow table followed by a wide one must not merge into one grid.

    The match spans consecutive pipe rows and blank lines do not break it, so
    both tables arrived as a single block. Column count came from the widest
    row, and the two-column reconciliation was padded out to four -- which is
    the row of empty cells that appeared in the FY2025A table.
    """
    html = re.sub(r'(\|.+\|[\n\r]*)+', _table_renderer(), TWO_TABLES_BACK_TO_BACK)
    assert html.count('<table') == 2, f"expected 2 tables, got {html.count('<table')}"
    assert '></td>' not in html and '></th>' not in html, 'padded empty cells remain'
    first, second = html.split('<table')[1], html.split('<table')[2]
    assert first.split('</tr>')[0].count('<th') == 2, 'first table is not 2 columns'
    assert second.split('</tr>')[0].count('<th') == 4, 'second table is not 4 columns'


def test_a_single_table_is_unaffected_by_the_split():
    """Splitting must not fragment an ordinary table."""
    html = re.sub(r'(\|.+\|[\n\r]*)+', _table_renderer(), PRICE_TARGET_TABLE)
    assert html.count('<table') == 1, 'a single table was split'


def test_every_chart_label_shares_a_page_with_its_chart(tmp_path):
    """A label must not describe the picture overleaf.

    Heading and image were sibling blocks with nothing binding them, so each
    label sat at the foot of one page and its chart at the top of the next --
    and the final chart printed with no label at all.
    """
    import base64
    import json as _json
    import os
    import subprocess as _sp
    import uuid
    os.environ.setdefault('DATABASE_URL', 'postgresql://localhost/charlie_test')
    import app_v3
    import segment_charts

    pytest.importorskip('xhtml2pdf.pisa')
    if not _sp.run(['which', 'pdftotext'], capture_output=True).stdout:
        pytest.skip('pdftotext not available')

    specs = []
    for period in ('FY2025A', 'FY2026E'):
        specs.append({'kind': 'revenue', 'period': period, 'data': [
            {'segment': 'Alpha', 'value': 143700}, {'segment': 'Beta', 'value': 190400}]})
        specs.append({'kind': 'profit', 'period': period, 'data': [
            {'segment': 'Alpha', 'value': 2935}, {'segment': 'Beta', 'value': 7286}]})
    charts = [{'type': c['type'], 'label': c['label'], 'kind': c['kind'],
               'period': c['period'], 'filename': c['filename'],
               'data': base64.b64encode(c['png']).decode()}
              for c in segment_charts.render_series('PGCK', specs)]
    assert len(charts) == 4

    note_md = '# Pgck Inc (PGCK)\n\n' + ('Body paragraph. ' * 120 + '\n\n') * 6
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM research_notes WHERE ticker='PGCK'")
        cur.execute("""INSERT INTO research_notes (id,ticker,version,note_markdown,
                       sources_markdown,changelog_markdown,note_docx,charts,metadata,status)
                       VALUES (%s,'PGCK',1,%s,' ',' ',' ',%s,%s,'published')""",
                    (str(uuid.uuid4()), note_md, _json.dumps(charts), _json.dumps({})))
    try:
        resp = app_v3.app.test_client().get('/api/notes/PGCK/pdf')
        pdf = tmp_path / 'pgck.pdf'
        pdf.write_bytes(base64.b64decode(resp.get_json()['fileData']))

        pages = int(re.search(r'^Pages:\s+(\d+)', _sp.run(
            ['pdfinfo', str(pdf)], capture_output=True, text=True).stdout,
            re.MULTILINE).group(1))
        seen_labels = 0
        for page in range(1, pages + 1):
            text = _sp.run(['pdftotext', '-f', str(page), '-l', str(page), str(pdf), '-'],
                           capture_output=True, text=True).stdout
            labels = len(re.findall(r'PGCK FY202', text))
            imgs = len(_sp.run(['pdfimages', '-list', '-f', str(page), '-l', str(page),
                                str(pdf)], capture_output=True, text=True
                               ).stdout.splitlines()[2:])
            seen_labels += labels
            if labels and imgs == 0:
                pytest.fail(f'page {page}: {labels} chart label(s) with no chart on the page')
            if imgs and labels == 0:
                pytest.fail(f'page {page}: a chart printed with no label')
        assert seen_labels == 4, f'expected 4 chart labels, found {seen_labels}'
    finally:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute("DELETE FROM research_notes WHERE ticker='PGCK'")


def test_a_rule_immediately_after_a_table_still_renders(tmp_path):
    """"---" right after a table printed literally into the PDF.

    The table match consumes the blank line that follows it, so the replacement
    left "</table>---" and the rule lost its line start -- the ^-anchored regex
    could no longer see it. One separator rendered correctly and the next did
    not, depending purely on whether a table preceded it.
    """
    import base64
    import json as _json
    import os
    import subprocess as _sp
    import uuid
    os.environ.setdefault('DATABASE_URL', 'postgresql://localhost/charlie_test')
    import app_v3

    pytest.importorskip('xhtml2pdf.pisa')
    if not _sp.run(['which', 'pdftotext'], capture_output=True).stdout:
        pytest.skip('pdftotext not available')

    md = ('# Test Co (TST)\n\n'
          '| Timing | Event |\n|---|---|\n| 2027 | Renewals |\n\n'
          '---\n\n## 6. Valuation Context\n\nText.\n\n'
          '---\n\n## 7. Risks\n\nMore.\n')
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM research_notes WHERE ticker='HRTS'")
        cur.execute("""INSERT INTO research_notes (id,ticker,version,note_markdown,
                       sources_markdown,changelog_markdown,note_docx,charts,metadata,status)
                       VALUES (%s,'HRTS',1,%s,' ',' ',' ',%s,%s,'published')""",
                    (str(uuid.uuid4()), md, _json.dumps([]), _json.dumps({})))
    try:
        resp = app_v3.app.test_client().get('/api/notes/HRTS/pdf')
        pdf = tmp_path / 'hr.pdf'
        pdf.write_bytes(base64.b64decode(resp.get_json()['fileData']))
        text = _sp.run(['pdftotext', str(pdf), '-'],
                       capture_output=True, text=True).stdout
        assert '---' not in text, 'a horizontal rule printed as literal dashes'
        assert 'Valuation Context' in text and 'Risks' in text
    finally:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute("DELETE FROM research_notes WHERE ticker='HRTS'")


def test_the_donut_centre_does_not_claim_to_be_a_company_total():
    """It sums reported segments, not the consolidated figure.

    Corporate/Other is excluded (a loss cannot be a pie slice) and segment
    revenue is gross of eliminations, so labelling the centre "Total" invited it
    to be read as consolidated -- a profit donut centred on $16.3B sat beside a
    note stating enterprise adjusted operating income of $14.4B.
    """
    import segment_charts
    png = segment_charts.render_donut(
        'CVS', 'Operating Profit',
        [{'segment': 'A', 'value': 2935}, {'segment': 'B', 'value': 7286}],
        'profit', period='FY2025A')
    assert png and png[:8] == b'\x89PNG\r\n\x1a\n'
    # Assert on the call, not the file: an earlier version of this matched the
    # explanatory comment and failed against correct code.
    src = (ROOT / 'segment_charts.py').read_text(encoding='utf-8')
    calls = [l.strip() for l in src.splitlines()
             if 'ax.text(0, 0.05,' in l and not l.strip().startswith('#')]
    assert calls, 'the centre label call is gone'
    assert all('"Segments"' in c for c in calls), calls
    assert not any('"Total"' in c for c in calls), calls


def test_every_chart_renders_at_the_same_size(tmp_path):
    """A chart landing after a full page of text must not be shrunk to fit.

    -pdf-keep-in-frame-mode:shrink squeezes a block into whatever space is left
    on the page rather than moving it to the next. A CRM note put the FY2026A
    donut at 474 ppi -- the same pixels in half the physical space -- directly
    above an identical FY2027E donut at 239 ppi. The first was unreadable.
    Measured in ppi because that is display scale; the source pixel dimensions
    are identical either way and prove nothing.
    """
    import base64
    import json as _json
    import os
    import re as _re
    import subprocess as _sp
    import uuid
    os.environ.setdefault('DATABASE_URL', 'postgresql://localhost/charlie_test')
    import app_v3
    import segment_charts

    pytest.importorskip('xhtml2pdf.pisa')
    if not _sp.run(['which', 'pdfimages'], capture_output=True).stdout:
        pytest.skip('pdfimages not available')

    specs = [{'kind': 'revenue', 'period': p, 'data': [
        {'segment': 'Agentforce Apps', 'value': 26697},
        {'segment': 'Data 360 & Other', 'value': 12691},
        {'segment': 'Professional Services', 'value': 2137}]}
        for p in ('FY2026A', 'FY2027E')]
    charts = [{'type': c['type'], 'label': c['label'], 'kind': c['kind'],
               'period': c['period'], 'filename': c['filename'],
               'data': base64.b64encode(c['png']).decode()}
              for c in segment_charts.render_series('SZCK', specs)]

    # Text sized to end mid-page, so the first chart meets a partial frame --
    # the condition that triggered the shrink.
    note = '# Szck Inc (SZCK)\n\n' + ('Body text. ' * 95 + '\n\n') * 7
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM research_notes WHERE ticker='SZCK'")
        cur.execute("""INSERT INTO research_notes (id,ticker,version,note_markdown,
                       sources_markdown,changelog_markdown,note_docx,charts,metadata,status)
                       VALUES (%s,'SZCK',1,%s,' ',' ',' ',%s,%s,'published')""",
                    (str(uuid.uuid4()), note, _json.dumps(charts), _json.dumps({})))
    try:
        resp = app_v3.app.test_client().get('/api/notes/SZCK/pdf')
        pdf = tmp_path / 'sz.pdf'
        pdf.write_bytes(base64.b64decode(resp.get_json()['fileData']))
        listing = _sp.run(['pdfimages', '-list', str(pdf)],
                          capture_output=True, text=True).stdout.splitlines()[2:]
        ppis = {int(f[12]) for f in (l.split() for l in listing) if len(f) > 13}
        assert ppis, 'no images found in the PDF'
        assert len(ppis) == 1, (
            f'charts rendered at different scales: {sorted(ppis)} ppi -- one was '
            'shrunk to fit the space left on its page')
    finally:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute("DELETE FROM research_notes WHERE ticker='SZCK'")
