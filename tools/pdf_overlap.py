"""Detect colliding text in a rendered PDF.

The DOM-based preflight cannot see this class of bug: --dump-dom renders in
screen media, and even a print-media preview does not paginate the way the
print pipeline does. The PDF is the artifact the reader gets, so measure that.

Words are grouped into lines; two lines from different columns whose boxes
overlap are reported. Overlapping glyph boxes are, by definition, text printed
on top of other text.
"""
import subprocess, sys, re
from xml.etree import ElementTree as ET


def words_of(pdf, page):
    """Individual word boxes.

    Line boxes are unreliable here: pdftotext groups text sharing a baseline
    across column boundaries, so a bullet in the left column and an annotation
    in the right column merge into one box that appears to overlap everything
    between them. Words never merge, so collisions found between words are
    real.
    """
    xml = subprocess.run(['pdftotext', '-bbox', '-f', str(page), '-l', str(page), pdf, '-'],
                         capture_output=True, text=True).stdout
    root = ET.fromstring(xml)
    W = '{http://www.w3.org/1999/xhtml}word'
    return [(float(w.get('xMin')), float(w.get('yMin')),
             float(w.get('xMax')), float(w.get('yMax')), (w.text or '').strip())
            for w in root.iter(W) if (w.text or '').strip()]


def lines_of(pdf, page):
    xml = subprocess.run(['pdftotext', '-bbox-layout', '-f', str(page), '-l', str(page), pdf, '-'],
                         capture_output=True, text=True).stdout
    root = ET.fromstring(xml)
    ns = {'x': 'http://www.w3.org/1999/xhtml'}
    out = []
    for ln in root.iter('{http://www.w3.org/1999/xhtml}line'):
        ws = [w for w in ln.iter('{http://www.w3.org/1999/xhtml}word')]
        if not ws:
            continue
        xs0 = [float(w.get('xMin')) for w in ws]; xs1 = [float(w.get('xMax')) for w in ws]
        ys0 = [float(w.get('yMin')) for w in ws]; ys1 = [float(w.get('yMax')) for w in ws]
        txt = ' '.join((w.text or '') for w in ws).strip()
        if txt:
            out.append((min(xs0), min(ys0), max(xs1), max(ys1), txt))
    return out


def overlaps(pdf, page, min_frac=0.55):
    """Pairs of lines that genuinely print on top of each other.

    Two things look like collisions but are not. Consecutive wrapped lines in
    one paragraph share a few px because glyph boxes include leading, and a
    small eyebrow label sits inside the leading of a large heading beneath it.
    Both are normal typography. Requiring the intersection to be a majority of
    the smaller box, and the vertical overlap to be at least half the smaller
    line's height, separates those from text actually printed over text.
    """
    ls = words_of(pdf, page)
    hits = []
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            ax0, ay0, ax1, ay1, at = ls[i]
            bx0, by0, bx1, by1, bt = ls[j]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox <= 0.5 or oy <= 0.5:
                continue
            inter = ox * oy
            small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
            min_h = min(ay1 - ay0, by1 - by0)
            # Successive lines of one paragraph share 2-4pt of glyph box because
            # boxes include leading; real overprinting buries a line much
            # deeper. Requiring the vertical overlap to be most of the smaller
            # line's height separates the two.
            if small > 0 and inter / small >= min_frac and oy >= 0.62 * min_h:
                hits.append((round(inter / small, 2), at[:44], bt[:44]))
    return hits


if __name__ == '__main__':
    pdf = sys.argv[1]
    pages = int(subprocess.run(['pdfinfo', pdf], capture_output=True, text=True)
                .stdout.split('Pages:')[1].split()[0])
    total = 0
    for p in range(1, pages + 1):
        hs = overlaps(pdf, p)
        total += len(hs)
        for frac, a, b in hs[:8]:
            print(f'  p{p} overlap {frac:.0%}: "{a}"  <>  "{b}"')
    print(f'{pdf}: {total} colliding line pair(s)')
    sys.exit(1 if total else 0)
