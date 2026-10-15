"""Render frozen source provenance without fetching, inferring or changing research."""
from datetime import date
import html
import re

KINDS = {'transcript': 'Transcript', 'broker-report': 'Broker report',
         'press-release': 'Press release', 'presentation': 'Presentation'}


def frozen_context(source, assignment):
    """Report IDs differ from document IDs; require an unambiguous frozen byte match."""
    sha = source.get('originalHash')
    name = source.get('filename')
    if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha) or not isinstance(name, str) or not name:
        return None
    pack = assignment.get('labPack')
    context = assignment.get('processingSourceContext')
    if not isinstance(pack, list) or not isinstance(context, dict):
        return None
    matches = [s for s in pack if isinstance(s, dict) and s.get('filename') == name and s.get('sha256') == sha]
    if len(matches) != 1:
        return None
    ident = matches[0].get('documentId')
    if not isinstance(ident, str) or not re.fullmatch('[1-9][0-9]{0,18}', ident):
        return None
    # A repeated document identity makes its context ambiguous even if one hash matches.
    if sum(isinstance(s, dict) and s.get('documentId') == ident for s in pack) != 1:
        return None
    entry = context.get(ident)
    return entry if isinstance(entry, dict) else None


def render(source, assignment):
    esc = lambda value: html.escape(str(value), quote=True)
    lines = []
    entry = frozen_context(source, assignment)
    if entry is None:
        lines.append('Selection role and publication date unavailable: no unambiguous frozen selection metadata matches this source. Current library metadata has not been substituted.')
    else:
        role = {'collection': 'Collected for the linked request', 'supplemental': 'Supplemental history; not collected by the linked request'}.get(entry.get('role'), 'Selection role not recorded')
        published = entry.get('publishedDate')
        try:
            published = date.fromisoformat(published).isoformat() if isinstance(published, str) else 'unknown'
        except ValueError:
            published = 'unknown'
        lines.append('Frozen selection: ' + role + '. Published: ' + published + '.')
        publisher = entry.get('publisher')
        if isinstance(publisher, str) and publisher:
            lines.append('Publisher: ' + publisher)
        lines.append('Recorded source type: ' + KINDS.get(entry.get('sourceKind'), 'unknown'))
        referenced = entry.get('referencedBySavedThesis')
        lines.append('Thesis register at processing proposal save: ' + (
            'exact-byte reference found; incorporation and current validity are not established.' if referenced is True else
            'no exact-byte reference found; this does not establish that the evidence is new or unused.' if referenced is False else
            'reference status not recorded; incorporation is unknown.'))
    coverage = source.get('coverage')
    if isinstance(coverage, dict):
        limitation = coverage.get('limitation')
        if isinstance(limitation, str) and limitation.strip():
            lines.append('Research context limitation: ' + limitation)
        elif coverage.get('mode') == 'bounded_excerpts' or (type(coverage.get('omittedCharacters')) in (int, float) and coverage['omittedCharacters'] > 0):
            lines.append('Research context limitation: some extracted text was omitted and was not reviewed.')
    extraction = source.get('textExtraction')
    limitations = extraction.get('limitations') if isinstance(extraction, dict) else None
    if isinstance(limitations, list):
        lines.extend('Original extraction limitation: ' + text for text in dict.fromkeys(
            text for text in limitations if isinstance(text, str) and text.strip()))
    return '<div class="source-provenance" style="overflow-wrap:anywhere">' + ''.join('<p>' + esc(line) + '</p>' for line in lines) + '</div>'
