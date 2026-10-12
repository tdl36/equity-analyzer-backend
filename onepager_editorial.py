"""Stable evidence ordering for one-pagers, independent of HTML layout.

Existing report order is retained, not certified as materiality ranking. Source
count, numeric density, statement length and sector keywords are not proxies for
investment importance. The renderer separately discloses omissions for space.
"""
import re


def ordered_candidates(citations, prefixes, exclude=()):
    """Return supported path/claim pairs by category priority and numeric row.

    Callers provide already validated citations. Never mutate or summarize their
    statements, merge evidence from different rows, or infer a company sector.
    """
    def order(item):
        path=item[0]
        group=next(i for i,prefix in enumerate(prefixes) if path.startswith(prefix))
        suffix=path[len(prefixes[group]):]
        # Tagged components keep comparisons safe for mixed numeric/text paths.
        components=tuple((0,int(part)) if part.isdigit() else (1,part)
                         for part in suffix.split('/'))
        return group,components

    items=[(path,claim) for path,claim in citations.items()
           if any(path.startswith(prefix) for prefix in prefixes)
           and not any(path.endswith(suffix) for suffix in exclude)]
    return sorted(items,key=order)


def statement_identity(text):
    """Only suppress repeated text; shared sector terminology is not duplication."""
    return re.sub(r'\s+',' ',text).strip().casefold()
