"""Read-only coverage of frozen selections; never infer thesis incorporation."""
from datetime import date

KINDS = ('transcript', 'broker-report', 'press-release', 'presentation')


def stamp(value):
    try:
        return date.fromisoformat(value).isoformat() if isinstance(value, str) else None
    except ValueError:
        return None


def summarize(frozen):
    sources = frozen['sources']
    context = frozen.get('sourceContext') or {}
    plan = frozen['plan']
    since, until = stamp(plan.get('since')), stamp(plan.get('until'))
    roles = dict(collection=0, supplemental=0, unknown=0)
    kinds = {k: 0 for k in (*KINDS, 'unknown')}
    references = dict(referenced=0, noReference=0, unknown=0)
    dates = [];before = within = after = recorded = 0
    for source in sources:
        entry = context.get(source['documentId'])
        if isinstance(entry, dict): recorded += 1
        else: entry = {}
        role = entry.get('role');roles[role if role in ('collection', 'supplemental') else 'unknown'] += 1
        kind = entry.get('sourceKind');kinds[kind if kind in KINDS else 'unknown'] += 1
        reference = entry.get('referencedBySavedThesis')
        references['referenced' if reference is True else 'noReference' if reference is False else 'unknown'] += 1
        published = stamp(entry.get('publishedDate'))
        if published:
            dates.append(published)
            if since and until:
                if published < since: before += 1
                elif published > until: after += 1
                else: within += 1
    return dict(version=1, total=len(sources), metadataRecorded=recorded,
                collectionWindow={'since': since, 'until': until}, roles=roles, kinds=kinds,
                publication={'earliest': min(dates) if dates else None, 'latest': max(dates) if dates else None,
                             'known': len(dates), 'unknown': len(sources)-len(dates),
                             'beforeWindow': before, 'withinWindow': within, 'afterWindow': after},
                thesisReferences=references, incorporation='not_assessed', exhaustiveCoverage='not_assessed')
