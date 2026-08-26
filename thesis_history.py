"""Longitudinal record of how a thesis changed, and what that drift means.

WHY THIS EXISTS
------------------------------------------------------------------------------
Every thesis refresh used to end in a single irreversible write: you accepted a
set of changes and the reasoning evaporated. That makes the most valuable
question in research unanswerable -- *what did I believe six months ago, and
what changed my mind?*

It also leaves a real gap in the structural/cyclical classifier. That classifier
judges a single proposed change in isolation, so it can only ever *infer* that
something structural is moving. But as the analyst put it: some long-term things
move precisely as an accumulation of short-term ones. A pillar quietly reworded
six quarters running is a structural change in progress, and no single one of
those six edits looks structural on its own.

Recording each revision makes that measurable rather than inferred. `drift()`
counts how often each pillar has actually moved, so `structural_challenge` can
fire on evidence -- "you have revised this margin assumption down three quarters
running" -- instead of on a hunch about one edit.
"""

from __future__ import annotations   # the local agent is still on Python 3.9

import json

# Sections of a thesis that carry labelled rows, in display order.
SECTIONS = ("pillars", "signposts", "threats")

# A pillar that keeps moving is the signal this module exists to surface. Below
# this many revisions it is ordinary maintenance; at or above it, the thesis is
# drifting under you one quarter at a time.
DRIFT_REVISION_THRESHOLD = 3


def ensure_schema(get_db):
    """Create the revisions table. Idempotent, safe to call on every boot."""
    with get_db(commit=True) as (_c, cur):
        cur.execute('''
            CREATE TABLE IF NOT EXISTS thesis_revisions (
                id          SERIAL PRIMARY KEY,
                ticker      VARCHAR(20) NOT NULL,
                source      VARCHAR(30) NOT NULL,
                summary     TEXT,
                counts      JSONB DEFAULT '{}'::jsonb,
                diff        JSONB DEFAULT '{}'::jsonb,
                layers      JSONB DEFAULT '{}'::jsonb,
                snapshot    JSONB,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_thesis_revisions_ticker '
                    'ON thesis_revisions (ticker, created_at DESC)')


def _summarize(counts, source):
    """One human line for the timeline, e.g. '+2 added, 3 reworded (pipeline)'."""
    counts = counts or {}
    bits = []
    if counts.get('added'):
        bits.append(f"+{counts['added']} added")
    if counts.get('changed'):
        bits.append(f"{counts['changed']} reworded")
    if counts.get('removed'):
        bits.append(f"-{counts['removed']} removed")
    if counts.get('conclusion'):
        bits.append('conclusion rewritten')
    return (', '.join(bits) or 'no textual change') + f' ({source})'


def record_revision(get_db, ticker, previous, current, source,
                    diff=None, layers=None, summary=None, max_revisions=200):
    """Append one revision. Returns its id, or None when nothing actually moved.

    `diff` and `layers` are accepted from callers that already computed them --
    the orchestrator has both in hand -- so the work is not repeated. Callers
    without them pass None and get a structural diff computed here.

    Never raises: a failure to journal must not fail the thesis write that
    triggered it. Losing a history row is an annoyance; losing the save is data
    loss.
    """
    try:
        ticker = (ticker or '').upper().strip()
        if not ticker:
            return None

        if diff is None:
            import onepager
            diff = onepager.diff_thesis(previous or {}, current or {})

        counts = dict((diff or {}).get('counts') or {})
        if ((diff or {}).get('conclusion') or {}).get('changed'):
            counts['conclusion'] = 1

        # A refresh that changed nothing is not a revision. Recording it would
        # bury the real changes in a timeline of no-ops -- except for the very
        # first save, which is the thesis coming into existence.
        if not any(counts.values()) and previous:
            return None

        with get_db(commit=True) as (_c, cur):
            cur.execute('''
                INSERT INTO thesis_revisions
                    (ticker, source, summary, counts, diff, layers, snapshot)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
                RETURNING id
            ''', (ticker, source,
                  summary or _summarize(counts, source),
                  json.dumps(counts), json.dumps(diff or {}),
                  json.dumps(layers or {}), json.dumps(current or {})))
            revision_id = cur.fetchone()['id']

            # Keep the journal bounded. The oldest entries are the least useful
            # and the snapshots are the bulky part.
            cur.execute('''
                DELETE FROM thesis_revisions
                 WHERE ticker = %s AND id NOT IN (
                    SELECT id FROM thesis_revisions WHERE ticker = %s
                     ORDER BY created_at DESC, id DESC LIMIT %s)
            ''', (ticker, ticker, max_revisions))
        return revision_id
    except Exception as e:
        print(f"thesis_history.record_revision failed for {ticker}: {e}")
        return None


def list_revisions(get_db, ticker, limit=50):
    """Timeline for one ticker, newest first, without the bulky payloads."""
    with get_db() as (_c, cur):
        cur.execute('''
            SELECT id, source, summary, counts, created_at
              FROM thesis_revisions
             WHERE ticker = %s
             ORDER BY created_at DESC, id DESC
             LIMIT %s
        ''', ((ticker or '').upper().strip(), limit))
        rows = cur.fetchall() or []
    return [{
        'id': r['id'],
        'source': r['source'],
        'summary': r['summary'],
        'counts': _as_dict(r['counts']),
        'createdAt': r['created_at'].isoformat() if r['created_at'] else None,
    } for r in rows]


def get_revision(get_db, revision_id):
    """One revision with its full diff, layer classification and snapshot."""
    with get_db() as (_c, cur):
        cur.execute('SELECT * FROM thesis_revisions WHERE id = %s', (revision_id,))
        row = cur.fetchone()
    if not row:
        return None
    return {
        'id': row['id'],
        'ticker': row['ticker'],
        'source': row['source'],
        'summary': row['summary'],
        'counts': _as_dict(row['counts']),
        'diff': _as_dict(row['diff']),
        'layers': _as_dict(row['layers']),
        'snapshot': _as_dict(row['snapshot']),
        'createdAt': row['created_at'].isoformat() if row['created_at'] else None,
    }


def _as_dict(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return value if value is not None else {}


def drift(get_db, ticker, limit=200, threshold=DRIFT_REVISION_THRESHOLD):
    """How often each labelled row has actually moved, newest activity first.

    This is the measured counterpart to the classifier's judgement. A row that
    keeps being reworded is drifting whatever any single edit was labelled, and
    a row the classifier calls structural while it moves every quarter is the
    exact case worth surfacing: a long-term view eroding by accumulation.
    """
    with get_db() as (_c, cur):
        cur.execute('''
            SELECT id, diff, layers, created_at
              FROM thesis_revisions
             WHERE ticker = %s
             ORDER BY created_at ASC, id ASC
             LIMIT %s
        ''', ((ticker or '').upper().strip(), limit))
        rows = cur.fetchall() or []

    tracked = {}
    for row in rows:
        diff_data = _as_dict(row['diff'])
        layers = _as_dict(row['layers'])
        when = row['created_at'].isoformat() if row['created_at'] else None

        # Map label -> layer for this revision, so a row's classification
        # history travels with it.
        by_label = {}
        for meta in (layers or {}).values():
            if isinstance(meta, dict) and meta.get('label'):
                by_label[_key(meta['label'])] = meta.get('layer')

        for section in SECTIONS:
            section_diff = (diff_data or {}).get(section) or {}
            for kind in ('added', 'changed', 'removed'):
                for item in section_diff.get(kind) or []:
                    label = _item_label(item)
                    if not label:
                        continue
                    entry = tracked.setdefault(_key(label), {
                        'label': label, 'section': section, 'revisions': 0,
                        'kinds': [], 'layers': [], 'firstSeen': when, 'lastChanged': when,
                        # Which revisions actually moved this row, so the UI can
                        # offer "show me those" rather than stating a count the
                        # reader has no way to investigate.
                        'revisionIds': [],
                    })
                    entry['revisions'] += 1
                    entry['kinds'].append(kind)
                    entry['lastChanged'] = when
                    if row['id'] not in entry['revisionIds']:
                        entry['revisionIds'].append(row['id'])
                    layer = by_label.get(_key(label))
                    if layer and layer not in entry['layers']:
                        entry['layers'].append(layer)

    out = []
    for entry in tracked.values():
        # 'added' then later edits is normal maturation; what matters is how many
        # times the row has been *revised* since it first appeared.
        entry['timesRevised'] = max(0, entry['revisions'] - entry['kinds'].count('added'))
        entry['accumulating'] = entry['timesRevised'] >= threshold
        # The case worth surfacing: a row the classifier kept calling structural
        # while it moved anyway -- a long-term view eroding one edit at a time.
        entry['structuralDrift'] = (
            entry['accumulating'] and 'structural' in entry['layers'])
        out.append(entry)

    out.sort(key=lambda e: (-e['timesRevised'], e['label'].lower()))
    return out


def _item_label(item):
    """Label for a diff row. `changed` rows carry it explicitly; the rest are
    raw pillars/signposts/threats that onepager._label already knows how to read."""
    import onepager
    if isinstance(item, dict):
        if item.get('label'):
            return str(item['label'])
        return onepager._label(item)
    return str(item or '')


def _key(label):
    return ' '.join(str(label or '').lower().split())
