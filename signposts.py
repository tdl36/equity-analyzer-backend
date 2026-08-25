"""Signpost monitoring -- makes a thesis watch itself between refreshes.

WHY THIS EXISTS
------------------------------------------------------------------------------
A thesis already names the things that would prove it right or wrong: signposts,
each with what it is today and what it needs to reach. Until now nothing ever
looked at them again. They were re-read at the next refresh, months later, by
which point the answer to "did that happen?" had to be reconstructed from memory.

This evaluates signposts against material as it arrives and raises an alert when
one is hit, breaks, or starts moving the wrong way. The thesis stops being a
document you remember to re-read and becomes something that taps you on the
shoulder.

WHAT IT DELIBERATELY DOES NOT DO
------------------------------------------------------------------------------
It does not fire on absence of evidence. A quarter that says nothing about a
signpost is not a signpost that broke, and an alerting system that cries wolf on
silence gets muted within a week -- at which point it is worse than nothing,
because the real alert arrives muted too. `no_evidence` is a first-class outcome
and never alerts.

It also does not fire without a quote. Every alert carries the sentence from the
source that justifies it, so the first question -- "says who?" -- is already
answered.
"""

from __future__ import annotations

import hashlib
import json

# Outcomes that are worth interrupting someone for. `approaching` is included
# because a signpost about to trigger is usually more actionable than one that
# already has.
ALERTING_STATUSES = ('hit', 'broken', 'approaching')

VALID_STATUSES = ('hit', 'broken', 'approaching', 'on_track', 'no_evidence')

SYSTEM_PROMPT = """You check investment thesis signposts against new source material.

A signpost is a pre-committed observable: something the analyst said they would
watch, with a current reading and a target. Your job is to say what the new
material shows about each one. Nothing else.

For every signpost return exactly one status:
  hit          - the target has been reached or passed
  broken       - it moved decisively AGAINST the thesis, or the premise is void
  approaching  - moved materially toward the target but not there yet
  on_track     - mentioned, consistent with expectations, nothing notable
  no_evidence  - the material does not speak to this signpost

RULES, IN ORDER OF IMPORTANCE
1. NEVER invent a number. Every figure must appear in the source material.
2. If the material does not address a signpost, the answer is `no_evidence`.
   Silence is not evidence of failure. Do not reason about what the absence
   might imply. This is the single most common way this task is done badly.
3. Every status other than `no_evidence` MUST carry `evidence`: a short verbatim
   quote from the source. No quote means `no_evidence`, whatever you inferred.
4. `observed` is what the material actually says the value is now -- quoted or
   directly computed from quoted figures. Empty if the material gives no value.
5. Judge against the signpost's stated target, not your own view of the company.
6. `confidence` is high only when a quoted figure maps directly onto the target.
   Use low when you are reading across a proxy or a qualitative statement.

Return JSON only:
{"evaluations": [{"signpost": "<exact signpost text as given>",
                  "status": "hit|broken|approaching|on_track|no_evidence",
                  "observed": "", "evidence": "", "why": "",
                  "confidence": "high|medium|low"}]}"""


def _signpost_label(item):
    if isinstance(item, dict):
        for key in ('signpost', 'metric', 'title', 'name'):
            if item.get(key):
                return str(item[key])
        return ''
    return str(item or '')


def build_prompt(signposts, material, ticker=''):
    """(items, user_message) -- items are the signposts actually being checked."""
    items = []
    for sp in signposts or []:
        label = _signpost_label(sp)
        if not label:
            continue
        d = sp if isinstance(sp, dict) else {}
        items.append({
            'signpost': label,
            # Thesis signposts are written as {metric, target, timeframe,
            # category}; the one-pager schema uses {signpost, current, target,
            # why}. Both shapes reach here, so read whichever keys are present.
            'current': d.get('current') or d.get('currentValue') or '',
            'target': d.get('target') or '',
            'timeframe': d.get('timeframe') or '',
            'why': d.get('why') or d.get('description') or d.get('category') or '',
        })
    if not items:
        return [], ''

    lines = [f'TICKER: {ticker}', '', 'SIGNPOSTS TO CHECK:']
    for it in items:
        lines.append(f"- {it['signpost']}")
        if it['current']:
            lines.append(f"    current as of the thesis: {it['current']}")
        if it['target']:
            lines.append(f"    target: {it['target']}"
                         f"{(' by ' + it['timeframe']) if it['timeframe'] else ''}")
        if it['why']:
            lines.append(f"    context: {it['why']}")
    lines += ['', 'NEW SOURCE MATERIAL:', '', (material or '')[:60000]]
    return items, '\n'.join(lines)


def evaluate(ticker, signposts, material, call_llm, extract_json,
             api_keys=None, tier='standard'):
    """Check each signpost against new material. Returns a list of evaluations.

    Returns [] on any failure. Monitoring is an assist: a model outage must not
    surface as a wave of spurious 'broken' alerts, so the safe failure is silence.
    """
    items, user_msg = build_prompt(signposts, material, ticker)
    if not items:
        return []

    keys = api_keys or {}
    try:
        result = call_llm(
            messages=[{'role': 'user', 'content': user_msg}],
            system=SYSTEM_PROMPT, tier=tier, max_tokens=4096, timeout=180,
            anthropic_api_key=keys.get('anthropic', ''),
            gemini_api_key=keys.get('gemini', ''),
            openai_api_key=keys.get('openai', ''),
        )
        parsed = extract_json(result['text']) or {}
    except Exception as e:
        print(f'[signposts] evaluation failed for {ticker}: {e}')
        return []

    by_label = {_norm(e.get('signpost')): e for e in parsed.get('evaluations', [])
                if isinstance(e, dict)}

    out = []
    for it in items:
        got = by_label.get(_norm(it['signpost'])) or {}
        status = got.get('status')
        if status not in VALID_STATUSES:
            status = 'no_evidence'
        evidence = (got.get('evidence') or '').strip()

        # Rule 3, enforced rather than trusted. A status without a supporting
        # quote is an inference, and inferences must not page anyone.
        if status != 'no_evidence' and not evidence:
            status = 'no_evidence'

        out.append({
            'signpost': it['signpost'],
            'target': it['target'],
            'previous': it['current'],
            'status': status,
            'observed': (got.get('observed') or '').strip(),
            'evidence': evidence,
            'why': (got.get('why') or '').strip(),
            'confidence': got.get('confidence') if got.get('confidence') in
                          ('high', 'medium', 'low') else 'low',
        })
    return out


def _norm(text):
    return ' '.join(str(text or '').lower().split())


def alert_id(ticker, signpost, status, observed):
    """Stable id for one finding, so the same reading cannot alert twice.

    Keyed on the observation as well as the status: a signpost that moves again
    is genuinely new news, but the same reading re-checked is not.
    """
    raw = f'{(ticker or "").upper()}|{_norm(signpost)}|{status}|{_norm(observed)}'
    return 'signpost-' + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]


def record_alerts(get_db, ticker, evaluations, source=''):
    """Insert an alert per notable finding. Returns the alerts created.

    Deduplicated on content, so re-running a check over the same documents is
    free and produces no repeat noise.
    """
    ticker = (ticker or '').upper().strip()
    created = []
    notable = [e for e in (evaluations or []) if e.get('status') in ALERTING_STATUSES]
    if not notable:
        return created

    verb = {'hit': 'reached its target',
            'broken': 'broke',
            'approaching': 'is approaching its target'}

    with get_db(commit=True) as (_c, cur):
        for ev in notable:
            aid = alert_id(ticker, ev['signpost'], ev['status'], ev.get('observed', ''))
            cur.execute('SELECT id FROM agent_alerts WHERE id = %s', (aid,))
            if cur.fetchone():
                continue
            title = f"{ticker}: signpost {verb.get(ev['status'], 'moved')} — {ev['signpost']}"
            detail = dict(ev)
            detail['source'] = source
            cur.execute('''
                INSERT INTO agent_alerts (id, alert_type, ticker, title, detail, status, created_at)
                VALUES (%s, 'signpost', %s, %s, %s::jsonb, 'new', NOW())
            ''', (aid, ticker, title[:500], json.dumps(detail)))
            created.append({'id': aid, 'title': title, **ev})
    return created


def summarize(evaluations):
    """Counts by status, for a compact status line."""
    counts = {s: 0 for s in VALID_STATUSES}
    for ev in evaluations or []:
        status = ev.get('status')
        if status in counts:
            counts[status] += 1
    counts['notable'] = sum(counts[s] for s in ALERTING_STATUSES)
    return counts
