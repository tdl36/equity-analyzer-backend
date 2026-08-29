"""Deep Dive — one canonical research run, three deliberately different artifacts.

Ported from the Investment Research Studio v24 prototype. The prototype was a
local Flask app; this is the same *product* rebuilt on Charlie's stack:

    prototype                    here
    ---------------------------  ------------------------------------------
    OpenAI Responses+web_search   Charlie's call_llm + Tavily search results
    yfinance snapshot             yfinance (already a Charlie dependency)
    in-memory 30-min CACHE        Postgres, so runs survive a restart
    synchronous request           Charlie's threaded-job pattern

THE ONE RULE THAT SHAPES EVERYTHING
------------------------------------------------------------------------------
There is exactly ONE canonical research object per run, and all three artifacts
render from it. The one-pager is an *editorial compression* of that object, not
a second research pass -- otherwise the one-pager and the memo drift into
disagreeing about the same company, which is worse than either being wrong
alone. compress_onepager() is therefore forbidden from adding facts.

WHY THE BUDGETS ARE ENFORCED IN CODE
------------------------------------------------------------------------------
The one-pager is a FIXED 1024x1536 canvas. Text that overruns does not reflow,
it clips -- and a silently clipped kill-criterion is a research artifact that
lies. The editor prompt states budgets, but a model will exceed them, so
onepager_violations() measures the returned object and repair is attempted
before anything reaches a renderer.
"""

from __future__ import annotations   # the local agent still runs Python 3.9

import json
import re
from datetime import datetime, timezone

import deepdive_prompts

SCHEMA_VERSION = "v24"

# The prototype cached for 30 minutes. Kept, but now persisted: a restart used
# to throw away a research run that cost real money and minutes to produce.
CACHE_TTL_MINUTES = 30

# Editorial budgets from ONEPAGER_EDITOR_SYSTEM, restated as machine-checkable
# limits. The prompt asks; this enforces. (field, max_words) for prose and
# (field, exact_count) for lists that the renderers assume are fixed-length.
# CALIBRATED TO THE ARTIFACT THAT FITS, NOT TO TASTE.
#
# The one-pager's boxes are fixed foreignObject rectangles at absolute
# coordinates; content that exceeds them does not reflow, it overlaps the next
# section. So the only budget that means anything is the volume the reviewed DE
# golden fixture actually uses, plus modest headroom.
#
# The first version of these numbers was ~2.5x too generous (thesis_summary
# allowed 115 words where the fitting artifact uses 42), so a fully "compliant"
# one-pager still overlapped its own sections. UNH came back at roughly double
# DE in every field and passed with six trivial violations.
#
# Format: (field, max_words). DE actuals are in the comments so the next person
# can see where each number came from.
_WORD_BUDGETS = [
    ("headline", 10),               # DE 6
    ("subheadline", 16),            # DE 11
    ("core_question", 24),          # DE 16
    ("overview_summary", 34),       # DE 22
    ("other_profit_pool", 20),      # DE 15
    ("valuation_callout", 22),      # DE 17
    ("final_takeaway", 50),         # DE 38
    ("bottom_line", 12),            # DE 9
    ("secondary_bottom_line", 7),   # DE 5
]

# Renderers index into these directly, so a short list is a layout hole and a
# long one overflows the canvas.
_EXACT_COUNTS = [
    ("thesis_bullets", 5),
    ("business_model", 4),
    ("opportunities", 5),
    ("financial_bullets", 6),
    ("valuation_metrics", 3),
    ("signposts", 6),
    ("threats", 4),
    ("bull_case", 5),
    ("bear_case", 5),
]

# Tightened after a live UNH run: at 14/12/6 the fifth thesis bullet ran past
# its box, the sixth financial bullet collided with the earnings-chart title,
# and the bull/bear tails clipped. The boxes hold slightly less than those caps
# implied, so these sit closer to the DE reference.
# Word count is only a proxy for what actually matters, which is rendered LINES.
# UNH's bullets carry figures like "(85.2% Q3'24 vs 84.1% prior)" that wrap to
# two lines at the same word count Deere's fit on one, so the fifth thesis
# bullet and the sixth financial bullet still overran. Pinned to the DE actuals.
_LIST_ITEM_WORDS = [
    ("thesis_bullets", 11),      # DE max item 11
    ("financial_bullets", 9),    # DE max item 9
    ("bull_case", 5),            # DE max item 4
    ("bear_case", 5),            # DE max item 4
]

# Per-item limits inside object lists, same derivation.
_NESTED_ITEM_WORDS = [
    ("opportunities", "detail", 12),      # DE max 9
    ("business_model", "description", 9), # DE max 6
    ("segments", "description", 11),      # DE max 8
    ("threats", "watch_for", 22),         # DE max 17
]


# Character caps, because words do not predict line wrapping and lines are what
# actually overflow a fixed box. The DE reference bullets run 60-76 characters
# and fit; UNH produced 84-character bullets at the SAME word count -- figures
# like "(85.2% Q3'24 vs 84.1% prior)" are long strings, not extra words -- and
# the fifth bullet wrapped to a second line and fell out of its box.
_LIST_ITEM_CHARS = [
    # Exactly the DE maximum. 72 was tried and it trimmed the DE reference itself,
    # which would mean degrading the calibration standard to make another company
    # fit -- the wrong trade. 76 leaves DE untouched and still pulls UNH in.
    ("thesis_bullets", 76),
    ("financial_bullets", 72),    # DE max 69
]

_NESTED_ITEM_CHARS = [
    ("opportunities", "detail", 74),
    ("business_model", "description", 56),
]


def _chars(text):
    return len(str(text or "").strip())


def _trim_chars(text, limit):
    """Cut to `limit` characters on a word boundary."""
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return text
    cut = raw[:limit].rsplit(" ", 1)[0].rstrip(",;:.")
    return (cut or raw[:limit]) + "\u2026"


def _words(text):
    return len([w for w in re.split(r"\s+", str(text or "").strip()) if w])


def onepager_violations(d):
    """Editorial budget breaches, as human-readable strings.

    Returns [] for a clean object. Used both to gate rendering and to tell the
    model precisely what to fix on the repair pass -- a generic "too long"
    retry wastes a call and usually comes back long again.
    """
    d = d or {}
    out = []

    for field, limit in _WORD_BUDGETS:
        n = _words(d.get(field))
        if n > limit:
            out.append(f"{field}: {n} words (max {limit})")

    # The editor prompt asks for 80-115 words, but only the upper bound is a
    # real failure: on a fixed canvas an over-long summary clips a kill
    # criterion off the page, while a short one merely leaves white space. The
    # reference-calibrated DE golden one-pager runs ~42 words here on purpose --
    # the space goes to the core-question box and the bullets -- so treating
    # under-length as a violation would condemn the calibration standard itself.
    thesis_words = _words(d.get("thesis_summary"))
    if thesis_words > 60:                       # DE 42
        out.append(f"thesis_summary: {thesis_words} words (max 60)")

    for field, count in _EXACT_COUNTS:
        got = d.get(field)
        n = len(got) if isinstance(got, list) else 0
        if n != count:
            out.append(f"{field}: {n} items (need exactly {count})")

    for field, limit in _LIST_ITEM_WORDS:
        for i, item in enumerate(d.get(field) or []):
            n = _words(item)
            if n > limit:
                out.append(f"{field}[{i}]: {n} words (max {limit})")

    for field, limit in _LIST_ITEM_CHARS:
        for i, item in enumerate(d.get(field) or []):
            n = _chars(item)
            if n > limit:
                out.append(f"{field}[{i}]: {n} chars (max {limit}, wraps otherwise)")

    for field, sub, limit in _NESTED_ITEM_CHARS:
        for i, item in enumerate(d.get(field) or []):
            if isinstance(item, dict) and _chars(item.get(sub)) > limit:
                out.append(f"{field}[{i}].{sub}: {_chars(item.get(sub))} chars (max {limit})")

    # Signpost cells are table cells on a fixed-width canvas; long ones wrap
    # the row taller and push the last signpost off the page.
    for i, sp in enumerate(d.get("signposts") or []):
        if not isinstance(sp, dict):
            continue
        for cell in ("signpost", "current", "target", "why"):
            n = _words(sp.get(cell))
            if n > 9:                           # DE max 7
                out.append(f"signposts[{i}].{cell}: {n} words (max 9)")

    for field, sub, limit in _NESTED_ITEM_WORDS:
        for i, item in enumerate(d.get(field) or []):
            if not isinstance(item, dict):
                continue
            n = _words(item.get(sub))
            if n > limit:
                out.append(f"{field}[{i}].{sub}: {n} words (max {limit})")

    return out


def _trim_words(text, limit):
    """Cut to `limit` words on a word boundary, with an ellipsis when cut."""
    words = [w for w in re.split(r"\s+", str(text or "").strip()) if w]
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(",;:.") + "\u2026"


def enforce_budgets(d):
    """Hard-trim a one-pager to its budgets. Returns (object, trimmed_fields).

    The prompt asks and the repair pass retries, but neither is a guarantee, and
    the consequence of an overrun is not a slightly long paragraph -- the
    one-pager's boxes are fixed rectangles at absolute coordinates, so excess
    text silently overlaps the next section and the artifact becomes unreadable.

    A universal template cannot depend on the model behaving. This is the last
    line of defence: deterministic, applied on save, and reported so the overrun
    is visible rather than hidden. Trimming loses the tail of a sentence;
    not trimming loses whole sections behind other sections.
    """
    if not isinstance(d, dict):
        return d, []
    out = dict(d)
    trimmed = []

    for field, limit in _WORD_BUDGETS:
        new = _trim_words(out.get(field), limit)
        if new != out.get(field):
            out[field] = new
            trimmed.append(field)

    if _words(out.get("thesis_summary")) > 60:
        out["thesis_summary"] = _trim_words(out.get("thesis_summary"), 60)
        trimmed.append("thesis_summary")

    for field, limit in _LIST_ITEM_WORDS:
        items = out.get(field)
        if isinstance(items, list):
            new_items = [_trim_words(x, limit) for x in items]
            if new_items != items:
                out[field] = new_items
                trimmed.append(field)

    for field, sub, limit in _NESTED_ITEM_WORDS:
        items = out.get(field)
        if isinstance(items, list):
            new_items = []
            changed = False
            for x in items:
                if isinstance(x, dict) and _words(x.get(sub)) > limit:
                    x = dict(x)
                    x[sub] = _trim_words(x.get(sub), limit)
                    changed = True
                new_items.append(x)
            if changed:
                out[field] = new_items
                trimmed.append(f"{field}.{sub}")

    for field, limit in _LIST_ITEM_CHARS:
        items = out.get(field)
        if isinstance(items, list):
            new_items = [_trim_chars(x, limit) for x in items]
            if new_items != items:
                out[field] = new_items
                trimmed.append(f"{field}(width)")

    for field, sub, limit in _NESTED_ITEM_CHARS:
        items = out.get(field)
        if isinstance(items, list):
            new_items, changed = [], False
            for x in items:
                if isinstance(x, dict) and _chars(x.get(sub)) > limit:
                    x = dict(x)
                    x[sub] = _trim_chars(x.get(sub), limit)
                    changed = True
                new_items.append(x)
            if changed:
                out[field] = new_items
                trimmed.append(f"{field}.{sub}(width)")

    fixed_segs, seg_changed = normalize_segments(out.get("segments"))
    if seg_changed:
        out["segments"] = fixed_segs
        trimmed.append("segments(rescaled to 100%)")

    signposts = out.get("signposts")
    if isinstance(signposts, list):
        new_sps, changed = [], False
        for sp in signposts:
            if isinstance(sp, dict):
                sp2 = dict(sp)
                for cell in ("signpost", "current", "target", "why"):
                    if _words(sp2.get(cell)) > 9:
                        sp2[cell] = _trim_words(sp2.get(cell), 9)
                        changed = True
                sp = sp2
            new_sps.append(sp)
        if changed:
            out["signposts"] = new_sps
            trimmed.append("signposts")

    # Fixed-length lists: the renderers index into these, so a long list pushes
    # rows off the canvas and a short one leaves a hole.
    for field, count in _EXACT_COUNTS:
        items = out.get(field)
        if isinstance(items, list) and len(items) > count:
            out[field] = items[:count]
            trimmed.append(f"{field}(count)")

    return out, sorted(set(trimmed))


# MEMO BUDGETS.
#
# THESE ARE A BACKSTOP, NOT A FIT MECHANISM.
#
# They were briefly tightened to Deere parity to force UNH's memo to fit, and
# printed exactly what that implies: sentences cut mid-clause -- "Bear: Aetna
# margin...", "must translate...", "Signify's...". That is deleting research to
# fit a box, and it reads as broken rather than concise.
#
# Fitting is mostly autoFitSections()' job: it scales an overflowing section
# down to 0.86 and keeps every word. The floor is not lowered further because
# below it the memo's body type drops under the readability threshold the
# handoff treats as non-negotiable -- so the two sections that still overran at
# the floor (overview, financial) carry moderate caps, at roughly 2-3x Deere
# rather than the parity that produced mid-sentence cuts. Everything else exists
# only to stop genuinely absurd output -- a 190-word "bottom line", seven threats in a grid built for
# four -- and sit well clear of what a normal company returns. Where a limit
# does bite, it should be cutting waffle, not the end of an argument.
#
# The two-pager and the 3-page memo render from the MASTER object, which had no
# budgets at all -- only the one-pager was ever compressed. So a verbose company
# walked straight into fixed-height memo sections: UNH returned a 190-word final
# takeaway against Deere's 39, a 111-word "bottom line" against Deere's 6, and
# SEVEN thesis threats into a kill-criteria grid built for four.
#
# The memo has more room than the one-pager's fixed boxes, so these are looser
# than the one-pager budgets -- roughly the DE master plus 40% -- but they are
# bounded, which is the point. DE actuals in the comments.
_MASTER_WORD_BUDGETS = [
    # Measured against the DE master, which renders with zero overflow. Every
    # cap sits ~30% above DE's own value; where a field had no cap at all the
    # gap was enormous -- variant_view came back at 86 words against Deere's 20,
    # and it feeds both the thesis and decision sections.
    (("investment_thesis", "summary"), 260),        # DE 118; research prompt asks 160-240
    (("investment_thesis", "variant_view"), 70),    # DE 20
    (("company_overview", "summary"), 100),         # DE 55
    (("final_takeaway",), 110),                     # DE 39
    (("bottom_line",), 30),                         # DE 6
    # The valuation panel and metric captions on memo page 2. All were
    # uncapped, and a live run returned a 101-word valuation_comment against
    # Deere's 14 -- the panel rendered 787px tall inside a 390px box, which was
    # the single largest source of clipping anywhere in the memo.
    (("financial_snapshot", "valuation_comment"), 40),   # DE 14
    (("financial_snapshot", "revenue_context"), 26),     # DE 5
    (("financial_snapshot", "margin_context"), 26),      # DE 5
    (("financial_snapshot", "eps_context"), 26),         # DE 7
    (("financial_snapshot", "fcf_context"), 26),         # DE 2
    (("financial_snapshot", "returns"), 20),             # DE 5
    (("financial_snapshot", "leverage"), 20),            # DE 5
    (("financial_snapshot", "historical_pe"), 20),       # DE 6
    # The note under the earnings chart on every artifact. Uncapped, and the
    # largest remaining gap: 61 words against Deere's 13.
    (("earnings_history", "cycle_note"), 45),            # DE 13
    (("investment_thesis", "core_question"), 45),        # DE 16
    (("tagline",), 24),                                  # DE 7
    # The headline numbers on the six KPI cards. Uncapped, and a live run
    # returned values like "$445-448 billion (2025 guidance)" where Deere has
    # "$51.7B" -- each card grew to 165px against Deere's ~110, which is the
    # whole of the remaining page-2 overflow. These are figures, not sentences;
    # the qualifier belongs in the card's context line.
    (("financial_snapshot", "revenue"), 8),
    (("financial_snapshot", "operating_margin"), 8),
    (("financial_snapshot", "eps"), 8),
    (("financial_snapshot", "free_cash_flow"), 8),
    (("financial_snapshot", "forward_pe"), 8),
    (("financial_snapshot", "ev_ebitda"), 8),
    (("financial_snapshot", "fcf_yield"), 8),
]

# (path, exact_count) -- the memo renderers lay these out in fixed grids, so a
# long list overruns the page and a short one leaves a hole.
_MASTER_LIST_CAPS = [
    (("investment_thesis", "what_market_prices_in"), 3, 30),   # DE max 9
    (("investment_thesis", "what_must_be_true"), 3, 30),       # DE max 9
    (("investment_thesis", "falsification"), 3, 30),           # DE max 10
    # The two-pager renders these into a fixed box and they were never capped:
    # they drove tp-financial 41px past its bounds on the live UNH run.
    # These become the six KPI cards on memo page 2 and the bullet column on the
    # two-pager. At 14 words they wrapped to extra lines and pushed the metrics
    # block from Deere's 231px to 336px, which is most of the overflow on that
    # page. DE's own maximum is 10.
    (("financial_snapshot", "financial_bullets"), 6, 14),
    (("opportunities",), 5, None),
    (("business_model",), 4, None),
    (("signposts",), 6, None),
    (("thesis_threats",), 4, None),
    (("catalysts",), 3, None),
    # These render as telegraphic lines, not sentences: Deere averages 4 words.
    (("bull_case",), 5, 18),   # DE has 5
    (("bear_case",), 5, 18),   # DE has 5
]

# Per-item prose inside those lists.
_MASTER_ITEM_WORDS = [
    # Uncapped until a live run returned 40-word segment descriptions against
    # Deere's 8, which is most of why the memo's overview section ran 170px past
    # its box. These sit in a three-across grid; they are labels, not prose.
    (("company_overview", "segments"), "description", 13),   # DE max 8
    # a four-segment company has to fit the space Deere's three occupy
    (("opportunities",), "detail", 40),             # DE max 14
    (("business_model",), "description", 30),       # DE max 10
    (("thesis_threats",), "watch_for", 45),         # DE max 17
    (("signposts",), "why_it_matters", 42),         # DE max 7
    (("catalysts",), "why_it_matters", 26),         # DE max 8
    (("financial_snapshot", "management_targets"), "context", 20),  # DE max 2
    # Never capped before, and the worst offender: UNH returned 50-word scenario
    # logic against Deere's 6, which is most of why the decision section on page
    # 3 ran 127px past its box.
    (("valuation_scenarios",), "logic", 26),        # DE max 6
]


def _dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _put(d, path, value):
    cur = d
    for k in path[:-1]:
        nxt = cur.get(k)
        cur[k] = dict(nxt) if isinstance(nxt, dict) else {}
        cur = cur[k]
    cur[path[-1]] = value


# Fields the renderers call .slice()/.map() on. A model that wraps one of these
# in an object does not produce a slightly-off artifact, it throws and the whole
# page renders blank.
_MUST_BE_LIST = [
    ("signposts",), ("thesis_threats",), ("opportunities",), ("business_model",),
    ("catalysts",), ("bull_case",), ("bear_case",), ("valuation_scenarios",),
    ("sources",), ("company_overview", "segments"),
    ("company_overview", "other_profit_pools"),
    ("financial_snapshot", "financial_bullets"),
    ("financial_snapshot", "management_targets"),
    ("investment_thesis", "what_market_prices_in"),
    ("investment_thesis", "what_must_be_true"),
    ("investment_thesis", "falsification"),
]


def coerce_master_shape(m):
    """Repair structural slips before anything renders. Returns (obj, fixed).

    A live run came back with signposts as {"signposts": [...]} rather than a
    bare array, and the memo renderer -- which calls .slice() on it -- threw and
    produced an entirely blank page. The research prompt specifies the schema,
    but a schema in a prompt is a request; three renderers depending on it is a
    contract. So the common malformations are repaired here rather than trusted:

      {"signposts": [...]}  -> [...]      (self-wrapped)
      {"a": {...}, "b": {...}} -> [...]   (keyed object instead of a list)
      "single string"       -> ["..."]    (scalar where a list belongs)
    """
    if not isinstance(m, dict):
        return m, []
    out = json.loads(json.dumps(m))
    fixed = []

    for path in _MUST_BE_LIST:
        val = _dig(out, path)
        if val is None or isinstance(val, list):
            continue
        name = path[-1]
        new = None
        if isinstance(val, dict):
            inner = val.get(name)
            if isinstance(inner, list):
                new = inner                      # self-wrapped under its own key
            else:
                listy = [v for v in val.values() if isinstance(v, (dict, str))]
                if listy and len(listy) == len(val):
                    new = listy                  # keyed object standing in for a list
        elif isinstance(val, str) and val.strip():
            new = [val]                          # scalar where a list belongs
        if new is not None:
            _put(out, path, new)
            fixed.append(".".join(path))

    return out, fixed


def normalize_segments(segments):
    """Make segment shares internally consistent. Returns (segments, changed).

    Companies report overlapping segments -- UNH's Optum Rx revenue sits inside
    Optum -- and asking the model for mutually exclusive shares works only some
    of the time: two live runs came back at 119% and 118%. A pie cannot be drawn
    from shares that do not close, and a legend printing "75%" beside a wedge
    occupying 64% of the circle is simply wrong.

    So this is deterministic. When the shares miss 100 by more than a few points
    they are rescaled proportionally, and the `mix` LABEL is rewritten to match,
    because the label and the geometry disagreeing is the actual defect. Relative
    sizes -- the decision-useful part -- are preserved.
    """
    if not isinstance(segments, list) or not segments:
        return segments, False
    vals = [(s.get("mix_numeric") or 0) if isinstance(s, dict) else 0
            for s in segments]
    total = sum(v for v in vals if isinstance(v, (int, float)))
    if not total or abs(total - 100) <= 5:
        return segments, False

    out = []
    for seg, v in zip(segments, vals):
        if not isinstance(seg, dict):
            out.append(seg)
            continue
        seg = dict(seg)
        share = round(v / total * 100) if total else 0
        seg["mix_numeric"] = share
        # Rewrite the label only when it was a bare percentage; a label carrying
        # extra meaning is left alone rather than silently rewritten.
        label = str(seg.get("mix") or "").strip()
        if re.fullmatch(r"~?\d{1,3}(\.\d+)?%", label) or not label:
            seg["mix"] = f"{share}%"
        out.append(seg)
    return out, True


def enforce_master_budgets(m):
    """Bound the canonical object so the two-pager and memo cannot overflow.

    Returns (object, trimmed). Never raises. Same reasoning as the one-pager:
    these pages have fixed-height sections, so unbounded prose does not reflow,
    it clips -- and a clipped kill criterion is a research artifact that lies.
    """
    if not isinstance(m, dict):
        return m, []
    m, shape_fixed = coerce_master_shape(m)
    if shape_fixed:
        print(f"[deepdive] repaired malformed fields: {', '.join(shape_fixed)}")
    out = json.loads(json.dumps(m))   # deep copy; paths are nested
    trimmed = []

    for path, limit in _MASTER_WORD_BUDGETS:
        cur = _dig(out, path)
        new = _trim_words(cur, limit)
        if new != cur:
            _put(out, path, new)
            trimmed.append(".".join(path))

    for path, count, item_words in _MASTER_LIST_CAPS:
        items = _dig(out, path)
        if not isinstance(items, list):
            continue
        changed = False
        if len(items) > count:
            items = items[:count]
            changed = True
            trimmed.append(".".join(path) + "(count)")
        if item_words:
            new_items = [_trim_words(x, item_words) if isinstance(x, str) else x
                         for x in items]
            if new_items != items:
                items, changed = new_items, True
                trimmed.append(".".join(path))
        if changed:
            _put(out, path, items)

    segs = _dig(out, ("company_overview", "segments"))
    fixed_segs, seg_changed = normalize_segments(segs)
    if seg_changed:
        _put(out, ("company_overview", "segments"), fixed_segs)
        trimmed.append("company_overview.segments(rescaled to 100%)")

    for path, sub, limit in _MASTER_ITEM_WORDS:
        items = _dig(out, path)
        if not isinstance(items, list):
            continue
        new_items, changed = [], False
        for x in items:
            if isinstance(x, dict) and _words(x.get(sub)) > limit:
                x = dict(x)
                x[sub] = _trim_words(x.get(sub), limit)
                changed = True
            new_items.append(x)
        if changed:
            _put(out, path, new_items)
            trimmed.append(".".join(path) + "." + sub)

    return out, sorted(set(trimmed))


def master_violations(d):
    """Structural problems in the canonical object that would break a renderer.

    Deliberately thin. The master object feeds the two-pager and memo, which
    reflow across pages rather than living on a fixed canvas, so prose length
    matters far less here than the presence of the fields the renderers read.
    """
    d = d or {}
    out = []
    if not (d.get("company") or "").strip():
        out.append("company is empty")
    if not (d.get("ticker") or "").strip():
        out.append("ticker is empty")

    thesis = d.get("investment_thesis") or {}
    if not (thesis.get("summary") or "").strip():
        out.append("investment_thesis.summary is empty")

    for field in ("signposts", "thesis_threats", "opportunities"):
        if not isinstance(d.get(field), list) or not d.get(field):
            out.append(f"{field} is empty")

    # A pie chart cannot be drawn from shares that do not sum to ~100. The
    # prompt says to use 0 when unsupported; renderers fall back to a list.
    segs = ((d.get("company_overview") or {}).get("segments")) or []
    numeric = [s.get("mix_numeric") or 0 for s in segs if isinstance(s, dict)]
    total = sum(n for n in numeric if isinstance(n, (int, float)))
    if numeric and total and not (85 <= total <= 115):
        # UNH came back as 72/16/23/8 = 119, because the model mixed a parent
        # segment with its own sub-units. The pie then cannot both match the
        # geometry and show the stated numbers, so the renderer normalises and
        # says so -- but the underlying research is still wrong, and this is
        # where that gets surfaced rather than quietly drawn.
        out.append(f"segment mix_numeric sums to {total:.0f}, not ~100 — segments "
                   f"are probably overlapping (parent and child reported together)")

    return out


# ---------------------------------------------------------------------------
# market snapshot
# ---------------------------------------------------------------------------

def market_snapshot(ticker):
    """Best-effort live quote/fundamentals. Never raises.

    A market-data outage must not block a research run: the research itself is
    the expensive, valuable part, and the snapshot is a garnish that the
    renderers already treat as optional.
    """
    out = {"ticker": (ticker or "").upper().strip(), "ok": False}
    try:
        import yfinance as yf
        info = yf.Ticker(out["ticker"]).info or {}
        if not info:
            return out

        def pick(*keys):
            for k in keys:
                v = info.get(k)
                if v not in (None, "", 0):
                    return v
            return None

        price = pick("currentPrice", "regularMarketPrice", "previousClose")
        cap = pick("marketCap")
        out.update({
            "ok": True,
            "company": info.get("longName") or info.get("shortName") or "",
            "share_price": f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A",
            "market_cap": _human_cap(cap),
            "forward_pe": _round_or_na(pick("forwardPE")),
            "trailing_pe": _round_or_na(pick("trailingPE")),
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
            "employees": f"{info.get('fullTimeEmployees'):,}" if info.get("fullTimeEmployees") else "",
            "website": info.get("website") or "",
            "exchange": info.get("exchange") or "",
            "data_as_of": datetime.utcnow().strftime("%Y-%m-%d"),
        })
    except Exception as e:
        print(f"[deepdive] market_snapshot({ticker}) failed, continuing without it: {e}")
    return out


def _human_cap(v):
    if not isinstance(v, (int, float)) or v <= 0:
        return "N/A"
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= cutoff:
            return f"~${v / cutoff:.1f}{suffix}"
    return f"~${v:,.0f}"


def _round_or_na(v):
    return f"{v:.1f}x" if isinstance(v, (int, float)) and v > 0 else "N/A"


def merge_live_market(master, market):
    """Overlay verified live fields onto the researched object.

    Live market data beats a researched share price, which is stale the moment
    it is written. Everything else the model researched is left alone -- this
    is a targeted overlay, not a merge of two opinions.
    """
    if not master or not market or not market.get("ok"):
        return master
    glance = dict(master.get("at_glance") or {})
    for key in ("share_price", "market_cap", "sector", "industry",
                "employees", "website", "data_as_of"):
        val = market.get(key)
        if val:
            glance[key] = val
    if market.get("exchange") and not glance.get("exchange"):
        glance["exchange"] = market["exchange"]
    master["at_glance"] = glance
    master["_market"] = {k: v for k, v in market.items() if k != "ok"}
    return master


# ---------------------------------------------------------------------------
# research pipeline
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "{t} {c} investment thesis bull bear case analyst",
    "{t} {c} latest quarterly earnings results guidance",
    "{t} {c} segment revenue mix breakdown",
    "{t} {c} valuation forward P/E multiple versus history",
    "{t} {c} risks competition headwinds",
    "{t} {c} management long-term targets investor day",
]


def gather_web_context(ticker, company, search_fn, per_query=4):
    """Current web material plus the source trail, from Charlie's search layer.

    Returns (context_text, sources). Sources are kept whole and persisted: the
    printed artifacts deliberately do not show a Sources block, but the trail
    has to remain inspectable or the research is unfalsifiable.
    """
    seen, sources, blocks = set(), [], []
    for template in SEARCH_QUERIES:
        query = template.format(t=ticker, c=company or "")
        try:
            results = search_fn(query, max_results=per_query) or []
        except Exception as e:
            print(f"[deepdive] search failed for {query!r}: {e}")
            continue
        for r in results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({
                "title": (r.get("title") or "").strip()[:300],
                "url": url,
                "date": (r.get("published_date") or "")[:40],
            })
            body = (r.get("content") or r.get("raw_content") or "").strip()
            if body:
                blocks.append(f"--- {r.get('title') or url} ({url}) ---\n{body[:2500]}")

    return "\n\n".join(blocks)[:120000], sources


def research_with_web_search(ticker, company, market, api_key, extract_json,
                             model="claude-sonnet-4-5-20250929", on_step=None,
                             max_searches=8):
    """Canonical research using the model provider's own web search.

    The prototype ran OpenAI's Responses API with its web_search tool, so the
    model searched WHILE it researched rather than being handed a pre-baked
    context blob. This is the same shape on Anthropic, and it matters for two
    reasons: the model can follow a thread it did not know to ask for up front,
    and the citations come back attached to the claims rather than as a
    separate list of things that were merely read.

    Chosen over a third-party search API because it needs no additional vendor
    or key, and because in practice it reaches primary sources -- SEC EDGAR
    filings and earnings releases -- which is the source priority the research
    standards ask for.

    Returns (master, sources). Raises if no usable JSON comes back.
    """
    import anthropic

    step = on_step or (lambda _m: None)
    step(f"Researching {ticker} with live web search...")

    client = anthropic.Anthropic(api_key=api_key)
    parts = [f"Company to research: {company or ticker} (ticker {ticker})."]
    if market and market.get("ok"):
        parts.append(
            "VERIFIED LIVE MARKET SNAPSHOT (authoritative for price/market cap; "
            "do not contradict):\n" + json.dumps(
                {k: v for k, v in market.items() if k != "ok"}, indent=2))
    parts.append(
        "Search the web for current filings, earnings releases and reputable "
        "financial reporting, then produce the canonical research JSON described "
        "in your system prompt. Ground every figure in what you actually read; "
        "use \"N/A\" rather than inventing precision. Return the JSON as your "
        "final message.")

    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=deepdive_prompts.MASTER_RESEARCH_SYSTEM,
        tools=[{"type": "web_search_20250305", "name": "web_search",
                "max_uses": max_searches}],
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
        timeout=900.0,
    )

    text_parts, sources, seen = [], [], set()
    for block in resp.content or []:
        btype = getattr(block, "type", "")
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
            for cit in (getattr(block, "citations", None) or []):
                url = getattr(cit, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": (getattr(cit, "title", "") or "")[:300],
                                    "url": url, "date": ""})
        elif btype == "web_search_tool_result":
            for item in (getattr(block, "content", None) or []):
                url = getattr(item, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": (getattr(item, "title", "") or "")[:300],
                                    "url": url,
                                    "date": (getattr(item, "page_age", "") or "")[:40]})

    master = extract_json("\n".join(text_parts))
    if not isinstance(master, dict) or not master:
        raise RuntimeError("Web research returned no usable JSON object.")

    master.setdefault("ticker", (ticker or "").upper())
    if sources:
        master["sources"] = sources
    master, trimmed = enforce_master_budgets(master)
    if trimmed:
        print(f"[deepdive] master trimmed to memo budget: {', '.join(trimmed)}")
    step(f"Research complete — {len(sources)} sources.")
    return master, sources


def build_research_prompt(ticker, company, market, web_context):
    parts = [f"Company to research: {company or ticker} (ticker {ticker})."]
    if market and market.get("ok"):
        parts.append(
            "VERIFIED LIVE MARKET SNAPSHOT (authoritative for price/market cap; "
            "do not contradict):\n" + json.dumps(
                {k: v for k, v in market.items() if k not in ("ok",)}, indent=2))
    if web_context:
        parts.append("CURRENT WEB RESEARCH MATERIAL:\n" + web_context)
    parts.append(
        "Produce the canonical research JSON for this company using the schema and "
        "standards in your system prompt. Ground every figure in the material above "
        "or in your verified knowledge; use \"N/A\" rather than inventing precision.")
    return "\n\n".join(parts)


def build_editor_prompt(master):
    return (
        "CANONICAL RESEARCH OBJECT:\n"
        + json.dumps(master, indent=2)[:110000]
        + "\n\nCompress this into the dense one-pager JSON described in your system "
          "prompt. Use only facts present above. Respect every editorial budget."
    )


def build_repair_prompt(onepager, violations):
    return (
        "This one-pager JSON breaks its editorial budgets:\n"
        + "\n".join(f"- {v}" for v in violations)
        + "\n\nCURRENT JSON:\n" + json.dumps(onepager, indent=2)[:100000]
        + "\n\nReturn the corrected JSON only. Fix exactly the listed problems by "
          "tightening wording or adding/removing items to hit the required counts. "
          "Do not add new facts and do not weaken any other field to compensate."
    )


def research_company(ticker, company, market, call_llm, extract_json, search_fn,
                     api_keys=None, tier="deep", on_step=None):
    """Run the canonical research pass. Returns (master, sources).

    Raises on failure: a Deep Dive with no canonical object has nothing to
    render, so failing loudly beats handing three renderers an empty dict.
    """
    keys = api_keys or {}
    step = on_step or (lambda _m: None)

    step("Searching current web sources...")
    web_context, sources = gather_web_context(ticker, company, search_fn)

    step(f"Researching {ticker} across {len(sources)} sources...")
    result = call_llm(
        messages=[{"role": "user", "content": build_research_prompt(
            ticker, company, market, web_context)}],
        system=deepdive_prompts.MASTER_RESEARCH_SYSTEM,
        tier=tier, max_tokens=16000, timeout=600,
        anthropic_api_key=keys.get("anthropic", ""),
        gemini_api_key=keys.get("gemini", ""),
        openai_api_key=keys.get("openai", ""),
    )
    master = extract_json(result.get("text") or "")
    if not isinstance(master, dict) or not master:
        raise RuntimeError("Research returned no usable JSON object.")

    master.setdefault("ticker", ticker.upper())
    # Keep the researched sources only if the model bothered; ours are better
    # because they carry the URLs actually read.
    if sources:
        master["sources"] = sources

    # Bound the canonical object before anything renders from it. The two-pager
    # and memo lay it out in fixed-height sections, so unbounded prose clips.
    master, trimmed = enforce_master_budgets(master)
    if trimmed:
        print(f"[deepdive] master trimmed to memo budget: {', '.join(trimmed)}")
    return master, sources


def compress_onepager(master, call_llm, extract_json, api_keys=None,
                      tier="standard", on_step=None):
    """Editorial compression of the canonical object into the one-pager schema.

    Repairs once against measured violations. The prototype did the same but
    never re-checked; here the second result is measured too, and whichever of
    the two objects has fewer violations is the one returned -- a repair pass
    that made things worse should not win by being last.
    """
    keys = api_keys or {}
    step = on_step or (lambda _m: None)

    step("Compressing to one-pager...")
    result = call_llm(
        messages=[{"role": "user", "content": build_editor_prompt(master)}],
        system=deepdive_prompts.ONEPAGER_EDITOR_SYSTEM,
        tier=tier, max_tokens=12000, timeout=420,
        anthropic_api_key=keys.get("anthropic", ""),
        gemini_api_key=keys.get("gemini", ""),
        openai_api_key=keys.get("openai", ""),
    )
    onepager = extract_json(result.get("text") or "")
    if not isinstance(onepager, dict) or not onepager:
        raise RuntimeError("One-pager editor returned no usable JSON object.")

    violations = onepager_violations(onepager)
    if violations:
        step(f"Repairing {len(violations)} editorial overrun(s)...")
        try:
            repaired_raw = call_llm(
                messages=[{"role": "user", "content": build_repair_prompt(
                    onepager, violations)}],
                system=deepdive_prompts.ONEPAGER_EDITOR_SYSTEM,
                tier=tier, max_tokens=12000, timeout=420,
                anthropic_api_key=keys.get("anthropic", ""),
                gemini_api_key=keys.get("gemini", ""),
                openai_api_key=keys.get("openai", ""),
            )
            repaired = extract_json(repaired_raw.get("text") or "")
            if isinstance(repaired, dict) and repaired:
                if len(onepager_violations(repaired)) <= len(violations):
                    onepager, violations = repaired, onepager_violations(repaired)
        except Exception as e:
            print(f"[deepdive] one-pager repair pass failed, keeping original: {e}")

    onepager.setdefault("ticker", master.get("ticker", ""))
    onepager.setdefault("company", master.get("company", ""))

    # Last line of defence. Whatever the model did, what gets stored fits.
    #
    # The returned `violations` describe the object that is actually STORED, not
    # the draft the model handed over. Once a field has been trimmed it is
    # within budget, so continuing to report "headline: 40 words (max 10)" would
    # flag a problem that no longer exists. What the reader needs to know is
    # that the research came back too verbose and was cut -- that is a signal
    # about research quality, and it is reported as exactly that.
    onepager, trimmed = enforce_budgets(onepager)
    if trimmed:
        print(f"[deepdive] hard-trimmed to budget: {', '.join(trimmed)}")
        violations = [f"auto-trimmed to fit: {', '.join(trimmed)}"]
    return onepager, violations


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def ensure_schema(get_db):
    with get_db(commit=True) as (_c, cur):
        cur.execute('''
            CREATE TABLE IF NOT EXISTS deepdive_runs (
                id           SERIAL PRIMARY KEY,
                ticker       VARCHAR(20) NOT NULL,
                company      VARCHAR(255),
                master       JSONB,
                onepager     JSONB,
                sources      JSONB DEFAULT '[]'::jsonb,
                violations   JSONB DEFAULT '[]'::jsonb,
                meta         JSONB DEFAULT '{}'::jsonb,
                created_at   TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'UTC')
            )
        ''')
        # created_at is stored in UTC deliberately, because is_fresh() compares
        # it against datetime.utcnow(). A bare NOW() records the *server's local*
        # time, and on an Eastern box that made every run read as 240 minutes
        # old the instant it was written -- so a 30-minute cache never hit and
        # every request paid for a fresh multi-minute research pass.
        cur.execute("ALTER TABLE deepdive_runs ALTER COLUMN created_at "
                    "SET DEFAULT (NOW() AT TIME ZONE 'UTC')")
        cur.execute('CREATE INDEX IF NOT EXISTS idx_deepdive_ticker '
                    'ON deepdive_runs (ticker, created_at DESC)')


def save_run(get_db, ticker, company, master, onepager, sources, violations, meta,
             max_runs=20):
    """Persist one run and return its id. History is kept, capped per ticker."""
    ticker = (ticker or "").upper().strip()
    with get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO deepdive_runs
                (ticker, company, master, onepager, sources, violations, meta)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
            RETURNING id
        ''', (ticker, company or "", json.dumps(master or {}),
              json.dumps(onepager or {}), json.dumps(sources or []),
              json.dumps(violations or []), json.dumps(meta or {})))
        run_id = cur.fetchone()["id"]
        cur.execute('''
            DELETE FROM deepdive_runs
             WHERE ticker = %s AND id NOT IN (
                SELECT id FROM deepdive_runs WHERE ticker = %s
                 ORDER BY created_at DESC, id DESC LIMIT %s)
        ''', (ticker, ticker, max_runs))
    return run_id


def _as_obj(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value is not None else default


def _row_to_run(row):
    return {
        "id": row["id"],
        "ticker": row["ticker"],
        "company": row["company"],
        "master": _as_obj(row["master"], {}),
        "onepager": _as_obj(row["onepager"], {}),
        "sources": _as_obj(row["sources"], []),
        "violations": _as_obj(row["violations"], []),
        "meta": _as_obj(row["meta"], {}),
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
    }


def load_latest(get_db, ticker):
    with get_db() as (_c, cur):
        cur.execute('SELECT * FROM deepdive_runs WHERE ticker = %s '
                    'ORDER BY created_at DESC, id DESC LIMIT 1',
                    ((ticker or "").upper().strip(),))
        row = cur.fetchone()
    return _row_to_run(row) if row else None


def load_run(get_db, run_id):
    with get_db() as (_c, cur):
        cur.execute('SELECT * FROM deepdive_runs WHERE id = %s', (run_id,))
        row = cur.fetchone()
    return _row_to_run(row) if row else None


def list_runs(get_db, limit=100):
    """One entry per ticker, newest first — the tab's sidebar."""
    with get_db() as (_c, cur):
        cur.execute('''
            SELECT DISTINCT ON (ticker) id, ticker, company, created_at, violations
              FROM deepdive_runs
             ORDER BY ticker, created_at DESC, id DESC
        ''')
        rows = cur.fetchall() or []
    out = [{
        "id": r["id"], "ticker": r["ticker"], "company": r["company"],
        "createdAt": r["created_at"].isoformat() if r["created_at"] else None,
        "violationCount": len(_as_obj(r["violations"], [])),
    } for r in rows]
    out.sort(key=lambda x: x["createdAt"] or "", reverse=True)
    return out[:limit]


def is_fresh(run, ttl_minutes=CACHE_TTL_MINUTES):
    """Whether a stored run is inside the cache window."""
    if not run or not run.get("createdAt"):
        return False
    try:
        created = datetime.fromisoformat(run["createdAt"])
        if created.tzinfo is not None:
            # An aware value came from a timestamptz column; normalise to UTC
            # rather than discarding the offset, which is what silently shifted
            # freshness by the server's UTC offset.
            created = created.astimezone(timezone.utc).replace(tzinfo=None)
        age = (datetime.utcnow() - created).total_seconds()
        # A clock skew that puts creation in the future should not read as
        # "fresh forever"; treat anything negative as just-created.
        return 0 <= age < ttl_minutes * 60 or (age < 0 and ttl_minutes > 0)
    except Exception:
        return False


def load_golden_fixture(path):
    """The Deere calibration fixture — renderer work without a research call."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("master") or {}, data.get("onepager") or {}, \
        (data.get("master") or {}).get("sources") or []
