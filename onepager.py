"""Investment one-pager assembly.

Turns a ticker into a single normalized JSON document that the frontend renders
as a hand-drawn one-page investment poster (see OnePager in src/app.jsx).

WHY JSON AND NOT AN IMAGE
------------------------------------------------------------------------------
Charlie already generates infographics as images via gemini-3-pro-image-preview
(THESIS_INFOGRAPHIC_STYLES in app_v3.py). That path is great for atmosphere and
useless for numbers: diffusion models re-draw table cells and chart axes as
plausible-looking glyphs, so a multiple or an EPS value can come out wrong and
differs on every run. A one-pager that a PM reads off has to be exact, so this
module stops at structured data and lets the client draw it deterministically.
The hand-drawn look lives in CSS/SVG, not in a sampler.

WHERE THE CONTENT COMES FROM
------------------------------------------------------------------------------
Nothing here invents a thesis from nothing when Charlie already has one:

    portfolio_analyses   -> thesis pillars, signposts, threats, conclusion
    stock_overviews      -> company overview, business model, mix, opps, risks
    thesis_scorecard_data-> pillar inclusion/exclusion overrides

Those are merged and handed to the LLM as source material; the LLM's job is to
normalize and compress into ONEPAGER_SCHEMA, not to re-research. Only the gaps
are researched. When a ticker has no thesis at all, the result is written back
as a DRAFT (is_draft=True) so it shows in the Thesis tab clearly marked rather
than silently becoming a curated position.
"""

import json
from datetime import datetime, timezone


# The contract. The renderer reads exactly these fields, so changing a key here
# means changing OnePager in src/app.jsx too. Kept as prompt text because that is
# what the LLM is held to; the renderer tolerates missing keys by hiding sections.
ONEPAGER_SCHEMA = """{
  "ticker": "string — uppercase symbol",
  "company": "string — full legal/common company name",
  "tagline": "string — ONE short line, max ~8 words, in the voice of the business. Not a slogan you invented for marketing; a compression of what the company does.",
  "at_a_glance": {
    "exchange": "e.g. 'DE (NYSE)'",
    "hq": "City, State/Country",
    "founded": "year",
    "employees": "approx headcount, e.g. '~79,000'",
    "fy_end": "fiscal year end, e.g. 'Oct 31'",
    "website": "domain only"
  },
  "investment_thesis": {
    "summary": "2-3 sentences. What kind of franchise is this and what is the setup right now.",
    "core_question": "The single question an investor is actually underwriting, phrased as a question ending in '?'. This is the spine of the page.",
    "points": ["4-6 short evidence bullets. Each must carry a number, date, or named driver — no generic quality claims."]
  },
  "company_overview": {
    "summary": "2-3 sentences on what the company designs/makes/sells.",
    "segments": [
      {
        "name": "segment name",
        "abbr": "short code e.g. 'PPA' (optional)",
        "share": 47.5,
        "share_label": "'~45-50%' — the hedged label to print",
        "description": "one line on what sits in this segment"
      }
    ],
    "segment_basis": "what the shares are a percentage OF, e.g. 'By Equipment Sales'",
    "footnote": "optional line for a segment that sits outside the percentages, e.g. financing arm"
  },
  "business_model": {
    "profit_pools": [{"name": "pool name", "description": "one short line"}],
    "caption": "one line on how the pools fit together"
  },
  "opportunities": [{"title": "short title", "description": "one line", "icon": "one of: leaf, people, cycle, globe, bank, chip, chart, truck, wrench, shield"}],
  "financial_snapshot": {
    "period": "e.g. 'FY2025E / Latest'",
    "metrics": [{"label": "Net Sales", "value": "~$61.0B", "note": "optional context e.g. 'vs ~$68.4B FY2023'"}],
    "mid_cycle_targets": [{"label": "Sales CAGR", "value": "~10%"}],
    "valuation": [{"label": "Forward P/E", "value": "~33-35x"}],
    "eps_chart": {
      "label": "e.g. 'Earnings Are Cyclical'",
      "y_label": "e.g. 'EPS ($)'",
      "points": [{"year": 2008, "eps": 28.5, "kind": "actual|estimate"}],
      "markers": [{"year": 2023, "label": "2023 peak $34.50"}]
    },
    "note": "optional one-line takeaway under the chart"
  },
  "signposts": [{"signpost": "what to watch", "current": "value today", "target": "target + by when", "why": "why it matters"}],
  "threats": [{"title": "short threat name", "watch_for": "the specific, falsifiable trigger to watch", "icon": "one of: cloud, chart_down, gauge, gear, scale, flag"}],
  "takeaway": {
    "summary": "2-3 sentences closing the argument.",
    "bull": ["3-5 short bullets"],
    "bear": ["3-5 short bullets"],
    "bottom_line": "one line, blunt"
  }
}"""


# Depth controls how much the assembler emits. It is orthogonal to visual style:
# style is a view-time choice and free to switch, depth changes the content and
# costs a generation. Brief is the one that actually earns the name "one-pager" —
# it is tuned to fit a single page rather than scroll.
ONEPAGER_DEPTHS = {
    "brief": {
        "label": "Brief",
        "note": "Dense true one-pager — fits a page",
        "directive": (
            "DEPTH: BRIEF. This must fit on ONE printed page, so be ruthless.\n"
            "- investment_thesis.points: at most 4, one line each, every one carrying a number.\n"
            "- Omit investment_thesis.summary and company_overview.summary entirely — the\n"
            "  core question and the bullets carry the argument.\n"
            "- opportunities: at most 4. signposts: at most 4. threats: at most 3.\n"
            "- business_model.profit_pools: at most 4, descriptions of 4 words or fewer.\n"
            "- takeaway.summary: one sentence. bull/bear: 3 bullets each, 4 words each.\n"
            "- Every field that survives must be shorter than you think is comfortable."
        ),
    },
    "standard": {
        "label": "Standard",
        "note": "Balanced — the default",
        "directive": (
            "DEPTH: STANDARD.\n"
            "- investment_thesis.points: 4-6. opportunities: 5. signposts: 5-6. threats: 4.\n"
            "- Section summaries are 2-3 sentences.\n"
            "- takeaway.bull / bear: 4-5 bullets each."
        ),
    },
    "deep": {
        "label": "Deep",
        "note": "Comprehensive — a document to read",
        "directive": (
            "DEPTH: DEEP. This is read, not glanced at, so give the reasoning room.\n"
            "- investment_thesis.points: 6-8, each 1-2 sentences, with the mechanism spelled\n"
            "  out rather than just the claim.\n"
            "- Section summaries are 3-4 sentences and may name second-order effects.\n"
            "- opportunities: 5-7 with a sentence of substantiation each.\n"
            "- signposts: every one the source supports, up to 8.\n"
            "- threats: 4-6, each with a specific, falsifiable trigger.\n"
            "- takeaway.summary: 3-4 sentences. bull/bear: 5 bullets each.\n"
            "- Still never invent a figure — depth means more reasoning, not more numbers."
        ),
    },
}
DEFAULT_DEPTH = "standard"


def depth_directive(depth):
    """Prompt fragment for a depth key; falls back to standard."""
    spec = ONEPAGER_DEPTHS.get(depth) or ONEPAGER_DEPTHS[DEFAULT_DEPTH]
    return spec["directive"]


SYSTEM_PROMPT = """You are a buy-side analyst compressing research into a single-page investment poster.

You will be given whatever Charlie already holds on this company — an existing
thesis, signposts, threats, and a company overview. Treat that as the primary
source of truth and preserve its judgements. Your job is compression and
normalization into a fixed JSON schema, NOT re-litigating the thesis.

Rules that matter more than style:

1. NEVER invent a number. Every figure you print must appear in the source
   material provided. If the source does not support a metric, omit that metric
   entirely rather than estimating it. An omitted row is fine; a wrong multiple
   is not — this page gets read off directly in meetings.
2. Hedge the way the source hedges. If the source says "~45-50%", print
   "~45-50%", not "47%".
3. Prefer specificity over completeness. Four bullets that each carry a number
   beat six that sound like an annual report.
4. `core_question` is the spine of the page. It must be the actual debate — the
   thing that decides whether the stock works — not a summary restated as a question.
5. Signposts must be falsifiable and paired: a current value and a target with a
   date. "Execution improves" is not a signpost. "Engaged acres 500M -> 600M by
   2030" is.
6. Threats must state the observable trigger, not the worry. "Ag downcycle" is a
   worry; "commodity prices stay low through '27, trough extends >36 months" is
   a trigger.
7. For eps_chart, only emit points you can support from the source. Mark future
   years kind="estimate". If there is no usable series, omit eps_chart.

Respond with ONLY the JSON object. No prose, no markdown fences."""


def _clean(value, limit=None):
    """Coerce a DB text/JSON column into a plain string for the prompt."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    if limit and len(text) > limit:
        text = text[:limit] + "\n…[truncated]"
    return text


def gather_source_material(ticker, get_db, parse_analysis_data):
    """Pull everything Charlie already knows about `ticker`.

    Returns (source_text, facts) where `facts` records which stores actually had
    data — the caller uses it to decide whether research is needed and whether
    the result should be persisted as a draft.
    """
    ticker = (ticker or "").upper().strip()
    facts = {
        "ticker": ticker,
        "company": "",
        "has_thesis": False,
        "has_overview": False,
        "sources": [],
    }
    parts = []

    with get_db() as (_, cur):
        cur.execute("SELECT * FROM portfolio_analyses WHERE ticker = %s", (ticker,))
        analysis_row = cur.fetchone()

        cur.execute("SELECT * FROM stock_overviews WHERE ticker = %s", (ticker,))
        overview_row = cur.fetchone()

        cur.execute(
            "SELECT scorecard_data FROM thesis_scorecard_data WHERE ticker = %s",
            (ticker,),
        )
        scorecard_row = cur.fetchone()

    if analysis_row:
        facts["has_thesis"] = True
        facts["sources"].append("thesis")
        row_dict = dict(analysis_row)
        facts["company"] = row_dict.get("company") or facts["company"]
        parsed = parse_analysis_data(row_dict)
        parts.append(
            "=== EXISTING INVESTMENT THESIS (authoritative — preserve these judgements) ===\n"
            f"Company: {parsed.get('company')}\n"
            f"Last updated: {parsed.get('date_str')}\n\n"
            f"THESIS:\n{_clean(parsed.get('thesis'), 14000)}\n\n"
            f"SIGNPOSTS:\n{_clean(parsed.get('signposts'), 8000)}\n\n"
            f"THREATS:\n{_clean(parsed.get('threats'), 8000)}\n\n"
            f"CONCLUSION:\n{_clean(parsed.get('conclusion'), 4000)}"
        )

    if overview_row:
        facts["has_overview"] = True
        facts["sources"].append("overview")
        ov = dict(overview_row)
        facts["company"] = facts["company"] or ov.get("company_name") or ""
        parts.append(
            "=== COMPANY OVERVIEW (authoritative for business description/mix) ===\n"
            f"Company: {_clean(ov.get('company_name'))}\n\n"
            f"OVERVIEW:\n{_clean(ov.get('company_overview'), 6000)}\n\n"
            f"BUSINESS MODEL:\n{_clean(ov.get('business_model'), 6000)}\n\n"
            f"BUSINESS MIX / SEGMENTS:\n{_clean(ov.get('business_mix'), 6000)}\n\n"
            f"OPPORTUNITIES:\n{_clean(ov.get('opportunities'), 6000)}\n\n"
            f"RISKS:\n{_clean(ov.get('risks'), 6000)}\n\n"
            f"CONCLUSION:\n{_clean(ov.get('conclusion'), 3000)}"
        )

    if scorecard_row and scorecard_row.get("scorecard_data"):
        sc = scorecard_row["scorecard_data"]
        if isinstance(sc, str):
            try:
                sc = json.loads(sc)
            except Exception:
                sc = None
        if sc:
            facts["sources"].append("scorecard")
            parts.append(
                "=== SCORECARD OVERRIDES (which pillars are included/excluded) ===\n"
                + _clean(sc, 4000)
            )

    return "\n\n".join(parts), facts


def build_onepager(
    ticker,
    *,
    get_db,
    parse_analysis_data,
    call_llm,
    extract_json,
    api_keys=None,
    tier="advanced",
    research_fn=None,
    force_research=False,
    depth=DEFAULT_DEPTH,
):
    """Assemble the one-pager JSON for `ticker`.

    `research_fn(ticker) -> str` is called when Charlie has no thesis and no
    overview for the ticker, or when `force_research` is set. Its output is
    appended to the source material clearly labelled as researched (not curated)
    so the assembler holds it to a higher bar before printing a number from it.

    A one-pager built purely from research is marked is_draft — the caller
    persists it that way so it never passes as a curated position.

    Returns (onepager_dict, facts).
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        raise ValueError("ticker is required")

    source_text, facts = gather_source_material(ticker, get_db, parse_analysis_data)
    has_curated = facts["has_thesis"] or facts["has_overview"]

    if not has_curated and research_fn is None:
        raise LookupError(
            f"No thesis or overview found for {ticker}, and no research function was supplied"
        )

    if not has_curated or force_research:
        researched = research_fn(ticker)
        facts["sources"].append("research")
        research_block = (
            "=== WEB RESEARCH (generated just now — NOT curated by the analyst) ===\n"
            "Treat with more caution than curated material. Do not print a number\n"
            "from here unless the text states it explicitly. Where this conflicts\n"
            "with curated material above, the curated material wins.\n\n"
            + _clean(researched, 30000)
        )
        source_text = f"{source_text}\n\n{research_block}" if source_text else research_block

    # Draft = nothing an analyst actually curated stands behind this page.
    facts["is_draft"] = not has_curated

    user_msg = (
        f"Build the investment one-pager JSON for {ticker}"
        + (f" ({facts['company']})" if facts["company"] else "")
        + ".\n\nSOURCE MATERIAL:\n\n"
        + source_text
        + "\n\n---\n\n"
        + depth_directive(depth)
        + "\n\n---\n\nEmit JSON matching exactly this schema:\n\n"
        + ONEPAGER_SCHEMA
    )

    keys = api_keys or {}
    result = call_llm(
        messages=[{"role": "user", "content": user_msg}],
        system=SYSTEM_PROMPT,
        tier=tier,
        max_tokens=8192,
        timeout=300,
        anthropic_api_key=keys.get("anthropic", ""),
        gemini_api_key=keys.get("gemini", ""),
        openai_api_key=keys.get("openai", ""),
    )

    data = extract_json(result["text"])
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object")

    # Trust the caller's ticker over the model's echo of it.
    data["ticker"] = ticker
    if facts["company"] and not data.get("company"):
        data["company"] = facts["company"]

    data["meta"] = {
        "depth": depth if depth in ONEPAGER_DEPTHS else DEFAULT_DEPTH,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": facts["sources"],
        "is_draft": facts.get("is_draft", False),
        "provider": result.get("provider"),
        "model": result.get("model"),
    }
    return data, facts


def save_onepager(ticker, data, get_db, depth=DEFAULT_DEPTH, max_versions=10):
    """Persist one (ticker, depth) page, pushing the previous version into history.

    Keyed on (ticker, depth) rather than ticker alone so a name can hold Brief and
    Deep at once and the UI can flip between them without paying for a new run.
    """
    ticker = (ticker or "").upper().strip()
    depth = depth if depth in ONEPAGER_DEPTHS else DEFAULT_DEPTH
    with get_db(commit=True) as (_, cur):
        cur.execute(
            "SELECT onepager, history FROM stock_onepagers WHERE ticker = %s AND depth = %s",
            (ticker, depth),
        )
        row = cur.fetchone()

        history = []
        if row:
            history = row.get("history") or []
            if isinstance(history, str):
                try:
                    history = json.loads(history)
                except Exception:
                    history = []
            previous = row.get("onepager")
            if previous:
                if isinstance(previous, str):
                    try:
                        previous = json.loads(previous)
                    except Exception:
                        previous = None
                if previous:
                    history.append(
                        {
                            "timestamp": (previous.get("meta") or {}).get("generated_at"),
                            "onepager": previous,
                        }
                    )
            history = history[-max_versions:]

        cur.execute(
            """
            INSERT INTO stock_onepagers (ticker, depth, company, onepager, history, updated_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (ticker, depth) DO UPDATE
              SET company = EXCLUDED.company,
                  onepager = EXCLUDED.onepager,
                  history = EXCLUDED.history,
                  updated_at = CURRENT_TIMESTAMP
            """,
            (
                ticker,
                depth,
                data.get("company", ""),
                json.dumps(data, ensure_ascii=False),
                json.dumps(history, ensure_ascii=False),
            ),
        )
    return True


def load_onepager(ticker, get_db, depth=None):
    """Return the stored page for (ticker, depth), or None.

    With no depth, returns whichever depth was written most recently — so opening
    a ticker shows what you last looked at rather than an arbitrary row.
    """
    ticker = (ticker or "").upper().strip()
    with get_db() as (_, cur):
        if depth:
            cur.execute(
                "SELECT onepager FROM stock_onepagers WHERE ticker = %s AND depth = %s",
                (ticker, depth),
            )
        else:
            cur.execute(
                "SELECT onepager FROM stock_onepagers WHERE ticker = %s "
                "ORDER BY updated_at DESC LIMIT 1",
                (ticker,),
            )
        row = cur.fetchone()
    if not row or not row.get("onepager"):
        return None
    data = row["onepager"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return None
    return data


def list_onepager_depths(ticker, get_db):
    """Which depths exist for a ticker, newest first."""
    ticker = (ticker or "").upper().strip()
    with get_db() as (_, cur):
        cur.execute(
            "SELECT depth, updated_at FROM stock_onepagers WHERE ticker = %s "
            "ORDER BY updated_at DESC",
            (ticker,),
        )
        rows = cur.fetchall()
    return [
        {"depth": r["depth"], "updatedAt": r["updated_at"].isoformat() if r["updated_at"] else None}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# AI poster prompt
# ---------------------------------------------------------------------------
# Renders the SAME verified JSON through an image model, for when the hand-drawn
# aesthetic matters more than exactness. The HTML styles remain the source of
# truth: a diffusion model redraws table cells and axis labels as plausible
# glyphs, so a figure here can be wrong and will differ between runs. The prompt
# fights that as hard as a prompt can — every value is quoted and the model is
# told to copy characters rather than compose them — but it cannot guarantee it,
# which is why the UI labels this style as unverified.

def _bullets(items, key=None, limit=6):
    out = []
    for it in (items or [])[:limit]:
        v = it.get(key) if (key and isinstance(it, dict)) else it
        if v:
            out.append(f'  - "{v}"')
    return "\n".join(out)


def build_poster_prompt(data):
    """Turn a one-pager JSON document into an image-generation prompt."""
    d = data or {}
    thesis = d.get("investment_thesis") or {}
    overview = d.get("company_overview") or {}
    model = d.get("business_model") or {}
    fin = d.get("financial_snapshot") or {}
    take = d.get("takeaway") or {}
    glance = d.get("at_a_glance") or {}

    segs = "\n".join(
        f'  - "{s.get("name","")}" {s.get("share_label","")} — "{s.get("description","")}"'
        for s in (overview.get("segments") or [])[:4]
    )
    pools = "\n".join(
        f'  - "{p.get("name","")}": "{p.get("description","")}"'
        for p in (model.get("profit_pools") or [])[:4]
    )
    opps = "\n".join(
        f'  - "{o.get("title","")}": "{o.get("description","")}"'
        for o in (d.get("opportunities") or [])[:5]
    )
    metrics = "\n".join(
        f'  - "{m.get("label","")}: {m.get("value","")}"'
        for m in (fin.get("metrics") or [])[:6]
    )
    signposts = "\n".join(
        f'  - "{s.get("signpost","")}" | now: "{s.get("current","")}" | target: "{s.get("target","")}"'
        for s in (d.get("signposts") or [])[:5]
    )
    threats = "\n".join(
        f'  - "{t.get("title","")}": "{t.get("watch_for","")}"'
        for t in (d.get("threats") or [])[:4]
    )

    return f"""Create a single-page hand-drawn investment research poster, portrait orientation.

VISUAL STYLE
A warm off-white notebook page. Everything drawn by hand in fine black pen with
restrained pastel pencil accents — sage green, sky blue, soft yellow, lavender,
muted red. Neat, confident handwriting throughout, not messy. Uneven hand-ruled
boxes with slightly imperfect corners, hand-drawn arrows, small circled section
numbers. A designer's analytical notebook page, not a corporate infographic and
not a whiteboard photo. No gradients, no drop shadows, no 3D, no stock icons,
no glossy finish. Leave comfortable margins; do not fill every pixel.

TEXT ACCURACY — THE MOST IMPORTANT INSTRUCTION
Every quoted string below must appear on the page reproduced EXACTLY, character
for character. Copy the characters; do not paraphrase, round, translate or
invent. Numbers, tickers, percentages, currency amounts and dates are the whole
point of this page — a wrong digit makes it worthless. If you cannot fit a
string legibly, leave that item out entirely rather than altering it. Do not add
any figure that does not appear below. All text in English. No watermarks.

LAYOUT — a masthead, then boxed sections in a two-column grid
Masthead: large hand-lettered title "{d.get('company','')} ({d.get('ticker','')})"
with the line "{d.get('tagline','')}" beneath it, underlined with a yellow
highlighter stroke. Top-right, a small boxed "AT A GLANCE" panel:
  - "Ticker: {glance.get('exchange','')}"
  - "HQ: {glance.get('hq','')}"
  - "Founded: {glance.get('founded','')}"
  - "Employees: {glance.get('employees','')}"
  - "FY End: {glance.get('fy_end','')}"

Box 1 — "INVESTMENT THESIS". Show this question prominently in its own outlined
box: "{thesis.get('core_question','')}"
Then these as checkbox bullets:
{_bullets(thesis.get('points'), limit=5)}

Box 2 — "COMPANY OVERVIEW". A hand-drawn pie chart of the segment mix, each
wedge labelled with its share, plus a legend:
{segs}

Box 3 — "BUSINESS MODEL". These profit pools as boxes joined by "+" signs:
{pools}
Caption underneath: "{model.get('caption','')}"

Box 4 — "KEY OPPORTUNITIES", each with a small hand-drawn doodle icon:
{opps}

Box 5 — "FINANCIAL SNAPSHOT". These as a bulleted list:
{metrics}
Include a small hand-drawn line chart labelled "{(fin.get('eps_chart') or {}).get('label','Earnings')}"
showing a cyclical earnings curve with peaks and troughs, the forward portion
drawn as a dashed line.

Box 6 — "KEY SIGNPOSTS". A hand-ruled table with columns Signpost / Current / Target:
{signposts}

Box 7 — "THESIS THREATS". A hand-ruled table of risk and what to watch for:
{threats}

Final box — "FINAL TAKEAWAY": "{take.get('summary','')}"
Two facing boxes, a green "BULL CASE" and a red "BEAR CASE":
  BULL:
{_bullets(take.get('bull'), limit=4)}
  BEAR:
{_bullets(take.get('bear'), limit=4)}

Footer strip across the bottom: "Bottom line: {take.get('bottom_line','')}"
"""


# ---------------------------------------------------------------------------
# thesis diff
# ---------------------------------------------------------------------------
# A refresh must never silently replace curated judgement. The candidate thesis
# produced from new documents is diffed against the live one and shown for
# approval, so the analyst sees exactly what a document changed before it lands.
# This doubles as document triage: the diff IS the answer to "did this filing
# actually move anything?"

def _norm(text):
    return " ".join(str(text or "").lower().split())


def _label(item):
    """Best available human label for a pillar / signpost / threat."""
    if not isinstance(item, dict):
        return str(item or "")
    for key in ("title", "signpost", "metric", "name", "pillar", "thesis"):
        if item.get(key):
            return str(item[key])
    # Fall back to the first non-empty string value so unlabelled rows still pair up.
    for v in item.values():
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _body(item):
    """Everything except the label, for detecting a reworded-but-same row."""
    if not isinstance(item, dict):
        return _norm(item)
    return _norm(json.dumps({k: v for k, v in sorted(item.items())
                             if k not in ("title", "signpost", "metric", "name")},
                            ensure_ascii=False, sort_keys=True))


def _diff_list(current, candidate):
    """Pair rows by label, then classify as added / removed / changed / same."""
    cur_items = [x for x in (current or []) if x]
    new_items = [x for x in (candidate or []) if x]

    cur_by = {}
    for it in cur_items:
        cur_by.setdefault(_norm(_label(it)), []).append(it)

    added, changed, same = [], [], []
    matched = set()

    for it in new_items:
        key = _norm(_label(it))
        pool = cur_by.get(key)
        if pool:
            before = pool.pop(0)
            matched.add(id(before))
            if _body(before) == _body(it):
                same.append(it)
            else:
                changed.append({"label": _label(it), "before": before, "after": it})
        else:
            added.append(it)

    removed = [it for it in cur_items if id(it) not in matched]

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": len(same),
    }


def diff_thesis(current_analysis, candidate_analysis):
    """Structured diff between the live thesis and a freshly generated candidate.

    Returns a dict the UI can render directly, plus a `has_changes` flag so a
    no-op refresh can say "nothing moved" instead of showing an empty approval
    dialog.
    """
    cur = current_analysis or {}
    new = candidate_analysis or {}

    cur_thesis = cur.get("thesis") or {}
    new_thesis = new.get("thesis") or {}

    out = {
        "pillars": _diff_list(cur_thesis.get("pillars"), new_thesis.get("pillars")),
        "signposts": _diff_list(cur.get("signposts"), new.get("signposts")),
        "threats": _diff_list(cur.get("threats"), new.get("threats")),
    }

    cur_concl = _norm(cur.get("conclusion"))
    new_concl = _norm(new.get("conclusion"))
    out["conclusion"] = {
        "changed": bool(new_concl) and cur_concl != new_concl,
        "before": cur.get("conclusion", ""),
        "after": new.get("conclusion", ""),
    }

    counts = {
        "added": sum(len(out[k]["added"]) for k in ("pillars", "signposts", "threats")),
        "removed": sum(len(out[k]["removed"]) for k in ("pillars", "signposts", "threats")),
        "changed": sum(len(out[k]["changed"]) for k in ("pillars", "signposts", "threats")),
    }
    out["counts"] = counts
    out["has_changes"] = bool(
        counts["added"] or counts["removed"] or counts["changed"] or out["conclusion"]["changed"]
    )
    return out


# ---------------------------------------------------------------------------
# change classification
# ---------------------------------------------------------------------------
# A thesis has layers with different half-lives. What the company does and why we
# own it should not move on a quarterly print; near-term setup, estimates and
# live controversies should be superseded by the newest quarter outright.
#
# But the boundary is not binary, and treating it as binary is worse than not
# classifying at all. Long-term claims move precisely BECAUSE short-term readings
# accumulate: three consecutive quarters of margin deterioration is not three
# cyclical blips, it is evidence the structural claim about pricing discipline is
# wrong. A rigid classifier files each quarter as "cyclical, expected" and
# suppresses the one signal worth surfacing.
#
# Hence three states, not two. STRUCTURAL_CHALLENGE exists specifically for a
# change that looks cyclical in isolation but reads as accumulating evidence
# against something structural — it is routed to the top of the review, because
# slow drift caught late is the expensive failure.

CLASSIFY_SYSTEM = """You classify proposed changes to an equity investment thesis by how
durable the claim being changed is. Your output decides what a busy analyst reads
first, so precision about WHICH changes deserve attention matters more than
tidiness.

Three classes:

"structural" — the slow-moving core: what the company does, how it makes money,
  segment economics, competitive position and moat, capital allocation posture,
  the multi-year opportunity, and the reason the position is held at all. These
  should rarely move on a single quarter. When one does move, that is the most
  important thing on the page.

"cyclical" — the fast-moving layer: the latest quarter's results, near-term
  setup versus consensus, current estimates, guidance for the next period, live
  controversies, and figures carrying a period label. These are EXPECTED to change
  every quarter and should be superseded by the newest reading rather than
  blended with the old one.

"structural_challenge" — a change that looks cyclical in isolation but reads as
  accumulating evidence against a structural claim. Use this when a short-term
  movement continues a direction already visible in the existing thesis, or when
  a quarterly datapoint contradicts a stated long-term assumption rather than
  merely updating it. Examples: a third consecutive quarter of the same margin
  deterioration; a "temporary" competitive loss recurring in a new market; churn
  that the thesis calls one-off appearing again.

  Do NOT reserve this for dramatic changes. Its purpose is catching slow drift
  early, when each individual quarter still looks unremarkable.

Judgement rules:
- Structural does not mean frozen. Moat erosion is structural and is exactly what
  must be caught. Classify by WHAT is changing, never by whether the change is
  welcome.
- When a change is genuinely ambiguous, prefer the higher-attention class
  (structural_challenge over cyclical, structural over structural_challenge). A
  false alarm costs seconds; a missed structural change costs a position.
- For "structural" and "structural_challenge" give a one-sentence `why` naming the
  specific evidence that forced it. For "cyclical", `why` may be brief.

Respond with ONLY valid JSON."""


def build_classify_prompt(diff, current_analysis=None):
    """Prompt for classifying the changes in a diff."""
    items = []
    for section in ("pillars", "signposts", "threats"):
        sec = (diff or {}).get(section) or {}
        for it in sec.get("added", []):
            items.append({"id": f"{section}:added:{_label(it)}", "section": section,
                          "kind": "added", "label": _label(it), "after": it})
        for it in sec.get("removed", []):
            items.append({"id": f"{section}:removed:{_label(it)}", "section": section,
                          "kind": "removed", "label": _label(it), "before": it})
        for ch in sec.get("changed", []):
            items.append({"id": f"{section}:changed:{ch.get('label')}", "section": section,
                          "kind": "changed", "label": ch.get("label"),
                          "before": ch.get("before"), "after": ch.get("after")})

    if (diff or {}).get("conclusion", {}).get("changed"):
        items.append({"id": "conclusion:changed", "section": "conclusion", "kind": "changed",
                      "label": "conclusion",
                      "before": diff["conclusion"].get("before"),
                      "after": diff["conclusion"].get("after")})

    context = ""
    if current_analysis:
        context = (
            "\n\nEXISTING THESIS, for judging whether a change continues a direction "
            "already present rather than starting a new one:\n"
            + _clean(current_analysis, 12000)
        )

    return items, (
        "Classify each proposed change below.\n\n"
        "CHANGES:\n" + _clean(items, 24000) + context +
        '\n\nReturn JSON: {"classifications": [{"id": "<id verbatim>", '
        '"layer": "structural|cyclical|structural_challenge", "why": "one sentence"}]}'
    )


def classify_diff(diff, call_llm, extract_json, api_keys=None, current_analysis=None,
                  tier="standard"):
    """Annotate a diff with a layer per change.

    Returns the diff with `layers` added: {id: {layer, why}} plus counts. On any
    failure the diff is returned unchanged — classification is a review aid, and
    losing it must never block an approval.
    """
    items, user_msg = build_classify_prompt(diff, current_analysis)
    if not items:
        return diff

    keys = api_keys or {}
    try:
        result = call_llm(
            messages=[{"role": "user", "content": user_msg}],
            system=CLASSIFY_SYSTEM, tier=tier, max_tokens=4096, timeout=180,
            anthropic_api_key=keys.get("anthropic", ""),
            gemini_api_key=keys.get("gemini", ""),
            openai_api_key=keys.get("openai", ""),
        )
        parsed = extract_json(result["text"]) or {}
        by_id = {c.get("id"): c for c in parsed.get("classifications", []) if c.get("id")}
    except Exception as e:
        print(f"[classify] failed, leaving diff unclassified: {e}")
        return diff

    valid = {"structural", "cyclical", "structural_challenge"}
    layers, counts = {}, {"structural": 0, "cyclical": 0, "structural_challenge": 0}
    for it in items:
        got = by_id.get(it["id"]) or {}
        layer = got.get("layer") if got.get("layer") in valid else "cyclical"
        layers[it["id"]] = {
            "layer": layer,
            "why": got.get("why", ""),
            "section": it["section"],
            "kind": it["kind"],
            "label": it["label"],
        }
        counts[layer] += 1

    out = dict(diff)
    out["layers"] = layers
    out["layerCounts"] = counts
    # What the analyst must actually read before approving.
    out["needsReview"] = counts["structural"] + counts["structural_challenge"]
    return out


# ---------------------------------------------------------------------------
# selective approval
# ---------------------------------------------------------------------------
# All-or-nothing approval is unusable at 20+ changes: one bad reword forces you
# to reject a run that got the other nineteen right, and re-running costs money
# without any guarantee the next attempt is better. So changes are applied
# individually, starting from the CURRENT thesis and adding only what was
# accepted — never from the candidate with rejected parts stripped out, which
# would silently carry through edits nobody looked at.
#
# Ids match the ones classify_diff assigns, so the UI can key selections,
# classifications and edits off a single identifier.

def _change_id(section, kind, label):
    return f"{section}:{kind}:{label}"


def apply_selected_changes(current_analysis, diff, accepted_ids, edits=None):
    """Apply only the accepted changes to the current thesis.

    accepted_ids: iterable of change ids to apply. Anything absent is left as it
                  is in the current thesis.
    edits:        {change_id: replacement_object_or_text} for changes the analyst
                  hand-edited before accepting. An edit implies acceptance.

    Returns (new_analysis, applied_count).
    """
    accepted = set(accepted_ids or [])

    # The editor hands back text. Parse it when it is JSON so a hand-edited
    # pillar lands as an object; keep it as a string when it is not, which is
    # correct for free-text fields like the conclusion. Never let a malformed
    # edit raise here — that would fail the whole approval.
    parsed_edits = {}
    for cid, val in (edits or {}).items():
        if isinstance(val, str):
            stripped = val.strip()
            if stripped.startswith(("{", "[")):
                try:
                    parsed_edits[cid] = json.loads(stripped)
                    continue
                except Exception:
                    pass
            parsed_edits[cid] = val
        else:
            parsed_edits[cid] = val
    edits = parsed_edits
    accepted |= set(edits.keys())

    out = json.loads(json.dumps(current_analysis or {}))   # deep copy
    applied = 0

    section_paths = {
        "pillars": ("thesis", "pillars"),
        "signposts": (None, "signposts"),
        "threats": (None, "threats"),
    }

    for section, (parent, key) in section_paths.items():
        sec = (diff or {}).get(section) or {}
        container = out.setdefault(parent, {}) if parent else out
        items = list(container.get(key) or [])

        # Removals first, so a remove+add on the same label cannot collide.
        for it in sec.get("removed", []):
            cid = _change_id(section, "removed", _label(it))
            if cid not in accepted:
                continue
            # Remove ONE match, not every row sharing the label. Two pillars can
            # legitimately carry the same title, and deleting both when the diff
            # proposed removing one silently loses content.
            target = _norm(_label(it))
            for i, x in enumerate(items):
                if _norm(_label(x)) == target:
                    items.pop(i)
                    applied += 1
                    break

        for ch in sec.get("changed", []):
            cid = _change_id(section, "changed", ch.get("label"))
            if cid not in accepted:
                continue
            replacement = edits.get(cid, ch.get("after"))
            target = _norm(ch.get("label") or "")
            hit = False
            for i, x in enumerate(items):
                if _norm(_label(x)) == target:
                    items[i] = replacement
                    hit = True
                    break
            if not hit:
                # The row it meant to change is gone; adding it back loses less
                # than dropping an accepted change silently.
                items.append(replacement)
            applied += 1

        for it in sec.get("added", []):
            cid = _change_id(section, "added", _label(it))
            if cid not in accepted:
                continue
            items.append(edits.get(cid, it))
            applied += 1

        container[key] = items

    concl = (diff or {}).get("conclusion") or {}
    if concl.get("changed"):
        cid = "conclusion:changed:conclusion"
        legacy = "conclusion:changed"          # id used before labels were added
        if cid in accepted or legacy in accepted:
            out["conclusion"] = edits.get(cid, edits.get(legacy, concl.get("after", "")))
            applied += 1

    return out, applied
