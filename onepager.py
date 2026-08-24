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
