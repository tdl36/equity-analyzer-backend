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
_WORD_BUDGETS = [
    ("headline", 10),
    ("subheadline", 20),
    ("core_question", 30),
    ("overview_summary", 70),
    ("other_profit_pool", 24),
    ("valuation_callout", 24),
    ("final_takeaway", 65),
    ("bottom_line", 12),
    ("secondary_bottom_line", 7),
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

_LIST_ITEM_WORDS = [
    ("thesis_bullets", 20),
    ("financial_bullets", 17),
    ("bull_case", 10),
    ("bear_case", 10),
]


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
    if thesis_words > 115:
        out.append(f"thesis_summary: {thesis_words} words (max 115)")

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

    # Signpost cells are table cells on a fixed-width canvas; long ones wrap
    # the row taller and push the last signpost off the page.
    for i, sp in enumerate(d.get("signposts") or []):
        if not isinstance(sp, dict):
            continue
        for cell in ("signpost", "current", "target", "why"):
            n = _words(sp.get(cell))
            if n > 12:
                out.append(f"signposts[{i}].{cell}: {n} words (max 12)")

    for i, th in enumerate(d.get("threats") or []):
        if isinstance(th, dict) and _words(th.get("watch_for")) > 28:
            out.append(f"threats[{i}].watch_for: {_words(th.get('watch_for'))} words (max 28)")

    return out


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
        out.append(f"segment mix_numeric sums to {total:.0f} (want ~100, or all 0)")

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
