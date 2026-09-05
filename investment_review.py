"""Investment review: a structured investment state, and its renderings.

The existing note generator asks a model to write a research note and parses
prose back out of the answer. This module inverts that. The atomic unit here is
a ReviewState -- thesis, KPIs, expectations, scenarios, risks, actions -- and a
report is one rendering of it. The same state renders a one-page flash, a
six-page review, or a full initiation; nothing downstream re-derives meaning
from paragraphs.

Two rules shape everything below.

Deterministic work belongs in code. Multiples, expected returns, return
decomposition, probability weighting and reverse valuation are arithmetic. A
model asked to do arithmetic in prose gets it right most of the time, which is
the worst possible failure rate: wrong often enough to matter, right often
enough to trust. The model supplies inputs and interpretation; every number
below is computed here.

Nothing renders that the data does not support. Every block carries an
activation condition. A company that discloses one operating segment gets no
segment chart, a name with two comparable peers gets no peer table, and neither
leaves an empty frame behind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Fact provenance
# --------------------------------------------------------------------------

# What a statement is, kept separate from what it says. Management guidance
# quoted as though it were a reported result is the single easiest way for a
# research note to mislead, and prose does not preserve the distinction.
FACT_TYPES = (
    'reported_fact',      # in a filing or release
    'management_claim',   # said by the company, not independently verified
    'consensus',          # street estimate
    'calculated_fact',    # derived here, in code, from other facts
    'analyst_inference',  # our judgement
)


@dataclass
class Fact:
    statement: str
    type: str = 'analyst_inference'
    value: Optional[float] = None
    unit: str = ''
    period: str = ''
    source: str = ''
    source_date: str = ''
    basis: str = ''            # GAAP / non-GAAP / organic / reported / cc
    confidence: float = 0.8

    def is_verified(self) -> bool:
        return self.type in ('reported_fact', 'calculated_fact')


# --------------------------------------------------------------------------
# Investment state
# --------------------------------------------------------------------------

@dataclass
class KPI:
    """A metric the thesis actually depends on.

    bull_threshold and bear_threshold are what make this a scorecard rather
    than a dashboard: a number with no threshold cannot tell you whether the
    thesis is strengthening.
    """
    name: str
    current: Optional[float] = None
    prior: Optional[float] = None
    unit: str = '%'
    bull_threshold: Optional[float] = None
    bear_threshold: Optional[float] = None
    higher_is_better: bool = True
    importance: str = 'important'      # critical | important | secondary
    note: str = ''

    def trend(self) -> str:
        if self.current is None or self.prior is None:
            return '—'
        if abs(self.current - self.prior) < 1e-9:
            return '→'
        rising = self.current > self.prior
        return '↑' if rising == self.higher_is_better else '↓'

    def status(self) -> str:
        """Against its own thresholds, not against a general sense of good."""
        if self.current is None:
            return '—'
        hib = self.higher_is_better
        if self.bull_threshold is not None:
            if (self.current >= self.bull_threshold) if hib else (self.current <= self.bull_threshold):
                return 'on-thesis'
        if self.bear_threshold is not None:
            if (self.current <= self.bear_threshold) if hib else (self.current >= self.bear_threshold):
                return 'off-thesis'
        return 'watch'


@dataclass
class VariantView:
    """A debate is only useful when our view differs and something resolves it."""
    question: str
    consensus: str = ''
    our_view: str = ''
    why_different: str = ''
    supporting_evidence: str = ''
    disconfirming_evidence: str = ''
    resolves_when: str = ''
    resolution_date: str = ''

    def is_variant(self) -> bool:
        """No disagreement means no reason to print it."""
        return bool(self.our_view and self.consensus
                    and self.our_view.strip().lower() != self.consensus.strip().lower())


@dataclass
class Scenario:
    name: str                       # bear | base | bull
    probability: float              # 0..1
    metric_value: Optional[float] = None   # the EPS/FCF the multiple applies to
    multiple: Optional[float] = None
    target_price: Optional[float] = None
    assumptions: str = ''


@dataclass
class Risk:
    risk: str
    probability: float = 0.25
    severity: str = 'medium'        # low | medium | high
    horizon: str = ''
    evidence_today: str = ''
    trigger: str = ''
    action_if_triggered: str = ''   # without this a risk register only describes

    def is_actionable(self) -> bool:
        return bool(self.trigger and self.action_if_triggered)


@dataclass
class Catalyst:
    event: str
    window: str = ''
    probability: Optional[float] = None
    expectation: str = ''
    key_metric: str = ''
    thesis_impact: str = ''


@dataclass
class Change:
    """One movement since the last review, with its thesis consequence."""
    item: str
    prior: str = ''
    current: str = ''
    direction: str = 'neutral'      # positive | negative | neutral
    implication: str = ''


@dataclass
class ReviewState:
    ticker: str
    company: str = ''
    sector: str = ''
    as_of: str = ''
    mode: str = 'review'            # flash | review | initiation

    rating: str = ''                # own | hold | trim | avoid | none
    conviction: Optional[float] = None      # 0..10
    horizon: str = ''

    price: Optional[float] = None
    price_date: str = ''
    shares_out_m: Optional[float] = None
    net_debt_m: Optional[float] = None

    thesis: List[str] = field(default_factory=list)          # exactly 3
    changes: List[Change] = field(default_factory=list)
    kpis: List[KPI] = field(default_factory=list)
    variant_views: List[VariantView] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    catalysts: List[Catalyst] = field(default_factory=list)

    add_below: Optional[float] = None
    trim_above: Optional[float] = None
    key_question: str = ''
    upgrade_if: str = ''
    downgrade_if: str = ''

    priced_in: str = ''             # narrative of what the price implies
    implied_growth_pct: Optional[float] = None   # computed, not asserted

    business_quality: str = ''
    estimates: List[Dict[str, Any]] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    qc_findings: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# --------------------------------------------------------------------------
# Deterministic calculations
# --------------------------------------------------------------------------

def market_cap(state: ReviewState) -> Optional[float]:
    if state.price is None or state.shares_out_m is None:
        return None
    return state.price * state.shares_out_m


def enterprise_value(state: ReviewState) -> Optional[float]:
    mc = market_cap(state)
    if mc is None or state.net_debt_m is None:
        return None
    return mc + state.net_debt_m


def upside_pct(target: Optional[float], price: Optional[float]) -> Optional[float]:
    if target is None or not price:
        return None
    return (target - price) / price * 100.0


def normalise_probabilities(scenarios: List[Scenario]) -> List[Scenario]:
    """Probabilities that do not sum to 1 make the weighted return meaningless.

    Rescaled rather than rejected: a model that says 30/50/30 has expressed a
    usable view of the relative odds and a useless view of the absolute ones.
    """
    total = sum(s.probability or 0 for s in scenarios)
    if total <= 0:
        return scenarios
    for s in scenarios:
        s.probability = (s.probability or 0) / total
    return scenarios


def scenario_target(s: Scenario) -> Optional[float]:
    """Target from multiple x metric where both are given, else as stated."""
    if s.metric_value is not None and s.multiple is not None:
        return round(s.metric_value * s.multiple, 2)
    return s.target_price


def expected_return(state: ReviewState) -> Optional[Dict[str, float]]:
    """Probability-weighted return, as a consistency check on the rating.

    Not a forecast. Its job is to catch a note that recommends adding while its
    own scenarios imply a negative expected return -- an inconsistency that is
    invisible when the scenarios sit in a table and the recommendation sits
    three pages away.
    """
    if not state.scenarios or not state.price:
        return None
    scen = normalise_probabilities(list(state.scenarios))
    weighted = 0.0
    for s in scen:
        tgt = scenario_target(s)
        if tgt is None:
            return None
        weighted += (s.probability or 0) * tgt
    return {
        'weighted_target': round(weighted, 2),
        'expected_return_pct': round((weighted - state.price) / state.price * 100.0, 1),
    }


def return_decomposition(price: Optional[float], target: Optional[float],
                         start_metric: Optional[float], end_metric: Optional[float],
                         start_multiple: Optional[float], end_multiple: Optional[float],
                         dividend_yield_pct: float = 0.0) -> Optional[Dict[str, float]]:
    """Split an expected return into growth, re-rating and capital returns.

    A 10% expected return built entirely on multiple expansion is a different
    proposition from the same 10% built on earnings, and the two are
    indistinguishable in a price target.
    """
    if not all(v not in (None, 0) for v in (price, start_metric, end_metric,
                                            start_multiple, end_multiple)):
        return None
    growth = (end_metric / start_metric) - 1.0
    rerate = (end_multiple / start_multiple) - 1.0
    total = (1 + growth) * (1 + rerate) - 1.0
    return {
        'fundamental_growth_pct': round(growth * 100, 1),
        'multiple_change_pct': round(rerate * 100, 1),
        'capital_returns_pct': round(dividend_yield_pct, 2),
        'total_pct': round(total * 100 + dividend_yield_pct, 1),
    }


def implied_growth(price: Optional[float], metric_per_share: Optional[float],
                   terminal_multiple: float, years: int,
                   discount_rate: float = 0.09) -> Optional[float]:
    """What growth today's price requires, given an exit multiple.

    The reverse of the usual question. "Is 15x cheap?" depends on a view you
    have not stated; "the price requires 11% annual growth for four years"
    is a claim that can be argued with.
    """
    if not price or not metric_per_share or metric_per_share <= 0 or years <= 0:
        return None
    required = price * ((1 + discount_rate) ** years) / terminal_multiple
    ratio = required / metric_per_share
    if ratio <= 0:
        return None
    return round((ratio ** (1.0 / years) - 1.0) * 100, 1)


def cagr(start: Optional[float], end: Optional[float], years: float) -> Optional[float]:
    if not start or not end or start <= 0 or years <= 0:
        return None
    return round(((end / start) ** (1.0 / years) - 1.0) * 100, 1)


# --------------------------------------------------------------------------
# Consistency checks
# --------------------------------------------------------------------------

def consistency_findings(state: ReviewState) -> List[str]:
    """Arithmetic and internal-logic checks, run before anything renders.

    These are the failures a reader notices immediately and a generator never
    does: a rating that contradicts its own expected return, scenario
    probabilities that do not sum, an add-level above the current price.
    """
    out: List[str] = []

    probs = [s.probability or 0 for s in state.scenarios]
    if probs and abs(sum(probs) - 1.0) > 0.02:
        out.append(f'scenario probabilities sum to {sum(probs):.2f}, not 1.00')

    er = expected_return(state)
    if er and state.rating:
        r = state.rating.lower()
        if r in ('own', 'buy', 'add') and er['expected_return_pct'] < 0:
            out.append(f"rating '{state.rating}' against an expected return of "
                       f"{er['expected_return_pct']}%")
        if r in ('avoid', 'sell', 'trim') and er['expected_return_pct'] > 15:
            out.append(f"rating '{state.rating}' against an expected return of "
                       f"+{er['expected_return_pct']}%")

    if state.price:
        if state.add_below and state.add_below > state.price:
            out.append(f'add-below level ({state.add_below}) is above the current '
                       f'price ({state.price})')
        if state.trim_above and state.trim_above < state.price:
            out.append(f'trim-above level ({state.trim_above}) is below the current '
                       f'price ({state.price})')

    targets = {s.name.lower(): scenario_target(s) for s in state.scenarios}
    if all(k in targets and targets[k] is not None for k in ('bear', 'base', 'bull')):
        if not (targets['bear'] <= targets['base'] <= targets['bull']):
            out.append('scenario targets are not ordered bear <= base <= bull')

    if state.thesis and len(state.thesis) != 3:
        out.append(f'thesis has {len(state.thesis)} points; the format asks for 3')

    unactionable = [r.risk for r in state.risks if not r.is_actionable()]
    if unactionable:
        out.append('risks with no trigger or no action: ' + '; '.join(unactionable[:3]))

    without_threshold = [k.name for k in state.kpis
                         if k.bull_threshold is None and k.bear_threshold is None]
    if without_threshold:
        out.append('KPIs with no threshold, so no status can be computed: '
                   + '; '.join(without_threshold[:3]))

    non_variant = [v.question for v in state.variant_views if not v.is_variant()]
    if non_variant:
        out.append('debates where our view matches consensus: '
                   + '; '.join(non_variant[:2]))

    if state.rating and state.rating.lower() != 'none' and not state.horizon:
        out.append('a target with no time horizon')

    return out


# --------------------------------------------------------------------------
# Sector modules
# --------------------------------------------------------------------------

# The KPIs that decide a thesis are not the same across sectors, and a template
# that asks every company for the same eight metrics produces blanks for most
# of them. Suggestions only -- the model picks from these and may add its own.
SECTOR_KPIS: Dict[str, List[str]] = {
    'software': ['Organic subscription growth (cc)', 'cRPO growth', 'Net revenue retention',
                 'Bookings / net new AOV', 'Seat growth', 'Gross margin',
                 'SBC as % of revenue', 'FCF margin'],
    'medtech': ['Procedure volume growth', 'Price/mix', 'Installed base growth',
                'Utilisation per system', 'Market share', 'Gross margin',
                'Pipeline / clinical readouts'],
    'pharma': ['Key drug revenue growth', 'TRx trend', 'Market share',
               'LOE exposure (% of revenue)', 'Pipeline readouts',
               'Gross margin', 'R&D as % of revenue'],
    'managed care': ['Medical loss ratio', 'Membership growth', 'Segment margin',
                     'Prior-year development', 'Star ratings', 'Rate adequacy'],
    'industrials': ['Organic volume growth', 'Price/mix', 'Backlog',
                    'Book-to-bill', 'Incremental margin', 'Capacity utilisation',
                    'Working capital / sales'],
    'banks': ['Net interest income', 'Net interest margin', 'Deposit beta',
              'Loan growth', 'Net charge-offs', 'CET1 ratio', 'ROTCE'],
    'reits': ['Same-store NOI growth', 'Occupancy', 'Leasing spreads',
              'AFFO per share', 'NAV per share', 'Cap rate', 'Net debt / EBITDA'],
    'energy': ['Production growth', 'Realised price', 'Cash cost per unit',
               'Reserve replacement', 'Capex / cash flow', 'Breakeven price'],
    'consumer': ['Same-store sales', 'Volume vs price', 'Gross margin',
                 'Market share', 'Inventory turns', 'Customer acquisition cost'],
}

_SECTOR_ALIASES = {
    'information technology': 'software', 'technology': 'software',
    'health care': 'medtech', 'healthcare': 'medtech',
    'biotechnology': 'pharma', 'pharmaceuticals': 'pharma',
    'health insurance': 'managed care', 'insurance': 'managed care',
    'financials': 'banks', 'financial services': 'banks',
    'real estate': 'reits', 'industrial': 'industrials',
    'consumer staples': 'consumer', 'consumer discretionary': 'consumer',
}


def sector_kpis(sector: str) -> List[str]:
    """Suggested KPIs for a sector, empty when we have no opinion.

    Returning nothing is the right answer for an unrecognised sector: a
    generic list would be worse than letting the model choose from the
    filings in front of it.
    """
    key = (sector or '').strip().lower()
    key = _SECTOR_ALIASES.get(key, key)
    return SECTOR_KPIS.get(key, [])


# --------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------

def active_blocks(state: ReviewState, mode: str = 'review') -> Dict[str, bool]:
    """Which blocks have enough behind them to be worth rendering.

    An empty section is worse than a missing one: it reads as a template that
    ran, rather than as an analyst who looked and found nothing.
    """
    mode = (mode or 'review').lower()
    variant = [v for v in state.variant_views if v.is_variant()]
    return {
        'dashboard': True,
        'changes': bool(state.changes),
        'scorecard': len(state.kpis) >= 2,
        'variant_views': bool(variant) and mode != 'flash',
        'priced_in': bool(state.priced_in or state.implied_growth_pct is not None),
        'estimates': len(state.estimates) >= 1 and mode != 'flash',
        'business_quality': bool(state.business_quality) and mode == 'initiation',
        'valuation': len(state.scenarios) >= 2,
        'catalysts': bool(state.catalysts) and mode != 'flash',
        'risks': bool(state.risks),
        'falsification': bool(state.upgrade_if or state.downgrade_if),
    }


MODES = {
    'flash':      {'label': 'Flash Update', 'pages': '1-2',
                   'note': 'One event, what it changes, what to do'},
    'review':     {'label': 'Investment Review', 'pages': '5-7',
                   'note': 'Standard portfolio review'},
    'initiation': {'label': 'Deep Dive / Initiation', 'pages': '10-15',
                   'note': 'Establish the full thesis'},
}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You are an experienced buy-side analyst producing structured investment "
    "state, not prose. You return JSON only. You never compute derived figures "
    "-- multiples, upside, expected return, CAGR and implied growth are "
    "calculated downstream in code -- you supply the inputs they need and the "
    "judgement code cannot supply."
)


def extract_prompt(ticker: str, company: str, sector: str, mode: str,
                   price_block: str, prior_state_json: str = '') -> str:
    """Ask for investment state. Everything derived is computed after."""
    kpi_hint = sector_kpis(sector)
    kpi_line = ('Typical KPIs for this sector (use what the sources support, add '
                'your own, drop what does not apply): ' + ', '.join(kpi_hint)
                if kpi_hint else
                'Choose the KPIs the thesis actually depends on, from the sources.')
    prior = (f"\n\nPREVIOUS REVIEW STATE (compare against this and populate "
             f"`changes`; do not restate what has not moved):\n{prior_state_json}\n"
             if prior_state_json else
             "\n\nNo previous review exists. Leave `changes` empty rather than "
             "inventing movement.\n")

    return f"""Produce the investment state for {company} ({ticker}).

{price_block}

SECTOR: {sector or 'unclassified'}
MODE: {mode} ({MODES.get(mode, MODES['review'])['note']})
{kpi_line}{prior}

Answer the questions a portfolio manager needs answered to act today:
what is the thesis, what changed, what does the market expect, where do we
differ, which KPIs decide it, what is the return and the downside, what would
prove us wrong, and what should be done.

RULES
- Return ONLY the JSON object below. No prose outside it.
- Do NOT compute multiples, upside percentages, expected returns, CAGRs or
  implied growth. Supply the inputs; the arithmetic happens in code. Any
  number you do state must be one you read in a source, not one you derived.
- `thesis` is exactly 3 sentences: why the business is interesting, why the
  stock is mispriced, and what has to happen for it to work.
- Every KPI needs a bull_threshold and a bear_threshold, or it cannot report a
  status. If you cannot name thresholds, the metric is not a thesis KPI.
- Every risk needs a measurable `trigger` and an `action_if_triggered`. A risk
  with neither is a description, not a risk.
- A debate belongs in `variant_views` only where our view genuinely differs
  from consensus. If we agree with the street, leave it out.
- `priced_in` states what operating outcome the current price requires -- not
  whether the multiple looks cheap against history.
- Facts: classify each material statement. Management guidance is
  management_claim, not reported_fact, however confident the company sounded.
- Prefer omitting a field to filling it with something unsupported. Empty
  sections are dropped at render time; invented ones are not detectable.

{{
  "company": "", "sector": "",
  "rating": "own|hold|trim|avoid|none",
  "conviction": 0.0,
  "horizon": "e.g. 12 months",
  "shares_out_m": 0, "net_debt_m": 0,
  "thesis": ["", "", ""],
  "changes": [{{"item": "", "prior": "", "current": "",
                "direction": "positive|negative|neutral", "implication": ""}}],
  "kpis": [{{"name": "", "current": 0, "prior": 0, "unit": "%",
             "bull_threshold": 0, "bear_threshold": 0,
             "higher_is_better": true,
             "importance": "critical|important|secondary", "note": ""}}],
  "variant_views": [{{"question": "", "consensus": "", "our_view": "",
                      "why_different": "", "supporting_evidence": "",
                      "disconfirming_evidence": "", "resolves_when": "",
                      "resolution_date": ""}}],
  "scenarios": [{{"name": "bear|base|bull", "probability": 0.25,
                  "metric_value": 0, "multiple": 0, "target_price": 0,
                  "assumptions": ""}}],
  "risks": [{{"risk": "", "probability": 0.25, "severity": "low|medium|high",
              "horizon": "", "evidence_today": "", "trigger": "",
              "action_if_triggered": ""}}],
  "catalysts": [{{"event": "", "window": "", "probability": 0.5,
                  "expectation": "", "key_metric": "", "thesis_impact": ""}}],
  "add_below": 0, "trim_above": 0,
  "key_question": "",
  "upgrade_if": "", "downgrade_if": "",
  "priced_in": "",
  "business_quality": "",
  "estimates": [{{"metric": "", "period": "", "street": 0, "ours": 0,
                  "unit": "", "why_different": ""}}],
  "facts": [{{"statement": "", "type": "reported_fact|management_claim|consensus|calculated_fact|analyst_inference",
              "period": "", "source": "", "source_date": "", "basis": "",
              "confidence": 0.9}}]
}}"""


QC_SYSTEM = (
    "You are a skeptical portfolio manager reviewing an analyst's draft. You "
    "did not write it and you have no stake in its conclusion. You return JSON "
    "only. You are looking for what is wrong, missing, or unsupported -- not "
    "for things to praise."
)


def qc_prompt(state_json: str, computed_json: str) -> str:
    """The adversarial pass. Deliberately a separate call and context.

    A model asked to critique its own draft rewrites the critique to fit the
    draft. This one is given the state and the computed figures and never sees
    the drafting conversation.
    """
    return f"""Review this investment state as a skeptical PM. Be specific and
be hard on it. Nothing is off limits.

INVESTMENT STATE:
{state_json}

FIGURES COMPUTED IN CODE (these are arithmetic; do not dispute them, but do
check whether the state's conclusions are consistent with them):
{computed_json}

Ask, and answer only where there is something to say:
- Which claims carry no evidence?
- Where is management's framing being repeated as though it were fact?
- What is the strongest argument against the thesis, and is it made here?
- Which KPI that decides this thesis is missing?
- Are our estimates actually different from consensus, or does the note only
  assert that they are?
- Does the valuation count the same favourable outcome twice?
- Does the bull case require assumptions that contradict each other, or the
  bear case?
- Is the bear case genuinely bearish, or a softer version of the base case?
- Is the recommendation consistent with the expected return computed above?
- What would a PM ask after reading this that it does not answer?

Return:
{{"findings": [{{"issue": "", "severity": "high|medium|low", "where": "",
                 "fix": ""}}],
  "verdict": "ship|revise",
  "strongest_counterargument": ""}}"""


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt(v: Optional[float], unit: str = '', dp: int = 1) -> str:
    if v is None:
        return '—'
    if unit == '$':
        return f'${v:,.2f}'
    s = f'{v:,.{dp}f}'.rstrip('0').rstrip('.') if dp else f'{v:,.0f}'
    return f'{s}{unit}'


_STATUS_MARK = {'on-thesis': 'ON', 'off-thesis': 'OFF', 'watch': 'WATCH', '—': '—'}


def render_markdown(state: ReviewState, mode: str = 'review') -> str:
    """The state as a memo. One rendering of many, not the source of truth.

    Ordered by decision value: the conclusion is on page one because a reader
    deciding whether to act should not have to reach page sixteen to find out
    what the analyst thinks.
    """
    active = active_blocks(state, mode)
    er = expected_return(state)
    mc, ev = market_cap(state), enterprise_value(state)
    out: List[str] = []
    w = out.append

    # ---- page 1: decision ----
    w(f'# {state.company or state.ticker} ({state.ticker})')
    w('')
    head = []
    if state.rating and state.rating.lower() != 'none':
        head.append(f'**{state.rating.upper()}**')
    if state.conviction is not None:
        head.append(f'Conviction {state.conviction:.1f}/10')
    if state.horizon:
        head.append(f'Horizon {state.horizon}')
    if head:
        w(' | '.join(head))
        w('')
    if state.price:
        line = f'Price {_fmt(state.price, "$", 2)}'
        if state.price_date:
            line += f' (close {state.price_date})'
        if mc:
            line += f' · Market cap ${mc/1000:,.1f}bn'
        if ev:
            line += f' · EV ${ev/1000:,.1f}bn'
        w(line)
        w('')

    if state.thesis:
        w('## Thesis')
        w('')
        for t in state.thesis[:3]:
            w(f'- {t}')
        w('')

    if active['valuation']:
        w('## Scenarios and expected return')
        w('')
        w('| Scenario | Prob | Target | vs price | Assumptions |')
        w('|---|---|---|---|---|')
        for s in normalise_probabilities(list(state.scenarios)):
            tgt = scenario_target(s)
            up = upside_pct(tgt, state.price)
            w(f'| {s.name.title()} | {s.probability*100:.0f}% | {_fmt(tgt, "$", 2)} '
              f'| {_fmt(up, "%")} | {s.assumptions} |')
        if er:
            w(f'| **Weighted** | 100% | **{_fmt(er["weighted_target"], "$", 2)}** '
              f'| **{_fmt(er["expected_return_pct"], "%")}** | probability-weighted |')
        w('')

    if active['changes']:
        w('## What changed since the last review')
        w('')
        w('| Change | Prior | Current | | Thesis impact |')
        w('|---|---|---|---|---|')
        arrow = {'positive': '+', 'negative': '−', 'neutral': '='}
        for c in state.changes[:5]:
            w(f'| {c.item} | {c.prior} | {c.current} | '
              f'{arrow.get(c.direction, "=")} | {c.implication} |')
        w('')

    if active['scorecard']:
        w('## Thesis scorecard')
        w('')
        w('| KPI | Prior | Current | Bull | Bear | Trend | Status |')
        w('|---|---|---|---|---|---|---|')
        rank = {'critical': 0, 'important': 1, 'secondary': 2}
        for k in sorted(state.kpis, key=lambda x: rank.get(x.importance, 3)):
            w(f'| {k.name} | {_fmt(k.prior, k.unit)} | {_fmt(k.current, k.unit)} '
              f'| {_fmt(k.bull_threshold, k.unit)} | {_fmt(k.bear_threshold, k.unit)} '
              f'| {k.trend()} | {_STATUS_MARK.get(k.status(), k.status())} |')
        w('')

    pos = []
    if state.add_below:
        pos.append(f'Add below {_fmt(state.add_below, "$", 2)}')
    if state.trim_above:
        pos.append(f'Trim above {_fmt(state.trim_above, "$", 2)}')
    if pos:
        w('## Positioning')
        w('')
        w(' · '.join(pos))
        w('')
    if state.key_question:
        w(f'**The question that decides this:** {state.key_question}')
        w('')

    if mode == 'flash':
        return '\n'.join(out).strip() + '\n'

    # ---- page 2: expectations ----
    variant = [v for v in state.variant_views if v.is_variant()]
    if active['variant_views']:
        w('---')
        w('')
        w('## Expectations and where we differ')
        w('')
        for i, v in enumerate(variant, 1):
            w(f'### {i}. {v.question}')
            w('')
            w(f'- **Consensus:** {v.consensus}')
            w(f'- **Our view:** {v.our_view}')
            if v.why_different:
                w(f'- **Why we differ:** {v.why_different}')
            if v.supporting_evidence:
                w(f'- **Supporting:** {v.supporting_evidence}')
            if v.disconfirming_evidence:
                w(f'- **Against us:** {v.disconfirming_evidence}')
            if v.resolves_when:
                res = v.resolves_when + (f' ({v.resolution_date})' if v.resolution_date else '')
                w(f'- **Resolves:** {res}')
            w('')

    if active['priced_in']:
        w('## What is priced in')
        w('')
        if state.implied_growth_pct is not None:
            w(f'At {_fmt(state.price, "$", 2)}, the price requires roughly '
              f'**{state.implied_growth_pct}% annual growth** in the valuation '
              f'metric to clear a normal return over the stated horizon.')
            w('')
        if state.priced_in:
            w(state.priced_in)
            w('')

    # ---- page 3: estimates ----
    if active['estimates']:
        w('---')
        w('')
        w('## Where our numbers differ from the street')
        w('')
        w('| Metric | Period | Street | Ours | Δ | Why |')
        w('|---|---|---|---|---|---|')
        for e in state.estimates:
            street, ours = e.get('street'), e.get('ours')
            delta = '—'
            if isinstance(street, (int, float)) and isinstance(ours, (int, float)) and street:
                delta = f'{(ours - street) / abs(street) * 100:+.1f}%'
            w(f'| {e.get("metric","")} | {e.get("period","")} | '
              f'{_fmt(street, e.get("unit",""))} | {_fmt(ours, e.get("unit",""))} '
              f'| {delta} | {e.get("why_different","")} |')
        w('')

    # ---- page 4: quality (initiation only) ----
    if active['business_quality']:
        w('---')
        w('')
        w('## Business quality')
        w('')
        w(state.business_quality)
        w('')

    # ---- page 6: catalysts, risks, falsification ----
    if active['catalysts']:
        w('---')
        w('')
        w('## Catalysts')
        w('')
        w('| Window | Event | Watch | Thesis impact |')
        w('|---|---|---|---|')
        for c in state.catalysts:
            w(f'| {c.window} | {c.event} | {c.key_metric} | {c.thesis_impact} |')
        w('')

    if active['risks']:
        w('## Risks and what would trigger action')
        w('')
        w('| Risk | Prob | Severity | Evidence today | Trigger | Action if triggered |')
        w('|---|---|---|---|---|---|')
        for r in sorted(state.risks, key=lambda x: -(x.probability or 0)):
            w(f'| {r.risk} | {(r.probability or 0)*100:.0f}% | {r.severity} '
              f'| {r.evidence_today} | {r.trigger} | {r.action_if_triggered} |')
        w('')

    if active['falsification']:
        w('## What would change our mind')
        w('')
        if state.upgrade_if:
            w(f'- **Upgrade if:** {state.upgrade_if}')
        if state.downgrade_if:
            w(f'- **Downgrade if:** {state.downgrade_if}')
        w('')

    if state.qc_findings:
        w('---')
        w('')
        w('## Review notes')
        w('')
        w('*Raised by the independent review pass and left visible rather than '
          'silently resolved.*')
        w('')
        for f in state.qc_findings:
            w(f'- {f}')
        w('')

    return '\n'.join(out).strip() + '\n'


def thesis_changelog(prior: Optional[ReviewState], current: ReviewState) -> List[str]:
    """What moved between two reviews.

    The point of persisting state: the second review of a name should say what
    changed, not restate the thesis from scratch.
    """
    if not prior:
        return []
    out: List[str] = []
    if prior.rating and current.rating and prior.rating != current.rating:
        out.append(f'Rating: {prior.rating} → {current.rating}')
    if prior.conviction is not None and current.conviction is not None:
        if abs(prior.conviction - current.conviction) >= 0.1:
            out.append(f'Conviction: {prior.conviction:.1f} → {current.conviction:.1f}')
    prior_kpis = {k.name: k for k in prior.kpis}
    for k in current.kpis:
        was = prior_kpis.get(k.name)
        if was and was.status() != k.status():
            out.append(f'{k.name}: {was.status()} → {k.status()}')
    pb = {s.name.lower(): scenario_target(s) for s in prior.scenarios}
    cb = {s.name.lower(): scenario_target(s) for s in current.scenarios}
    if pb.get('base') and cb.get('base') and abs(pb['base'] - cb['base']) > 0.01:
        out.append(f'Base target: ${pb["base"]:,.2f} → ${cb["base"]:,.2f}')
    return out


# --------------------------------------------------------------------------
# HTML, straight from the state
# --------------------------------------------------------------------------

# Built from the state rather than by converting the markdown. The note
# generator has its own markdown->HTML converter; borrowing it would have meant
# either editing that route or keeping a second copy, and a second copy of a
# renderer has already drifted once in this codebase. Structured state makes
# the question moot -- there is nothing to parse.

_CSS = """
@page { size: letter portrait; margin: 0.6in; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.5pt; color: #1e293b; line-height: 1.45; }
h1 { font-size: 17pt; margin: 0 0 2px 0; }
h2 { font-size: 12pt; border-bottom: 2px solid #1e3a5f; padding-bottom: 3px;
     margin: 16px 0 7px 0; -pdf-keep-with-next: true; }
h3 { font-size: 10.5pt; margin: 11px 0 4px 0; -pdf-keep-with-next: true; }
p  { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 7px 0; font-size: 8.5pt;
        page-break-inside: avoid; table-layout: fixed; }
th { background: #f1f5f9; font-weight: bold; text-align: center; }
th, td { border: 1px solid #d1d5db; padding: 4px 5px; vertical-align: top;
         word-wrap: break-word; }
.rule { border: none; border-top: 1px solid #cbd5e1; margin: 14px 0; }
.head { font-size: 11pt; font-weight: bold; margin: 2px 0 6px 0; }
.sub  { color: #475569; margin: 0 0 10px 0; }
.on   { color: #15803d; font-weight: bold; }
.off  { color: #b91c1c; font-weight: bold; }
.watch{ color: #b45309; font-weight: bold; }
.qc   { background: #fffbeb; border: 1px solid #fde68a; padding: 8px; margin: 8px 0; }
"""

_STATUS_CLASS = {'on-thesis': 'on', 'off-thesis': 'off', 'watch': 'watch'}


def _esc(v) -> str:
    return (str(v if v is not None else '')
            .replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _table(headers: List[str], rows: List[List[str]], widths: List[int]) -> str:
    """Explicit widths: xhtml2pdf ignores percentages without table-layout fixed
    and an explicit width on every cell."""
    h = ''.join(f'<th width="{w}%" style="width:{w}%">{_esc(x)}</th>'
                for x, w in zip(headers, widths))
    body = ''
    for r in rows:
        body += '<tr>' + ''.join(
            f'<td width="{w}%" style="width:{w}%">{c}</td>'
            for c, w in zip(r, widths)) + '</tr>'
    return f'<table><tr>{h}</tr>{body}</table>'


def render_html(state: ReviewState, mode: str = 'review') -> str:
    active = active_blocks(state, mode)
    er = expected_return(state)
    mc, ev = market_cap(state), enterprise_value(state)
    o: List[str] = []
    w = o.append

    w(f'<h1>{_esc(state.company or state.ticker)} ({_esc(state.ticker)})</h1>')
    bits = []
    if state.rating and state.rating.lower() != 'none':
        bits.append(_esc(state.rating.upper()))
    if state.conviction is not None:
        bits.append(f'Conviction {state.conviction:.1f}/10')
    if state.horizon:
        bits.append(f'Horizon {_esc(state.horizon)}')
    if bits:
        w(f'<p class="head">{" &nbsp;|&nbsp; ".join(bits)}</p>')
    if state.price:
        line = f'Price {_fmt(state.price, "$", 2)}'
        if state.price_date:
            line += f' (close {_esc(state.price_date)})'
        if mc:
            line += f' &middot; Market cap ${mc/1000:,.1f}bn'
        if ev:
            line += f' &middot; EV ${ev/1000:,.1f}bn'
        w(f'<p class="sub">{line}</p>')

    if state.thesis:
        w('<h2>Thesis</h2>')
        for t in state.thesis[:3]:
            w(f'<p>&bull; {_esc(t)}</p>')

    if active['valuation']:
        w('<h2>Scenarios and expected return</h2>')
        rows = []
        for s in normalise_probabilities(list(state.scenarios)):
            tgt = scenario_target(s)
            rows.append([_esc(s.name.title()), f'{(s.probability or 0)*100:.0f}%',
                         _fmt(tgt, '$', 2), _fmt(upside_pct(tgt, state.price), '%'),
                         _esc(s.assumptions)])
        if er:
            rows.append(['<b>Weighted</b>', '100%',
                         f'<b>{_fmt(er["weighted_target"], "$", 2)}</b>',
                         f'<b>{_fmt(er["expected_return_pct"], "%")}</b>',
                         'probability-weighted'])
        w(_table(['Scenario', 'Prob', 'Target', 'vs price', 'Assumptions'],
                 rows, [12, 8, 13, 12, 55]))

    if active['changes']:
        w('<h2>What changed since the last review</h2>')
        mark = {'positive': '+', 'negative': '&minus;', 'neutral': '='}
        w(_table(['Change', 'Prior', 'Current', '', 'Thesis impact'],
                 [[_esc(c.item), _esc(c.prior), _esc(c.current),
                   mark.get(c.direction, '='), _esc(c.implication)]
                  for c in state.changes[:5]], [22, 15, 15, 4, 44]))

    if active['scorecard']:
        w('<h2>Thesis scorecard</h2>')
        rank = {'critical': 0, 'important': 1, 'secondary': 2}
        rows = []
        for k in sorted(state.kpis, key=lambda x: rank.get(x.importance, 3)):
            st = k.status()
            rows.append([_esc(k.name), _fmt(k.prior, k.unit), _fmt(k.current, k.unit),
                         _fmt(k.bull_threshold, k.unit), _fmt(k.bear_threshold, k.unit),
                         k.trend(),
                         f'<span class="{_STATUS_CLASS.get(st, "")}">'
                         f'{_STATUS_MARK.get(st, st)}</span>'])
        w(_table(['KPI', 'Prior', 'Current', 'Bull', 'Bear', 'Trend', 'Status'],
                 rows, [34, 11, 11, 11, 11, 8, 14]))

    pos = []
    if state.add_below:
        pos.append(f'Add below {_fmt(state.add_below, "$", 2)}')
    if state.trim_above:
        pos.append(f'Trim above {_fmt(state.trim_above, "$", 2)}')
    if pos:
        w('<h2>Positioning</h2>')
        w(f'<p>{" &middot; ".join(pos)}</p>')
    if state.key_question:
        w(f'<p><b>The question that decides this:</b> {_esc(state.key_question)}</p>')

    if mode != 'flash':
        variant = [v for v in state.variant_views if v.is_variant()]
        if active['variant_views']:
            w('<hr class="rule"><h2>Expectations and where we differ</h2>')
            for i, v in enumerate(variant, 1):
                w(f'<h3>{i}. {_esc(v.question)}</h3>')
                for lbl, val in (('Consensus', v.consensus), ('Our view', v.our_view),
                                 ('Why we differ', v.why_different),
                                 ('Supporting', v.supporting_evidence),
                                 ('Against us', v.disconfirming_evidence)):
                    if val:
                        w(f'<p>&bull; <b>{lbl}:</b> {_esc(val)}</p>')
                if v.resolves_when:
                    res = v.resolves_when + (f' ({v.resolution_date})' if v.resolution_date else '')
                    w(f'<p>&bull; <b>Resolves:</b> {_esc(res)}</p>')

        if active['priced_in']:
            w('<h2>What is priced in</h2>')
            if state.implied_growth_pct is not None:
                w(f'<p>At {_fmt(state.price, "$", 2)} the price requires roughly '
                  f'<b>{state.implied_growth_pct}% annual growth</b> in the '
                  f'valuation metric to clear a normal return over the stated '
                  f'horizon.</p>')
            if state.priced_in:
                w(f'<p>{_esc(state.priced_in)}</p>')

        if active['estimates']:
            w('<hr class="rule"><h2>Where our numbers differ from the street</h2>')
            rows = []
            for e in state.estimates:
                street, ours = e.get('street'), e.get('ours')
                delta = '&mdash;'
                if isinstance(street, (int, float)) and isinstance(ours, (int, float)) and street:
                    delta = f'{(ours - street) / abs(street) * 100:+.1f}%'
                rows.append([_esc(e.get('metric', '')), _esc(e.get('period', '')),
                             _fmt(street, e.get('unit', '')), _fmt(ours, e.get('unit', '')),
                             delta, _esc(e.get('why_different', ''))])
            w(_table(['Metric', 'Period', 'Street', 'Ours', 'Δ', 'Why'],
                     rows, [22, 11, 11, 11, 9, 36]))

        if active['business_quality']:
            w('<hr class="rule"><h2>Business quality</h2>')
            w(f'<p>{_esc(state.business_quality)}</p>')

        if active['catalysts']:
            w('<hr class="rule"><h2>Catalysts</h2>')
            w(_table(['Window', 'Event', 'Watch', 'Thesis impact'],
                     [[_esc(c.window), _esc(c.event), _esc(c.key_metric),
                       _esc(c.thesis_impact)] for c in state.catalysts],
                     [14, 28, 22, 36]))

        if active['risks']:
            w('<h2>Risks and what would trigger action</h2>')
            w(_table(['Risk', 'Prob', 'Sev', 'Evidence today', 'Trigger', 'Action if triggered'],
                     [[_esc(r.risk), f'{(r.probability or 0)*100:.0f}%', _esc(r.severity),
                       _esc(r.evidence_today), _esc(r.trigger), _esc(r.action_if_triggered)]
                      for r in sorted(state.risks, key=lambda x: -(x.probability or 0))],
                     [20, 7, 8, 21, 23, 21]))

        if active['falsification']:
            w('<h2>What would change our mind</h2>')
            if state.upgrade_if:
                w(f'<p>&bull; <b>Upgrade if:</b> {_esc(state.upgrade_if)}</p>')
            if state.downgrade_if:
                w(f'<p>&bull; <b>Downgrade if:</b> {_esc(state.downgrade_if)}</p>')

    if state.qc_findings:
        w('<hr class="rule"><h2>Review notes</h2>')
        w('<div class="qc">')
        w('<p><i>Raised by the independent review pass and left visible rather '
          'than silently resolved.</i></p>')
        for f in state.qc_findings:
            w(f'<p>&bull; {_esc(f)}</p>')
        w('</div>')

    return f'<html><head><style>{_CSS}</style></head><body>{"".join(o)}</body></html>'


def render_pdf(state: ReviewState, mode: str = 'review') -> bytes:
    """PDF bytes, or b'' when the renderer is unavailable."""
    import io as _io
    try:
        from xhtml2pdf import pisa
    except Exception:
        return b''
    buf = _io.BytesIO()
    pisa.CreatePDF(_io.StringIO(render_html(state, mode)), dest=buf)
    return buf.getvalue()
