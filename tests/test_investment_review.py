"""The investment review pipeline.

Separate from note generation and deliberately so: the unit here is structured
state, every derived figure is computed in code, and a second model reviews the
state before anything renders. These tests hold that boundary -- particularly
that the model is never the source of a number that arithmetic can produce.
"""
import base64
import json
import os
import uuid

import pytest

os.environ.setdefault('DATABASE_URL', 'postgresql://localhost/charlie_test')
import app_v3
import investment_review as ir


# --- deterministic layer ---------------------------------------------------

def test_derived_figures_match_hand_calculation():
    """Reproduced against a real CRM note, whose figures were checked by hand."""
    s = ir.ReviewState(ticker='CRM', price=260.11, shares_out_m=821.0,
                       net_debt_m=27900.0,
                       scenarios=[ir.Scenario('bear', 0.25, target_price=199),
                                  ir.Scenario('base', 0.50, target_price=286),
                                  ir.Scenario('bull', 0.25, target_price=344)])
    assert round(ir.market_cap(s) / 1000, 1) == 213.6
    assert round(ir.enterprise_value(s) / 1000, 1) == 241.5
    assert round(ir.upside_pct(199, 260.11), 1) == -23.5
    assert round(ir.upside_pct(344, 260.11), 1) == 32.3
    er = ir.expected_return(s)
    assert er['weighted_target'] == 278.75
    assert er['expected_return_pct'] == 7.2


def test_probabilities_are_rescaled_not_rejected():
    s = ir.ReviewState(ticker='X', price=100.0,
                       scenarios=[ir.Scenario('bear', 0.3, target_price=80),
                                  ir.Scenario('base', 0.5, target_price=110),
                                  ir.Scenario('bull', 0.3, target_price=140)])
    er = ir.expected_return(s)
    assert er is not None, 'a 1.1 total should be rescaled, not discarded'
    assert abs(sum(x.probability for x in s.scenarios) - 1.0) < 1e-9


def test_return_decomposition_separates_growth_from_rerating():
    """A return built on multiple expansion is a different proposition."""
    d = ir.return_decomposition(price=260.11, target=286, start_metric=15074,
                                end_metric=15924, start_multiple=16.0,
                                end_multiple=16.5, dividend_yield_pct=0.68)
    assert d['fundamental_growth_pct'] == 5.6
    assert d['multiple_change_pct'] == 3.1
    assert d['capital_returns_pct'] == 0.68


def test_implied_growth_answers_what_the_price_requires():
    g = ir.implied_growth(price=260.11, metric_per_share=16.69,
                          terminal_multiple=16.5, years=4)
    assert 6.0 < g < 9.0, g
    assert ir.implied_growth(None, 10, 15, 3) is None
    assert ir.implied_growth(100, 0, 15, 3) is None


# --- KPI semantics ---------------------------------------------------------

def test_kpi_status_uses_its_own_thresholds():
    on = ir.KPI('cRPO growth', current=14.0, prior=13.0, bull_threshold=14.0,
                bear_threshold=12.0)
    off = ir.KPI('Organic growth', current=7.0, prior=8.0, bull_threshold=8.0,
                 bear_threshold=7.0)
    assert on.status() == 'on-thesis' and on.trend() == '↑'
    assert off.status() == 'off-thesis' and off.trend() == '↓'


def test_a_lower_is_better_kpi_is_not_read_backwards():
    """Medical loss ratio falling is good; the scorecard has to know that."""
    mlr = ir.KPI('MLR', current=87.4, prior=89.0, bull_threshold=88.0,
                 bear_threshold=91.0, higher_is_better=False)
    assert mlr.trend() == '↑', 'a fall in MLR is an improvement'
    assert mlr.status() == 'on-thesis'


# --- consistency -----------------------------------------------------------

def test_a_rating_that_contradicts_its_own_expected_return_is_caught():
    s = ir.ReviewState(ticker='X', price=100.0, rating='own', horizon='12m',
                       scenarios=[ir.Scenario('bear', 0.5, target_price=70),
                                  ir.Scenario('base', 0.5, target_price=90)])
    found = ' '.join(ir.consistency_findings(s))
    assert 'expected return' in found


def test_unactionable_risks_and_thresholdless_kpis_are_flagged():
    s = ir.ReviewState(ticker='X', price=10.0,
                       risks=[ir.Risk('Competition', trigger='', action_if_triggered='')],
                       kpis=[ir.KPI('Revenue growth', current=5.0)])
    found = ' '.join(ir.consistency_findings(s))
    assert 'no trigger' in found and 'no threshold' in found


def test_a_debate_we_agree_with_is_not_a_variant_view():
    same = ir.VariantView('Q', consensus='8% growth', our_view='8% growth')
    diff = ir.VariantView('Q', consensus='8% growth', our_view='10% growth')
    assert not same.is_variant() and diff.is_variant()


# --- activation ------------------------------------------------------------

def test_blocks_with_nothing_behind_them_do_not_render():
    """An empty section reads as a template that ran, not an analyst who looked."""
    bare = ir.ReviewState(ticker='X', price=10.0)
    blocks = ir.active_blocks(bare)
    assert not blocks['scorecard'] and not blocks['valuation']
    assert not blocks['risks'] and not blocks['catalysts']
    html = ir.render_html(bare)
    for heading in ('Thesis scorecard', 'Catalysts', 'Risks and what'):
        assert heading not in html


def test_flash_mode_drops_everything_past_the_decision():
    s = ir.ReviewState(ticker='X', price=10.0,
                       variant_views=[ir.VariantView('Q', 'a', 'b')],
                       catalysts=[ir.Catalyst('E')],
                       risks=[ir.Risk('R', trigger='t', action_if_triggered='a')])
    flash = ir.render_markdown(s, 'flash')
    review = ir.render_markdown(s, 'review')
    assert 'Expectations and where we differ' not in flash
    assert 'Expectations and where we differ' in review


# --- changelog -------------------------------------------------------------

def test_the_changelog_reports_movement_not_the_whole_thesis():
    prior = ir.ReviewState(ticker='X', rating='own', conviction=8.0,
                           kpis=[ir.KPI('Growth', current=9.0, bull_threshold=8.0,
                                        bear_threshold=7.0)],
                           scenarios=[ir.Scenario('base', 1.0, target_price=100)])
    now = ir.ReviewState(ticker='X', rating='hold', conviction=7.5,
                         kpis=[ir.KPI('Growth', current=6.5, bull_threshold=8.0,
                                      bear_threshold=7.0)],
                         scenarios=[ir.Scenario('base', 1.0, target_price=115)])
    log = ir.thesis_changelog(prior, now)
    joined = ' | '.join(log)
    assert 'own → hold' in joined
    assert '8.0 → 7.5' in joined
    assert 'on-thesis → off-thesis' in joined
    assert '$100.00 → $115.00' in joined
    assert ir.thesis_changelog(None, now) == [], 'a first review has nothing to compare'


# --- prompt contract -------------------------------------------------------

def test_the_model_is_told_not_to_do_arithmetic():
    """The whole point of the deterministic layer."""
    p = ir.extract_prompt('CRM', 'Salesforce', 'software', 'review', 'PRICE: $260')
    assert 'Do NOT compute' in p
    assert 'calculated downstream in code' in ir.EXTRACT_SYSTEM


def test_the_reviewer_is_a_separate_skeptical_context():
    assert 'skeptical' in ir.QC_SYSTEM.lower()
    assert 'did not write it' in ir.QC_SYSTEM
    q = ir.qc_prompt('{}', '{}')
    assert 'strongest argument against' in q
    assert 'consistent with the expected return' in q


def test_sector_kpis_are_offered_only_where_we_have_a_view():
    assert 'cRPO growth' in ir.sector_kpis('software')
    assert 'Medical loss ratio' in ir.sector_kpis('managed care')
    assert ir.sector_kpis('Widget Fabrication') == []


# --- end to end ------------------------------------------------------------

MODEL_STATE = json.dumps({
    "company": "Salesforce, Inc.", "sector": "software", "rating": "hold",
    "conviction": 7.5, "horizon": "12 months",
    "shares_out_m": 821.0, "net_debt_m": 27900.0,
    "thesis": ["Front-office platform.", "Market mispriced disintermediation.",
               "Organic growth must reach 8%."],
    "changes": [{"item": "cRPO", "prior": "13%", "current": "14%",
                 "direction": "positive", "implication": "inflection"}],
    "kpis": [{"name": "Organic growth", "current": 7.0, "prior": 8.0, "unit": "%",
              "bull_threshold": 8.0, "bear_threshold": 7.0, "importance": "critical"},
             {"name": "cRPO growth", "current": 14.0, "prior": 13.0, "unit": "%",
              "bull_threshold": 14.0, "bear_threshold": 12.0, "importance": "critical"}],
    "variant_views": [{"question": "Growth above 8%?", "consensus": "7-8%",
                       "our_view": "8%+", "resolves_when": "3Q print"}],
    "scenarios": [{"name": "bear", "probability": 0.25, "target_price": 199},
                  {"name": "base", "probability": 0.50, "target_price": 286},
                  {"name": "bull", "probability": 0.25, "target_price": 344}],
    "risks": [{"risk": "Monetisation fails", "probability": 0.3, "severity": "high",
               "trigger": "refill below 30%", "action_if_triggered": "cut estimates"}],
    "catalysts": [{"event": "Dreamforce", "window": "16-Sep", "key_metric": "framework"}],
    "add_below": 230, "trim_above": 300, "key_question": "Does growth reach 8%?",
    "upgrade_if": "4Q above 8%", "downgrade_if": "refill below 30%",
    "facts": [{"statement": "cRPO grew 14% cc", "type": "reported_fact"},
              {"statement": "2H will reaccelerate", "type": "management_claim"}],
})

QC_REPLY = json.dumps({
    "findings": [{"issue": "Base case leans on multiple expansion the note calls done.",
                  "severity": "high", "where": "valuation", "fix": "state it"},
                 {"issue": "Trivial nit.", "severity": "low", "where": "", "fix": ""}],
    "verdict": "revise",
    "strongest_counterargument": "Total RPO decelerated.",
})


@pytest.fixture
def stub_review(monkeypatch):
    calls = []

    def fake(*, messages, system, model_key, max_tokens, api_key, **kw):
        calls.append({'system': system, 'model_key': model_key})
        text = QC_REPLY if 'skeptical' in (system or '').lower() else MODEL_STATE
        return {'text': text, 'provider': 'anthropic', 'model': 'stub',
                'usage': {}, 'stop_reason': 'end_turn'}

    monkeypatch.setattr(app_v3, '_call_pinned_long', fake)
    monkeypatch.setattr(app_v3, '_latest_close',
                        lambda t: {'price': 260.11, 'asOf': '2026-09-04'})
    return calls


def _seed(ticker):
    import io as _io
    try:
        from PyPDF2 import PdfWriter
        w = PdfWriter()
        w.add_blank_page(width=612, height=792)
        buf = _io.BytesIO(); w.write(buf)
        data = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        data = base64.b64encode(b'%PDF-1.4 stub').decode()
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM document_files WHERE ticker = %s", (ticker,))
        cur.execute("DELETE FROM investment_reviews WHERE ticker = %s", (ticker,))
        cur.execute("""INSERT INTO document_files (ticker, filename, file_data,
                       file_type, file_size) VALUES (%s,'10q.pdf',%s,'pdf',5000)""",
                    (ticker, data))


def test_a_review_runs_end_to_end_and_stores_state(stub_review):
    _seed('RVW')
    job = str(uuid.uuid4())
    app_v3._run_investment_review(job, 'RVW', 'test-key', 'review')
    assert app_v3._review_jobs[job]['status'] == 'complete', app_v3._review_jobs[job]

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT * FROM investment_reviews WHERE ticker='RVW'")
        row = cur.fetchone()
    state = row['state']
    if isinstance(state, str):
        state = json.loads(state)

    # price comes from the quote, never from the model
    assert state['price'] == 260.11
    assert state['rating'] == 'hold'
    assert len(state['kpis']) == 2
    assert 'Thesis scorecard' in row['review_markdown']
    assert row['review_pdf'], 'no PDF rendered'
    assert base64.b64decode(row['review_pdf'])[:5] == b'%PDF-'

    # both models were called, and the reviewer got its own system prompt
    systems = [c['system'] for c in stub_review]
    assert any('skeptical' in s.lower() for s in systems), 'no independent QC pass ran'
    assert any('structured investment state' in s for s in systems)


def test_high_severity_qc_findings_survive_into_the_memo(stub_review):
    """A finding that is quietly fixed is a finding the reader never sees."""
    _seed('RVW2')
    job = str(uuid.uuid4())
    app_v3._run_investment_review(job, 'RVW2', 'test-key', 'review')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT review_markdown FROM investment_reviews WHERE ticker='RVW2'")
        md = cur.fetchone()['review_markdown']
    assert 'multiple expansion' in md, 'the high-severity finding was dropped'
    assert 'Trivial nit' not in md, 'low-severity noise should not surface'


def test_the_second_review_reports_what_changed(stub_review):
    _seed('RVW3')
    app_v3._run_investment_review(str(uuid.uuid4()), 'RVW3', 'k', 'review')
    job2 = str(uuid.uuid4())
    app_v3._run_investment_review(job2, 'RVW3', 'k', 'review')
    # identical state twice, so nothing should be reported as having moved
    assert app_v3._review_jobs[job2]['changelog'] == []
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM investment_reviews WHERE ticker='RVW3'")
        assert cur.fetchone()['n'] == 2, 'history is kept, not overwritten'


def test_the_existing_note_generator_is_untouched():
    """This tab is additive. The old pipeline must still be exactly there."""
    assert hasattr(app_v3, '_generate_research_note')
    assert hasattr(app_v3, 'generate_research_note')
    rules = {str(r) for r in app_v3.app.url_map.iter_rules()}
    assert '/api/notes/generate' in rules
    assert '/api/review/generate' in rules
    assert '/api/notes/generate' != '/api/review/generate'


def _seed_many(ticker, names):
    import io as _io
    try:
        from PyPDF2 import PdfWriter
        w = PdfWriter(); w.add_blank_page(width=612, height=792)
        buf = _io.BytesIO(); w.write(buf)
        data = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        data = base64.b64encode(b'%PDF-1.4 stub').decode()
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM document_files WHERE ticker = %s", (ticker,))
        cur.execute("DELETE FROM investment_reviews WHERE ticker = %s", (ticker,))
        for n in names:
            cur.execute("""INSERT INTO document_files (ticker, filename, file_data,
                           file_type, file_size) VALUES (%s,%s,%s,'pdf',5000)""",
                        (ticker, n, data))


def test_only_the_selected_documents_are_read(stub_review):
    """The Review tab has a source picker; it has to actually filter.

    The first version of this pipeline loaded every document for the ticker and
    ignored any selection, which is not a defensible default when the note
    generator beside it honours one.
    """
    _seed_many('SELN', ['10k.pdf', 'broker-a.pdf', 'broker-b.pdf', 'transcript.pdf'])
    job = str(uuid.uuid4())
    app_v3._run_investment_review(job, 'SELN', 'k', 'review',
                                  file_selection=[{'filename': '10k.pdf'},
                                                  {'filename': 'transcript.pdf'}])
    assert app_v3._review_jobs[job]['status'] == 'complete', app_v3._review_jobs[job]
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT metadata FROM investment_reviews WHERE ticker='SELN'")
        meta = cur.fetchone()['metadata']
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert meta['documentsAvailable'] == 4
    assert meta['documentsSelected'] == 2, meta
    assert set(meta['documentsRead']) <= {'10k.pdf', 'transcript.pdf'}
    assert 'broker-a.pdf' not in meta['documentsRead']


def test_an_empty_selection_still_means_everything(stub_review):
    """Matching the note generator, where no config means use it all."""
    _seed_many('SELA', ['a.pdf', 'b.pdf'])
    job = str(uuid.uuid4())
    app_v3._run_investment_review(job, 'SELA', 'k', 'review', file_selection=[])
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT metadata FROM investment_reviews WHERE ticker='SELA'")
        meta = cur.fetchone()['metadata']
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert meta['documentsSelected'] == 2


def test_selecting_nothing_that_exists_fails_loudly(stub_review):
    """Better than silently reviewing on documents the user deselected."""
    _seed_many('SELX', ['a.pdf'])
    job = str(uuid.uuid4())
    app_v3._run_investment_review(job, 'SELX', 'k', 'review',
                                  file_selection=[{'filename': 'not-here.pdf'}])
    j = app_v3._review_jobs[job]
    assert j['status'] == 'failed'
    assert 'No documents selected' in j['error']


def test_documents_beyond_the_first_batch_are_named_not_dropped(stub_review, monkeypatch):
    """Only batch one is read; the reader has to be told which that was."""
    real = app_v3.notegen.plan_batches
    monkeypatch.setattr(app_v3.notegen, 'plan_batches',
                        lambda docs, **kw: [[docs[0]], docs[1:]] if len(docs) > 1
                        else real(docs, **kw))
    _seed_many('SELB', ['kept.pdf', 'left-out.pdf', 'also-out.pdf'])
    job = str(uuid.uuid4())
    app_v3._run_investment_review(job, 'SELB', 'k', 'review')
    with app_v3.get_db() as (_c, cur):
        cur.execute("""SELECT metadata, review_markdown FROM investment_reviews
                       WHERE ticker='SELB'""")
        row = cur.fetchone()
    meta = row['metadata']
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert set(meta['documentsNotRead']) == {'left-out.pdf', 'also-out.pdf'}
    assert 'left-out.pdf' in row['review_markdown'], (
        'documents that were not read must be named in the memo')


# --- defects found in the first real review ------------------------------

def test_units_are_placed_correctly():
    """A real scorecard printed "3,400$M" and "1,624M"."""
    assert ir._fmt(3400, '$M') == '$3,400M'
    assert ir._fmt(1624, 'M') == '1,624M'
    assert ir._fmt(14, '%') == '14%'
    assert ir._fmt(259.23, '$') == '$259.23'
    assert ir._fmt(None, '%') == '—'


def test_conviction_is_normalised_to_the_scale_it_is_shown_on():
    """0.6 arrived meaning 60% and rendered as "Conviction 0.6/10"."""
    assert ir.normalise_conviction(0.6) == 6.0
    assert ir.normalise_conviction(7.5) == 7.5
    assert ir.normalise_conviction(75) == 7.5
    assert ir.normalise_conviction(None) is None
    assert ir.normalise_conviction('nonsense') is None


def test_direction_is_taken_from_the_thresholds_not_a_flag():
    """Rising attrition read as on-thesis with a favourable trend.

    The model gave revenue attrition bull 8% and bear 9%, which already says
    lower is better, but left higher_is_better at its default.
    """
    worse = ir.KPI('Revenue attrition', current=9.5, prior=8.0,
                   bull_threshold=8.0, bear_threshold=9.0)
    assert worse.higher_is_better is False
    assert worse.trend() == '↓'
    assert worse.status() == 'off-thesis'

    better = ir.KPI('cRPO growth', current=14.0, prior=13.5,
                    bull_threshold=14.0, bear_threshold=11.0)
    assert better.higher_is_better is True
    assert better.status() == 'on-thesis'


def test_risks_are_ordered_by_expected_impact():
    """Probability alone led with a 60%-likely, low-severity item."""
    rs = [ir.Risk('likely but minor', 0.60, 'low'),
          ir.Risk('less likely but thesis-breaking', 0.30, 'high')]
    order = [r.risk for r in sorted(rs, key=ir._risk_weight, reverse=True)]
    assert order[0] == 'less likely but thesis-breaking'


def test_table_columns_widen_for_their_content():
    """"September-October 2026" in a 14% column printed over its neighbour."""
    import re as _re
    html = ir._table(['Window', 'Event', 'Watch', 'Impact'],
                     [['September-October 2026', 'Closing of Fin and Contentful',
                       'operating margin held at ~34.3%', 'neutral']],
                     [14, 28, 22, 36])
    widths = [float(x) for x in _re.findall(r'width="([\d.]+)%"', html)[:4]]
    assert abs(sum(widths) - 100) < 1.0
    assert widths[0] > 14.0, 'the Window column did not grow for its content'


def test_source_coverage_is_not_attributed_to_the_review_pass():
    """Which documents were read is our note, not the reviewer's finding."""
    s = ir.ReviewState(ticker='X', price=10.0,
                       coverage_note='6 of 12 selected documents were read.',
                       qc_findings=['The bear case is a softer base case.'])
    md = ir.render_markdown(s)
    html = ir.render_html(s)
    for doc in (md, html):
        assert 'Source coverage' in doc
        assert 'Review notes' in doc
        assert doc.index('Source coverage') < doc.index('Review notes')


def test_the_review_is_served_as_html_for_the_tab(stub_review):
    """renderMarkdown does not do tables and this memo is mostly tables."""
    _seed('HTM')
    app_v3._run_investment_review(str(uuid.uuid4()), 'HTM', 'k', 'review')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT review_html FROM investment_reviews WHERE ticker='HTM'")
        html = cur.fetchone()['review_html']
    assert html and '<table' in html, 'no HTML stored for the tab to render'
    assert '<html>' in html
