"""Tests for Deep Dive — canonical research object and its editorial budgets.

The budgets are not style preferences. The one-pager renders onto a FIXED
1024x1536 canvas, so text that exceeds its budget does not reflow, it clips —
and a silently clipped kill-criterion is a research artifact that lies. These
tests pin the measurement, and use the Deere golden fixture (the reviewed
calibration standard) as the reference for what a correct object looks like.
"""
import json
import os

import pytest

import app_v3
import deepdive as dd

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'fixtures', 'deepdive_de_golden.json')


@pytest.fixture(scope='module')
def golden():
    return dd.load_golden_fixture(GOLDEN)


@pytest.fixture
def clean_runs():
    dd.ensure_schema(app_v3.get_db)
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('DELETE FROM deepdive_runs')
    yield
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('DELETE FROM deepdive_runs')


# ---------------------------------------------------------------------------
# the golden fixture is the calibration standard
# ---------------------------------------------------------------------------

def test_golden_onepager_passes_every_editorial_budget(golden):
    """If the reviewed reference artifact fails, the checker is wrong."""
    _master, onepager, _sources = golden
    assert dd.onepager_violations(onepager) == []


def test_golden_master_is_structurally_renderable(golden):
    master, _op, _s = golden
    assert dd.master_violations(master) == []


def test_golden_carries_the_shapes_the_renderers_index_into(golden):
    """The DE fixture deliberately exercises the hard cases."""
    _m, op, _s = golden
    assert len(op['signposts']) == 6          # six rows, all must be visible
    assert len(op['threats']) == 4            # four kill criteria
    assert len(op['bull_case']) == 5 and len(op['bear_case']) == 5


def test_golden_segments_sum_to_a_drawable_pie(golden):
    master, _op, _s = golden
    segs = master['company_overview']['segments']
    total = sum(s.get('mix_numeric') or 0 for s in segs)
    assert 85 <= total <= 115, f'pie would not close: {total}'


# ---------------------------------------------------------------------------
# budget enforcement
# ---------------------------------------------------------------------------

def test_overlong_prose_is_caught(golden):
    _m, op, _s = golden
    bad = dict(op, bottom_line=' '.join(['word'] * 30))
    assert any('bottom_line' in v for v in dd.onepager_violations(bad))


def test_a_short_thesis_summary_is_not_a_violation(golden):
    """The reference one-pager runs ~42 words here on purpose.

    Only the upper bound can clip the canvas; under-length just leaves space.
    """
    _m, op, _s = golden
    assert dd._words(op['thesis_summary']) < 80
    assert dd.onepager_violations(op) == []
    over = dict(op, thesis_summary=' '.join(['word'] * 200))
    assert any('thesis_summary' in v for v in dd.onepager_violations(over))


def test_wrong_list_lengths_are_caught(golden):
    """Renderers index these directly — five signposts leaves a hole."""
    _m, op, _s = golden
    short = dict(op, signposts=op['signposts'][:5])
    assert any('signposts' in v and 'exactly 6' in v
               for v in dd.onepager_violations(short))


def test_a_verbose_signpost_cell_is_caught(golden):
    """A long cell wraps the row taller and pushes the last signpost off-page."""
    _m, op, _s = golden
    sps = [dict(s) for s in op['signposts']]
    sps[0]['current'] = ' '.join(['word'] * 25)
    assert any('signposts[0].current' in v
               for v in dd.onepager_violations(dict(op, signposts=sps)))


def test_violations_name_the_field_so_repair_can_be_targeted(golden):
    _m, op, _s = golden
    bad = dict(op, headline=' '.join(['word'] * 40), threats=op['threats'][:2])
    problems = dd.onepager_violations(bad)
    assert any('headline' in p for p in problems)
    assert any('threats' in p for p in problems)


def test_master_violations_flags_an_unclosable_pie():
    master = {'company': 'X', 'ticker': 'X',
              'investment_thesis': {'summary': 's'},
              'signposts': [1], 'thesis_threats': [1], 'opportunities': [1],
              'company_overview': {'segments': [{'mix_numeric': 20},
                                                {'mix_numeric': 30}]}}
    assert any('mix_numeric' in v for v in dd.master_violations(master))


def test_master_violations_accepts_all_zero_segments():
    """All-zero is the documented way to say 'shares unsupported' — not an error."""
    master = {'company': 'X', 'ticker': 'X',
              'investment_thesis': {'summary': 's'},
              'signposts': [1], 'thesis_threats': [1], 'opportunities': [1],
              'company_overview': {'segments': [{'mix_numeric': 0},
                                                {'mix_numeric': 0}]}}
    assert dd.master_violations(master) == []


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_a_run_round_trips(golden, clean_db, clean_runs):
    master, op, sources = golden
    rid = dd.save_run(app_v3.get_db, 'DE', 'Deere & Company', master, op,
                      sources, [], {'golden_fixture': True})
    run = dd.load_run(app_v3.get_db, rid)
    assert run['ticker'] == 'DE'
    assert run['master']['company'] == 'Deere & Company'
    assert len(run['onepager']['signposts']) == 6


def test_latest_wins_and_history_is_capped(golden, clean_db, clean_runs):
    master, op, _s = golden
    for i in range(6):
        dd.save_run(app_v3.get_db, 'DE', f'run{i}', dict(master, company=f'run{i}'),
                    op, [], [], {}, max_runs=3)
    assert dd.load_latest(app_v3.get_db, 'DE')['master']['company'] == 'run5'
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) c FROM deepdive_runs WHERE ticker='DE'")
        assert cur.fetchone()['c'] == 3


def test_list_shows_one_entry_per_ticker(golden, clean_db, clean_runs):
    master, op, _s = golden
    dd.save_run(app_v3.get_db, 'DE', 'Deere', master, op, [], [], {})
    dd.save_run(app_v3.get_db, 'DE', 'Deere', master, op, [], [], {})
    dd.save_run(app_v3.get_db, 'CVS', 'CVS', dict(master, ticker='CVS'), op, [], [], {})
    tickers = [r['ticker'] for r in dd.list_runs(app_v3.get_db)]
    assert sorted(tickers) == ['CVS', 'DE']


def test_freshness_window(golden, clean_db, clean_runs):
    master, op, _s = golden
    rid = dd.save_run(app_v3.get_db, 'DE', 'Deere', master, op, [], [], {})
    assert dd.is_fresh(dd.load_run(app_v3.get_db, rid)) is True
    assert dd.is_fresh(dd.load_run(app_v3.get_db, rid), ttl_minutes=0) is False
    assert dd.is_fresh(None) is False


# ---------------------------------------------------------------------------
# market snapshot must never block the run
# ---------------------------------------------------------------------------

def test_market_snapshot_survives_a_broken_provider(monkeypatch):
    """Research is the expensive part; a quote outage must not abort it."""
    import sys, types
    broken = types.ModuleType('yfinance')

    def _boom(*a, **k):
        raise RuntimeError('yfinance down')
    broken.Ticker = _boom
    monkeypatch.setitem(sys.modules, 'yfinance', broken)
    out = dd.market_snapshot('DE')
    assert out['ok'] is False and out['ticker'] == 'DE'


def test_merge_live_market_overlays_only_verified_fields(golden):
    master, _op, _s = golden
    before_hq = master['at_glance'].get('hq')
    merged = dd.merge_live_market(dict(master), {
        'ok': True, 'share_price': '$999.00', 'market_cap': '~$1.0T'})
    assert merged['at_glance']['share_price'] == '$999.00'
    assert merged['at_glance']['hq'] == before_hq   # researched fields untouched


def test_merge_is_a_noop_when_the_snapshot_failed(golden):
    master, _op, _s = golden
    merged = dd.merge_live_market(dict(master), {'ok': False})
    assert merged['at_glance'] == master['at_glance']


# ---------------------------------------------------------------------------
# the one rule: the one-pager never re-researches
# ---------------------------------------------------------------------------

def test_compress_onepager_sees_only_the_canonical_object(golden):
    """Its prompt must carry the master object and no search tool."""
    master, op, _s = golden
    captured = {}

    def fake_llm(**kwargs):
        captured['user'] = kwargs['messages'][0]['content']
        captured['system'] = kwargs['system']
        return {'text': json.dumps(op)}

    out, violations = dd.compress_onepager(master, fake_llm, json.loads)
    assert violations == []
    assert 'CANONICAL RESEARCH OBJECT' in captured['user']
    assert 'Do NOT add new facts' in captured['system']
    assert out['ticker'] == 'DE'


def test_a_repair_pass_that_makes_things_worse_is_discarded(golden):
    """Last answer should not win merely by being last."""
    master, op, _s = golden
    bad = dict(op, headline=' '.join(['word'] * 40))     # 1 violation
    worse = dict(op, headline=' '.join(['word'] * 40),
                 signposts=op['signposts'][:3])          # more violations
    calls = []

    def fake_llm(**kwargs):
        calls.append(1)
        return {'text': json.dumps(bad if len(calls) == 1 else worse)}

    out, violations = dd.compress_onepager(master, fake_llm, json.loads)
    assert len(calls) == 2, 'a violating object should trigger one repair pass'
    # Identify the winner by a structural difference rather than by length:
    # output is hard-trimmed to budget afterwards, so word counts no longer
    # distinguish the two candidates. `worse` dropped to three signposts.
    assert len(out['signposts']) == 6, 'the worse repair must be rejected'


def test_a_failed_repair_call_keeps_the_original(golden):
    master, op, _s = golden
    bad = dict(op, headline=' '.join(['word'] * 40))
    calls = []

    def fake_llm(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {'text': json.dumps(bad)}
        raise RuntimeError('model down')

    out, violations = dd.compress_onepager(master, fake_llm, json.loads)
    assert len(violations) == 1 and out['headline'].startswith('word')


def test_research_requires_a_usable_json_object(golden):
    """Three renderers fed an empty dict is worse than a loud failure."""
    with pytest.raises(RuntimeError, match='no usable JSON'):
        dd.research_company('DE', 'Deere', {}, lambda **k: {'text': 'not json'},
                            lambda t: None, lambda q, **k: [])


def test_search_failures_degrade_to_less_context_not_a_crash():
    def boom(query, **kwargs):
        raise RuntimeError('tavily down')
    context, sources = dd.gather_web_context('DE', 'Deere', boom)
    assert context == '' and sources == []


def test_sources_are_deduplicated_and_carry_urls():
    def fake_search(query, **kwargs):
        return [{'title': 'A', 'url': 'https://x.com/a', 'content': 'body a'},
                {'title': 'A dup', 'url': 'https://x.com/a', 'content': 'dup'}]
    context, sources = dd.gather_web_context('DE', 'Deere', fake_search)
    assert len(sources) == 1
    assert sources[0]['url'] == 'https://x.com/a'
    assert 'body a' in context


def test_a_freshly_saved_run_is_actually_fresh(golden, clean_db, clean_runs):
    """Regression: created_at was stored with a bare NOW().

    On a server whose local time is not UTC that made every run read as hours
    old the instant it was written, so the 30-minute cache never hit and every
    request paid for a full research pass.
    """
    master, op, _s = golden
    rid = dd.save_run(app_v3.get_db, 'DE', 'Deere', master, op, [], [], {})
    run = dd.load_run(app_v3.get_db, rid)

    from datetime import datetime
    age_min = (datetime.utcnow() - datetime.fromisoformat(run['createdAt'])
               ).total_seconds() / 60
    assert abs(age_min) < 5, f'stored clock is skewed by {age_min:.0f} minutes'
    assert dd.is_fresh(run) is True


# ---------------------------------------------------------------------------
# budgets must stay calibrated to the artifact that fits
# ---------------------------------------------------------------------------

def test_budgets_are_tight_enough_to_catch_a_two_times_overrun(golden):
    """The universal-template failure, pinned.

    The one-pager's boxes are fixed rectangles at absolute coordinates, so
    over-budget text overlaps the next section rather than reflowing. The first
    set of budgets allowed ~2.5x the reference volume (115 words of
    thesis_summary where the fitting artifact uses 42), so a real company came
    back at roughly double DE in every field, passed with six trivial
    violations, and rendered as overlapping mush.
    """
    _m, de_op, _s = golden

    # Roughly what a verbose company returns: every prose field doubled.
    verbose = dict(de_op)
    verbose['thesis_summary'] = ' '.join(['word'] * 93)
    verbose['overview_summary'] = ' '.join(['word'] * 58)
    verbose['final_takeaway'] = ' '.join(['word'] * 62)
    verbose['bull_case'] = [' '.join(['word'] * 11)] * 5

    problems = dd.onepager_violations(verbose)
    for field in ('thesis_summary', 'overview_summary', 'final_takeaway', 'bull_case'):
        assert any(field in p for p in problems), f'{field} overrun not caught'


def test_the_reference_artifact_still_passes_the_tighter_budgets(golden):
    """Tightening must not condemn the calibration standard itself."""
    _m, de_op, _s = golden
    assert dd.onepager_violations(de_op) == []


def test_nested_item_budgets_are_enforced(golden):
    """Opportunity/threat/segment prose sits in fixed boxes too."""
    _m, de_op, _s = golden
    op = dict(de_op)
    op['opportunities'] = [dict(o, detail=' '.join(['word'] * 30))
                           for o in de_op['opportunities']]
    assert any('opportunities[0].detail' in p for p in dd.onepager_violations(op))


def test_reported_violations_describe_what_was_stored(golden):
    """A trimmed field is no longer over budget, so it must not still be flagged.

    Reporting "headline: 40 words (max 10)" after the headline has been cut to
    10 would point at a problem that no longer exists in the saved artifact.
    What matters to the reader is that the research came back too verbose.
    """
    master, op, _s = golden
    verbose = dict(op, headline=' '.join(['word'] * 40))

    def fake_llm(**kwargs):
        return {'text': json.dumps(verbose)}

    stored, violations = dd.compress_onepager(master, fake_llm, json.loads)
    assert dd.onepager_violations(stored) == [], 'stored object must be in budget'
    assert len(violations) == 1
    assert violations[0].startswith('auto-trimmed to fit:')
    assert 'headline' in violations[0]


# ---------------------------------------------------------------------------
# web research
# ---------------------------------------------------------------------------

def test_web_research_collects_sources_from_citations_and_results(golden, monkeypatch):
    """The source trail is the only record that the research was grounded.

    Sources arrive two ways -- as citations attached to text blocks, and inside
    web_search_tool_result blocks -- and both must be captured and de-duped.
    """
    master, _op, _s = golden

    class _Cit:
        url = 'https://sec.gov/a'
        title = 'SEC filing'

    class _TextBlock:
        type = 'text'
        text = json.dumps(master)
        citations = [_Cit()]

    class _Result:
        url = 'https://ir.example.com/q3'
        title = 'Q3 release'
        page_age = '2026-01-02'

    class _SearchBlock:
        type = 'web_search_tool_result'
        content = [_Result(), _Cit()]      # _Cit repeats the citation URL

    class _Resp:
        content = [_SearchBlock(), _TextBlock()]

    class _Msgs:
        def create(self, **kwargs):
            _Msgs.kwargs = kwargs
            return _Resp()

    class _Client:
        def __init__(self, **kw): self.messages = _Msgs()

    import types, sys
    fake = types.ModuleType('anthropic')
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, 'anthropic', fake)

    out, sources = dd.research_with_web_search(
        'DE', 'Deere', {'ok': False}, 'key', json.loads)

    urls = [s['url'] for s in sources]
    assert 'https://sec.gov/a' in urls
    assert 'https://ir.example.com/q3' in urls
    assert len(urls) == len(set(urls)), 'sources must be de-duplicated'
    assert out['ticker'] == 'DE'
    # the search tool must actually be requested
    assert any(t.get('name') == 'web_search' for t in _Msgs.kwargs['tools'])


def test_web_research_fails_loudly_without_usable_json(monkeypatch):
    class _TextBlock:
        type = 'text'; text = 'I could not find anything.'; citations = []
    class _Resp: content = [_TextBlock()]
    class _Msgs:
        def create(self, **kw): return _Resp()
    class _Client:
        def __init__(self, **kw): self.messages = _Msgs()
    import types, sys
    fake = types.ModuleType('anthropic'); fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, 'anthropic', fake)

    with pytest.raises(RuntimeError, match='no usable JSON'):
        dd.research_with_web_search('DE', 'Deere', {}, 'key', lambda t: None)


# ---------------------------------------------------------------------------
# segment shares must close, deterministically
# ---------------------------------------------------------------------------

def test_overlapping_segment_shares_are_rescaled_to_100():
    """Two live runs came back at 119% and 118%, so the prompt is not enough.

    Companies report overlapping segments (Optum Rx sits inside Optum), and a
    pie cannot be drawn from shares that do not close. Rescaling preserves the
    relative sizes, which is the decision-useful part.
    """
    segs = [{'short_name': 'A', 'mix': '75%', 'mix_numeric': 75},
            {'short_name': 'B', 'mix': '16%', 'mix_numeric': 16},
            {'short_name': 'C', 'mix': '23%', 'mix_numeric': 23},
            {'short_name': 'D', 'mix': '4%', 'mix_numeric': 4}]
    out, changed = dd.normalize_segments(segs)
    assert changed is True
    assert sum(s['mix_numeric'] for s in out) == 100
    # order preserved, relative sizes preserved
    assert out[0]['mix_numeric'] > out[2]['mix_numeric'] > out[1]['mix_numeric']


def test_the_label_is_rewritten_so_it_agrees_with_the_wedge():
    """A legend saying 75% beside a wedge occupying 64% is the actual defect."""
    segs = [{'mix': '75%', 'mix_numeric': 75}, {'mix': '43%', 'mix_numeric': 43}]
    out, _ = dd.normalize_segments(segs)
    assert out[0]['mix'] == f"{out[0]['mix_numeric']}%"


def test_a_label_carrying_extra_meaning_is_left_alone():
    """Only bare percentages are rewritten; prose is not silently replaced."""
    segs = [{'mix': '75% rev, 55% op earnings', 'mix_numeric': 75},
            {'mix': '43%', 'mix_numeric': 43}]
    out, _ = dd.normalize_segments(segs)
    assert out[0]['mix'] == '75% rev, 55% op earnings'


def test_shares_that_already_close_are_untouched():
    segs = [{'mix': '60%', 'mix_numeric': 60}, {'mix': '40%', 'mix_numeric': 40}]
    out, changed = dd.normalize_segments(segs)
    assert changed is False and out is segs


def test_all_zero_segments_are_left_alone():
    """All-zero is the documented way to say 'shares unsupported'."""
    segs = [{'mix': 'N/A', 'mix_numeric': 0}, {'mix': 'N/A', 'mix_numeric': 0}]
    out, changed = dd.normalize_segments(segs)
    assert changed is False


# ---------------------------------------------------------------------------
# structural repair — a schema in a prompt is a request, not a contract
# ---------------------------------------------------------------------------

def test_a_self_wrapped_list_is_unwrapped():
    """A live run returned signposts as {"signposts": [...]}.

    The memo renderer calls .slice() on it, threw, and produced an entirely
    blank page -- so this is repaired rather than trusted.
    """
    m = {'signposts': {'signposts': [{'signpost': 'a'}, {'signpost': 'b'}]}}
    out, fixed = dd.coerce_master_shape(m)
    assert isinstance(out['signposts'], list) and len(out['signposts']) == 2
    assert 'signposts' in fixed


def test_a_keyed_object_standing_in_for_a_list_is_converted():
    m = {'opportunities': {'one': {'title': 'A'}, 'two': {'title': 'B'}}}
    out, fixed = dd.coerce_master_shape(m)
    assert [o['title'] for o in out['opportunities']] == ['A', 'B']


def test_a_scalar_where_a_list_belongs_is_wrapped():
    m = {'bull_case': 'Single point'}
    out, _ = dd.coerce_master_shape(m)
    assert out['bull_case'] == ['Single point']


def test_nested_paths_are_repaired_too():
    m = {'investment_thesis': {'falsification': {'falsification': ['x', 'y']}}}
    out, fixed = dd.coerce_master_shape(m)
    assert out['investment_thesis']['falsification'] == ['x', 'y']
    assert 'investment_thesis.falsification' in fixed


def test_well_formed_objects_are_untouched(golden):
    master, _op, _s = golden
    out, fixed = dd.coerce_master_shape(master)
    assert fixed == []


def test_shape_repair_runs_before_budgets(golden):
    """Budget enforcement skips non-lists, so repair has to come first."""
    master, _op, _s = golden
    broken = dict(master, signposts={'signposts': master['signposts']})
    out, _ = dd.enforce_master_budgets(broken)
    assert isinstance(out['signposts'], list)


def test_kpi_values_are_figures_not_sentences(golden):
    """A live run returned "$445-448 billion (2025 guidance)" as a KPI value.

    These render as the headline number on a card; Deere's are one word. Long
    values wrapped each card from ~110px to 165px, which was the whole of the
    remaining memo page-2 overflow.
    """
    master, _op, _s = golden
    verbose = json.loads(json.dumps(master))
    verbose['financial_snapshot']['revenue'] = (
        '$445-448 billion for full year 2025 under current management guidance, '
        'adjusted for divestitures')
    out, trimmed = dd.enforce_master_budgets(verbose)
    assert dd._words(out['financial_snapshot']['revenue']) <= 8
    assert any('revenue' in t for t in trimmed)


def test_the_reference_master_survives_every_cap(golden):
    """The guardrail: caps must never trim the artifact they were derived from.

    Tightening repeatedly threatened this -- twice a cap chosen to fix another
    company started cutting Deere, which would mean degrading the calibration
    standard to accommodate the thing being calibrated against it.
    """
    master, _op, _s = golden
    _out, trimmed = dd.enforce_master_budgets(master)
    assert trimmed == [], f'DE reference was trimmed: {trimmed}'


def test_buildinfo_reports_deployed_features_without_auth(client):
    """Deploy state must be observable from outside.

    The auth gate 401s every path including ones that do not exist, so a missing
    route and an unauthenticated caller are indistinguishable. That ambiguity
    made "is this deployed?" unanswerable during a live incident.
    """
    body = client.get('/api/buildinfo').get_json()
    assert body['ok'] is True
    assert body['features']['deepdive'] is True
    assert body['features']['explain'] is True
    assert body['routeCount'] > 50
    # nothing sensitive
    assert 'password' not in json.dumps(body).lower()


def test_analyze_reuses_an_inflight_run_for_the_same_ticker(client, monkeypatch):
    """A retried dispatch must not buy a second paid research run.

    The client retries this POST on transient network errors (Render restarts
    drop the socket mid-request), so a non-idempotent dispatch would spend real
    model budget twice for one user click.
    """
    import app_v3
    app_v3._deepdive_jobs.clear()
    app_v3._deepdive_jobs['abc123'] = {
        'status': 'running', 'step': 'Researching...', 'ticker': 'DE',
        'createdAt': '2026-08-29T00:00:00',
    }
    r = client.post('/api/deepdive/analyze',
                    json={'ticker': 'DE', 'apiKey': 'sk-test'})
    body = r.get_json()
    assert body['jobId'] == 'abc123'
    assert body['reused'] is True
    assert len(app_v3._deepdive_jobs) == 1, 'no second job may be spawned'


def test_force_still_starts_a_fresh_run_despite_an_inflight_one(client):
    """Dedup must not defeat the explicit "Force fresh" control."""
    import app_v3
    app_v3._deepdive_jobs.clear()
    app_v3._deepdive_jobs['abc123'] = {
        'status': 'running', 'step': '...', 'ticker': 'DE',
        'createdAt': '2026-08-29T00:00:00',
    }
    r = client.post('/api/deepdive/analyze',
                    json={'ticker': 'DE', 'force': True, 'apiKey': 'sk-test'})
    assert r.get_json().get('reused') is not True


def test_no_unscoped_blanket_visibility_hide_in_print_css():
    """A print stylesheet may not blank the whole page unconditionally.

    src/onepager.css did `body * { visibility: hidden }` and re-showed only
    .op-sheet. Deep Dive prints a different element, so it laid out perfectly --
    right page size, every element measurable via getBBox -- and painted
    nothing. The PDF was a correctly-sized blank page, which no geometry check
    or preflight could detect. Any blanket hide must name its context.
    """
    import glob, re
    offenders = []
    for path in glob.glob('src/*.css'):
        css = open(path, encoding='utf-8').read()
        for block in re.findall(r'@media\s+print\s*\{(.*?)\n\}', css, re.S):
            for rule in re.findall(r'([^{}]+)\{[^{}]*visibility\s*:\s*hidden', block):
                sel = rule.strip().split('\n')[-1].strip()
                # `body *` / `*` with no qualifier hides every future feature.
                if re.fullmatch(r'(html\s+)?(body\s+)?\*', sel):
                    offenders.append(f'{path}: {sel}')
    assert not offenders, (
        'Unscoped blanket visibility:hidden in a print block: '
        + '; '.join(offenders)
        + ' -- qualify it (e.g. body:not(.dd-printing) *) so it cannot silently '
          'blank an unrelated print feature.')
