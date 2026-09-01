"""Run the note generator end to end with a stubbed model.

Everything about note generation was tested except generating a note. The
planner had unit tests, the routes were checked for registration, the charts
were rendered in isolation -- and the function that ties them together had
never once been executed, so a missing `import notegen` reached production and
surfaced as a NameError in a background thread on a live run.

The model is stubbed; nothing else is. This exercises the real document
loading, ranking, batching, response parsing, chart generation, DOCX build and
draft insert.
"""
import base64
import io
import json
import uuid

import pytest

import app_v3


def _pdf(pages=4):
    from PyPDF2 import PdfWriter
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    return base64.b64encode(buf.getvalue()).decode()


MODEL_REPLY = """===NOTE_START===
# ACME Corp (ACME)

**Conclusion:** Own.

## 1. Executive Summary
ACME compounds earnings at a mid-teens rate.
===NOTE_END===

===SOURCES_START===
Section 1 informed by the 10-K.
===SOURCES_END===

===REVENUE_CHART_DATA===
[{"segment": "Widgets", "revenue": 6000}, {"segment": "Services", "revenue": 4000}]
===REVENUE_CHART_END===

===PROFIT_CHART_DATA===
[{"segment": "Widgets", "profit": 1800}, {"segment": "Services", "profit": 300}]
===PROFIT_CHART_END===
"""


@pytest.fixture
def stub_llm(monkeypatch):
    """Fake the network, not the calling logic.

    This used to replace call_llm, which sat above everything that decides how a
    model is reached -- streaming, timeout, retry, the pinned-model fallback. A
    stub that high tests the prompt and the parsing and nothing else, so when
    note generation was failing in production on a 120-second non-streaming
    timeout, every test here passed. Stubbing the adapter instead means
    _call_pinned_long runs for real and only the HTTP call is faked.
    """
    calls = []

    def fake_stream(*, messages, system, model, max_tokens, timeout, api_key):
        calls.append({'messages': messages, 'system': system, 'model': model,
                      'max_tokens': max_tokens, 'timeout': timeout})
        return {'text': MODEL_REPLY, 'provider': 'anthropic', 'model': model,
                'usage': {'input_tokens': 0, 'output_tokens': 0}}

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return {'text': MODEL_REPLY, 'provider': 'stub', 'model': 'stub-1',
                'usage': {}}

    monkeypatch.setattr(app_v3, '_call_anthropic_stream', fake_stream)
    monkeypatch.setattr(app_v3, 'call_llm', fake_call_llm)
    return calls


def _seed_documents(ticker, docs):
    with app_v3.get_db(commit=True) as (_c, cur):
        for name, pages in docs:
            cur.execute(
                """INSERT INTO document_files (ticker, filename, file_data, file_type, file_size)
                   VALUES (%s, %s, %s, 'pdf', %s)""",
                (ticker, name, _pdf(pages), pages * 40_000))


def _job(ticker):
    job_id = str(uuid.uuid4())
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("""INSERT INTO research_pipeline_jobs
                       (id, batch_id, ticker, job_type, status, progress, current_step, total_steps, steps_detail)
                       VALUES (%s, %s, %s, 'note', 'queued', 0, 'x', 6, '{}')""",
                    (job_id, str(uuid.uuid4()), ticker))
    return job_id


def test_generating_a_note_runs_to_completion(client, stub_llm):
    _seed_documents('ACME', [('ACME_10-K_2025.pdf', 6), ('broker_note.pdf', 3)])
    job_id = _job('ACME')

    app_v3._generate_research_note(job_id, 'ACME', api_key='test-key', mode='new')

    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status, current_step FROM research_pipeline_jobs WHERE id = %s",
                    (job_id,))
        job = cur.fetchone()
    assert job['status'] == 'complete', job['current_step']

    with app_v3.get_db() as (_c, cur):
        cur.execute("""SELECT * FROM research_notes WHERE ticker = 'ACME'
                       ORDER BY created_at DESC LIMIT 1""")
        note = cur.fetchone()
    assert note is not None, 'no note row was written'
    assert note['status'] == 'draft', 'a generated note must land as a draft'
    assert 'ACME compounds earnings' in note['note_markdown']
    assert 'informed by the 10-K' in note['sources_markdown']


def test_the_primary_source_shapes_the_note(client, stub_llm):
    """Batch one writes the note, so the 10-K must be in it."""
    _seed_documents('BETA', [('zz_broker_note.pdf', 3), ('BETA_10-K_2025.pdf', 6)])
    app_v3._generate_research_note(_job('BETA'), 'BETA', api_key='k', mode='new')

    first_call = stub_llm[0]
    attached = json.dumps(first_call['messages'])
    assert 'BETA_10-K_2025.pdf' in attached, 'the 10-K must be in the first batch'


def test_both_charts_are_produced_when_the_splits_differ(client, stub_llm):
    _seed_documents('GAMMA', [('GAMMA_10-K.pdf', 4)])
    app_v3._generate_research_note(_job('GAMMA'), 'GAMMA', api_key='k', mode='new')

    with app_v3.get_db() as (_c, cur):
        cur.execute("""SELECT charts FROM research_notes WHERE ticker = 'GAMMA'
                       ORDER BY created_at DESC LIMIT 1""")
        charts = cur.fetchone()['charts']
    if isinstance(charts, str):
        charts = json.loads(charts)
    kinds = {c['type'] for c in charts}
    assert kinds == {'revenue', 'profit'}, kinds


def test_a_file_selection_limits_what_is_sent(client, stub_llm):
    _seed_documents('DELTA', [('keep_me.pdf', 3), ('leave_me.pdf', 3)])
    app_v3._generate_research_note(
        _job('DELTA'), 'DELTA', api_key='k', mode='new',
        file_selection=[{'filename': 'keep_me.pdf', 'folder': 'main'}])

    attached = json.dumps(stub_llm[0]['messages'])
    assert 'keep_me.pdf' in attached
    assert 'leave_me.pdf' not in attached


def test_a_ticker_with_no_documents_fails_loudly(client, stub_llm):
    job_id = _job('EPSILON')
    app_v3._generate_research_note(job_id, 'EPSILON', api_key='k', mode='new')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT status FROM research_pipeline_jobs WHERE id = %s", (job_id,))
        assert cur.fetchone()['status'] == 'failed'


def test_note_generation_streams_and_does_not_use_the_120s_default(clean_db, stub_llm):
    """The bug that killed CVS: a long note on a two-minute non-streaming call.

    _call_anthropic uses messages.create, which the SDK refuses when a request's
    estimated duration passes ten minutes -- reported as "Request timed out or
    interrupted", which reads like a network fault. call_llm's default timeout
    was 120s on top of that, with no retry, so one slow response discarded a job
    that had already spent minutes reading PDFs. Asserting on the adapter and
    the timeout keeps note generation off that path.
    """
    _seed_documents('STRM', [('annual-report-2025.pdf', 20)])
    job_id = _job('STRM')
    app_v3._generate_research_note(job_id, 'STRM', 'test-key', 'new', None)

    assert stub_llm, 'no model call was made'
    for call in stub_llm:
        assert 'timeout' in call, (
            'note generation reached a non-streaming adapter; it must go '
            'through _call_anthropic_stream')
        assert call['timeout'] >= 900, (
            f"timeout {call['timeout']}s is too short for a multi-PDF note; "
            'the 120s default is what failed in production')


def test_note_generation_retries_a_transient_failure(clean_db, monkeypatch):
    """A dropped connection mid-note must not throw the job away."""
    attempts = []

    def flaky(*, messages, system, model, max_tokens, timeout, api_key):
        attempts.append(model)
        if len(attempts) == 1:
            raise app_v3.anthropic.APITimeoutError(request=None)
        return {'text': MODEL_REPLY, 'provider': 'anthropic', 'model': model,
                'usage': {'input_tokens': 0, 'output_tokens': 0}}

    monkeypatch.setattr(app_v3, '_call_anthropic_stream', flaky)
    monkeypatch.setattr(app_v3.time, 'sleep', lambda *_a, **_k: None)

    _seed_documents('RTRY', [('annual-report-2025.pdf', 10)])
    job_id = _job('RTRY')
    app_v3._generate_research_note(job_id, 'RTRY', 'test-key', 'new', None)

    assert len(attempts) >= 2, 'the timeout was not retried'
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT status FROM research_pipeline_jobs WHERE id = %s', (job_id,))
        assert cur.fetchone()['status'] == 'complete'


def test_the_picked_model_is_the_one_called(clean_db, stub_llm):
    """Choosing Sonnet must actually call Sonnet, not the default."""
    _seed_documents('PICK', [('annual-report-2025.pdf', 10)])
    job_id = _job('PICK')
    app_v3._generate_research_note(job_id, 'PICK', 'test-key', 'new', None,
                                   model_key='sonnet-5')
    assert stub_llm[0]['model'] == 'claude-sonnet-5', stub_llm[0]['model']


def test_an_unknown_model_key_falls_back_instead_of_failing(clean_db, stub_llm):
    """A stale saved preference must not be able to fail a job."""
    _seed_documents('STAL', [('annual-report-2025.pdf', 10)])
    job_id = _job('STAL')
    app_v3._generate_research_note(job_id, 'STAL', 'test-key', 'new', None,
                                   model_key='model-that-was-retired')
    assert stub_llm[0]['model'] == app_v3.resolve_picker_model(
        app_v3.PICKER_DEFAULT_MODEL)


def test_horizontal_rules_do_not_print_as_literal_dashes():
    """A "---" between sections must become a rule, not text.

    The strict ^---+$ matched only a bare line, so a rule written with a trailing
    space or CRLF endings survived into the PDF as a literal "---" -- which is
    what appeared between sections 3/4 and 5/6 of a real note.
    """
    import re
    rx = re.compile(r'^[ \t]*-{3,}[ \t]*\r?$', re.MULTILINE)
    for label, src in (('plain', 'a\n---\nb'), ('trailing space', 'a\n--- \nb'),
                       ('CRLF', 'a\r\n---\r\nb'), ('indented', 'a\n  ---\nb')):
        assert rx.search(src), f'{label}: rule not recognised'
    # a table separator must NOT be swallowed as a rule
    assert not rx.search('|---|---|')


NOTE_WITHOUT_CHARTS = """===NOTE_START===
# Acme Corporation (ACME)

Segment margins recovering ahead of plan.

## 2. Business Overview
Health Services FY25 revenue ~$190B, adjusted operating income $7.25bn.
Benefits FY25 revenue ~$143B, adjusted operating income $5.03bn.
Pharmacy FY25 revenue ~$139B, adjusted operating income $6.40bn.
===NOTE_END===

===SOURCES_START===
Section 2 informed by the 10-K.
===SOURCES_END===
"""

CHART_ONLY_REPLY = """===REVENUE_CHART_DATA===
[{"segment": "Health Services", "revenue": 190000},
 {"segment": "Benefits", "revenue": 143000},
 {"segment": "Pharmacy", "revenue": 139000}]
===REVENUE_CHART_END===

===PROFIT_CHART_DATA===
[{"segment": "Health Services", "profit": 7250},
 {"segment": "Benefits", "profit": 5030},
 {"segment": "Pharmacy", "profit": 6400}]
===PROFIT_CHART_END===
"""


def test_a_note_without_chart_blocks_still_gets_charts(clean_db, monkeypatch):
    """The CVS failure: a complete note that arrives with no chart data.

    The blocks sat last in the output contract, after an 8-12 page note and a
    sources document, so a long response simply ended before reaching them. The
    note looked finished and the charts were gone with no error. A second,
    focused call reads the segment figures back out of the finished note.
    """
    calls = []

    def staged(*, messages, system, model, max_tokens, timeout, api_key):
        calls.append(max_tokens)
        # first call writes the note and omits the charts; the re-extract returns them
        return {'text': NOTE_WITHOUT_CHARTS if len(calls) == 1 else CHART_ONLY_REPLY,
                'provider': 'anthropic', 'model': model,
                'usage': {'input_tokens': 0, 'output_tokens': 0}}

    monkeypatch.setattr(app_v3, '_call_anthropic_stream', staged)
    _seed_documents('ACME', [('annual-report-2025.pdf', 10)])
    job_id = _job('ACME')
    app_v3._generate_research_note(job_id, 'ACME', 'test-key', 'new', None)

    assert len(calls) >= 2, 'no re-extract was attempted'
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT status, charts FROM research_notes WHERE ticker = %s', ('ACME',))
        row = cur.fetchone()
    charts = row['charts']
    if isinstance(charts, str):
        charts = json.loads(charts)
    kinds = {c['type'] for c in (charts or [])}
    assert 'revenue' in kinds, f'no revenue chart recovered (got {kinds})'
    assert 'profit' in kinds, f'no profit chart recovered (got {kinds})'
    # The bytes, not just the label. Asserting on 'type' alone passed happily
    # while the insert dropped 'data', so the note shipped with two empty chart
    # rows and no images -- which is the bug this test was written to catch.
    for c in charts:
        assert c.get('data'), f"chart {c['type']} stored with no image data"
        raw = base64.b64decode(c['data'])
        assert raw[:8] == b'\x89PNG\r\n\x1a\n', f"chart {c['type']} is not a PNG"
        assert len(raw) > 5000, f"chart {c['type']} suspiciously small ({len(raw)}b)" 


def test_the_prompt_carries_todays_price_not_the_documents(clean_db, stub_llm, monkeypatch):
    """A note dated today must not price the stock as of a source document.

    The CVS note said "All valuation data as of August 12, 2026" on 31 August,
    because August 12 was the newest price in the PDFs. Nothing told the model
    what day it was or what the stock cost, so it used what it had.
    """
    monkeypatch.setattr(app_v3, '_latest_close',
                        lambda t: {'price': 93.91, 'asOf': '2026-08-31'})
    _seed_documents('PXOK', [('broker-note-august.pdf', 10)])
    job_id = _job('PXOK')
    app_v3._generate_research_note(job_id, 'PXOK', 'test-key', 'new', None)

    sent = ''
    for block in stub_llm[0]['messages'][0]['content']:
        if block.get('type') == 'text':
            sent += block['text']
    assert '93.91' in sent, 'the live price never reached the prompt'
    assert '2026-08-31' in sent, 'the price date never reached the prompt'
    assert 'supersedes any price in the source documents' in sent


def test_a_missing_quote_does_not_fail_the_note(clean_db, stub_llm, monkeypatch):
    """No quote is a missing price line, not a failed job."""
    monkeypatch.setattr(app_v3, '_latest_close', lambda t: None)
    _seed_documents('NOPX', [('annual-report-2025.pdf', 10)])
    job_id = _job('NOPX')
    app_v3._generate_research_note(job_id, 'NOPX', 'test-key', 'new', None)
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT status FROM research_pipeline_jobs WHERE id = %s', (job_id,))
        assert cur.fetchone()['status'] == 'complete'


def test_the_price_helper_never_raises(monkeypatch):
    """yfinance failing must not take the note with it."""
    def boom(*a, **k):
        raise RuntimeError('network down')
    monkeypatch.setattr(app_v3, '_latest_close', app_v3._latest_close)
    import sys, types
    fake = types.ModuleType('yfinance')
    fake.Ticker = boom
    monkeypatch.setitem(sys.modules, 'yfinance', fake)
    assert app_v3._latest_close('ANY') is None


NOTE_WITH_DUPLICATE_PROFIT = """===REVENUE_CHART_DATA===
[{"segment": "Alpha", "revenue": 100000}, {"segment": "Beta", "revenue": 50000}]
===REVENUE_CHART_END===

===PROFIT_CHART_DATA===
[{"segment": "Alpha", "profit": 10000}, {"segment": "Beta", "profit": 5000}]
===PROFIT_CHART_END===

===NOTE_START===
# Acme Corporation (ACME)

Alpha operating income $9,000mn; Beta operating income $12,000mn.
===NOTE_END===
"""

REAL_PROFIT_REPLY = """===REVENUE_CHART_DATA===
[{"segment": "Alpha", "revenue": 100000}, {"segment": "Beta", "revenue": 50000}]
===REVENUE_CHART_END===

===PROFIT_CHART_DATA===
[{"segment": "Alpha", "profit": 9000}, {"segment": "Beta", "profit": 12000}]
===PROFIT_CHART_END===
"""


def test_a_profit_split_mirroring_revenue_triggers_a_re_extract(clean_db, monkeypatch):
    """Duplicate profit means the model reused revenue, so go get the real thing.

    Previously this silently dropped the profit chart. The chart is now always
    drawn, so the duplicate signal has to drive a second look at the note
    instead -- otherwise the note would ship two identical donuts.
    """
    replies = [NOTE_WITH_DUPLICATE_PROFIT, REAL_PROFIT_REPLY]
    seen = []

    def staged(*, messages, system, model, max_tokens, timeout, api_key):
        seen.append(max_tokens)
        return {'text': replies[min(len(seen) - 1, len(replies) - 1)],
                'provider': 'anthropic', 'model': model,
                'usage': {'input_tokens': 0, 'output_tokens': 0}}

    monkeypatch.setattr(app_v3, '_call_anthropic_stream', staged)
    _seed_documents('DUPE', [('annual-report-2025.pdf', 10)])
    job_id = _job('DUPE')
    app_v3._generate_research_note(job_id, 'DUPE', 'test-key', 'new', None)

    assert len(seen) >= 2, 'a mirrored profit split did not trigger a re-extract'
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT charts FROM research_notes WHERE ticker = %s', ('DUPE',))
        charts = cur.fetchone()['charts']
    if isinstance(charts, str):
        charts = json.loads(charts)
    assert {c['type'] for c in charts} == {'revenue', 'profit'}


def test_both_charts_are_stored_even_when_profit_still_mirrors_revenue(clean_db, monkeypatch):
    """Last resort: a redundant chart beats a missing one.

    If the re-extract cannot find real segment profit either, the note still
    gets both charts and the job says why the profit chart looks familiar. A
    reader can see and question two similar donuts; they cannot question a chart
    that was never drawn.
    """
    def always_dupe(*, messages, system, model, max_tokens, timeout, api_key):
        return {'text': NOTE_WITH_DUPLICATE_PROFIT, 'provider': 'anthropic',
                'model': model, 'usage': {'input_tokens': 0, 'output_tokens': 0}}

    monkeypatch.setattr(app_v3, '_call_anthropic_stream', always_dupe)
    _seed_documents('STUB', [('annual-report-2025.pdf', 10)])
    job_id = _job('STUB')
    app_v3._generate_research_note(job_id, 'STUB', 'test-key', 'new', None)

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT charts, metadata FROM research_notes WHERE ticker = %s', ('STUB',))
        row = cur.fetchone()
    charts = row['charts']
    if isinstance(charts, str):
        charts = json.loads(charts)
    assert {c['type'] for c in charts} == {'revenue', 'profit'}, \
        'a note must carry both charts even when profit mirrors revenue'
    for c in charts:
        assert c.get('data'), f"{c['type']} chart stored with no image"
    meta = row['metadata']
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert 'mirrors revenue' in (meta.get('chartWarning') or ''), \
        f"the duplication was not reported: {meta.get('chartWarning')!r}"


FOUR_SERIES_REPLY = """===CHART_DATA===
{"priorYear": {"label": "FY2025A",
  "revenue": [{"segment": "Health Services", "value": 190000},
              {"segment": "Health Care Benefits", "value": 143000},
              {"segment": "Pharmacy & Consumer Wellness", "value": 139000}],
  "profit":  [{"segment": "Health Services", "value": 7100},
              {"segment": "Health Care Benefits", "value": 2900},
              {"segment": "Pharmacy & Consumer Wellness", "value": 6000}]},
 "currentYear": {"label": "FY2026E",
  "revenue": [{"segment": "Health Services", "value": 196600},
              {"segment": "Health Care Benefits", "value": 148000},
              {"segment": "Pharmacy & Consumer Wellness", "value": 141000}],
  "profit":  [{"segment": "Health Services", "value": 7250},
              {"segment": "Health Care Benefits", "value": 5200},
              {"segment": "Pharmacy & Consumer Wellness", "value": 6400}]}}
===CHART_DATA_END===

===NOTE_START===
# Acme Corporation (ACME)

Segment detail.
===NOTE_END===
"""


def test_a_note_gets_four_annual_charts_labelled_by_year(clean_db, monkeypatch):
    """Revenue and operating profit, prior year actual and current year estimate.

    The first working charts came out of a single quarter's segment mix and were
    titled only "Revenue Breakdown", so the page never said which period it was
    showing. Four series, each carrying its fiscal year, is the deliverable.
    """
    def reply(*, messages, system, model, max_tokens, timeout, api_key):
        return {'text': FOUR_SERIES_REPLY, 'provider': 'anthropic', 'model': model,
                'usage': {'input_tokens': 0, 'output_tokens': 0}}

    monkeypatch.setattr(app_v3, '_call_anthropic_stream', reply)
    _seed_documents('FOUR', [('annual-report-2025.pdf', 10)])
    job_id = _job('FOUR')
    app_v3._generate_research_note(job_id, 'FOUR', 'test-key', 'new', None)

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT charts FROM research_notes WHERE ticker = %s', ('FOUR',))
        charts = cur.fetchone()['charts']
    if isinstance(charts, str):
        charts = json.loads(charts)

    assert len(charts) == 4, f'expected 4 charts, got {len(charts)}: {[c["type"] for c in charts]}'
    assert {(c['kind'], c['period']) for c in charts} == {
        ('revenue', 'FY2025A'), ('profit', 'FY2025A'),
        ('revenue', 'FY2026E'), ('profit', 'FY2026E')}
    for c in charts:
        assert c['label'].startswith(('FY2025A', 'FY2026E')), c['label']
        assert c.get('data'), f"{c['label']} stored with no image"
        raw = base64.b64decode(c['data'])
        assert raw[:8] == b'\x89PNG\r\n\x1a\n', f"{c['label']} is not a PNG"


def test_a_legacy_two_block_response_still_renders(clean_db, stub_llm):
    """Notes written before the four-series contract must keep working.

    The local agent still emits the old REVENUE/PROFIT arrays with no period.
    Those render as an unlabelled pair rather than failing.
    """
    _seed_documents('LEGA', [('annual-report-2025.pdf', 10)])
    job_id = _job('LEGA')
    app_v3._generate_research_note(job_id, 'LEGA', 'test-key', 'new', None)
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT charts FROM research_notes WHERE ticker = %s', ('LEGA',))
        charts = cur.fetchone()['charts']
    if isinstance(charts, str):
        charts = json.loads(charts)
    assert {c['type'] for c in charts} == {'revenue', 'profit'}
    for c in charts:
        assert c.get('data'), 'legacy chart lost its image'
