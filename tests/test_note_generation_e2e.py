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
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return {'text': MODEL_REPLY, 'provider': 'stub', 'model': 'stub-1',
                'usage': {}}

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
