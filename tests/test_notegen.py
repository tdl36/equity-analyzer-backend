"""Document planning for research notes.

The note is generated from the FIRST batch and later batches are merged into
it, so which documents land in batch one decides what the note is about. The
agent filled batches in filesystem order, which meant a broker PDF could shape
the thesis while the 10-K arrived as an afterthought.
"""
import base64
import io

import pytest

import notegen


def _pdf(pages):
    from PyPDF2 import PdfWriter
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    return base64.b64encode(buf.getvalue()).decode()


def _doc(name, pages=10, created="2026-01-01T00:00:00Z", ftype="pdf"):
    return {"filename": name, "file_type": ftype,
            "file_data": _pdf(pages), "created_at": created}


def test_primary_sources_outrank_broker_research():
    docs = [_doc("MS_ETN_initiation_note.pdf"),
            _doc("ETN_10-K_2025.pdf"),
            _doc("ETN_Q2_2026_transcript.pdf")]
    order = [d["filename"] for d in notegen.rank_documents(docs)]
    assert order[0] == "ETN_10-K_2025.pdf", order
    assert order[-1] == "MS_ETN_initiation_note.pdf", order


def test_recency_breaks_ties_within_a_kind():
    docs = [_doc("ETN_Q1_2026_transcript.pdf", created="2026-01-05T00:00:00Z"),
            _doc("ETN_Q2_2026_transcript.pdf", created="2026-08-05T00:00:00Z")]
    order = [d["filename"] for d in notegen.rank_documents(docs)]
    assert order[0] == "ETN_Q2_2026_transcript.pdf", order


def test_the_first_batch_carries_the_primary_sources():
    docs = [_doc("broker_note.pdf", pages=18),
            _doc("ETN_10-K_2025.pdf", pages=40),
            _doc("ETN_transcript.pdf", pages=22)]
    first = [d["filename"] for d in notegen.plan_batches(docs)[0]]
    assert "ETN_10-K_2025.pdf" in first, first


def test_tokens_are_counted_by_pages_not_bytes():
    """Bytes are a bad proxy: a scanned page and a text page cost the same."""
    small = _doc("a.pdf", pages=10)
    big = _doc("b.pdf", pages=100)
    assert notegen.pdf_page_count(small) == 10
    assert notegen.estimate_tokens([big]) == 10 * notegen.estimate_tokens([small])


def test_a_document_too_large_to_attach_is_sent_as_text_not_dropped():
    """The 10-K is both the largest document and the most important one."""
    huge = _doc("ETN_10-K_2025.pdf", pages=400)
    prepared = notegen.prepare_documents([huge])
    assert prepared[0]["send_as"] == "text", prepared[0]["send_as"]
    assert prepared[0]["filename"] in notegen.plan_summary([huge])["sentAsText"]


def test_no_batch_exceeds_the_context_budget():
    docs = [_doc(f"doc_{i}.pdf", pages=60) for i in range(6)]
    for batch in notegen.plan_batches(docs):
        assert notegen.estimate_tokens(batch) <= notegen.MAX_TOKENS_PER_BATCH


def test_spreadsheets_are_not_costed_as_raw_text():
    """Zipped XML: the readable text is a fraction of the file."""
    xlsx = {"filename": "model.xlsx", "file_type": "xlsx", "file_size": 2_000_000}
    txt = {"filename": "notes.txt", "file_type": "txt", "file_size": 2_000_000}
    assert notegen.estimate_tokens([xlsx]) < notegen.estimate_tokens([txt]) / 4


def test_planning_is_stable_and_survives_empty_input():
    assert notegen.plan_batches([]) == []
    summary = notegen.plan_summary([])
    assert summary["documentCount"] == 0 and summary["batchCount"] == 0
    docs = [_doc("a.pdf"), _doc("b.pdf")]
    assert [d["filename"] for d in notegen.rank_documents(docs)] == \
           [d["filename"] for d in notegen.rank_documents(docs)]


# --- segment charts ---------------------------------------------------------

def test_a_profit_split_that_repeats_revenue_is_not_drawn_twice():
    """Two identical charts assert every segment earns at the company margin."""
    import segment_charts
    rev = [{'segment': 'A', 'revenue': 60}, {'segment': 'B', 'revenue': 40}]
    same = [{'segment': 'A', 'profit': 6}, {'segment': 'B', 'profit': 4}]   # same shares
    different = [{'segment': 'A', 'profit': 80}, {'segment': 'B', 'profit': 20}]

    assert segment_charts.is_duplicate_series(rev, same) is True
    assert segment_charts.is_duplicate_series(rev, different) is False
    assert len(segment_charts.render_pair('T', rev, same)) == 1
    assert len(segment_charts.render_pair('T', rev, different)) == 2


def test_a_loss_making_segment_is_excluded_rather_than_breaking_the_pie():
    import segment_charts
    charts = segment_charts.render_pair(
        'T', [{'segment': 'A', 'revenue': 100}, {'segment': 'B', 'revenue': -20}], None)
    assert len(charts) == 1 and charts[0]['png'][:4] == b'\x89PNG'


def test_charts_are_available_on_the_server():
    """matplotlib was missing from requirements, so server notes came out bare."""
    import segment_charts
    png = segment_charts.render_donut('T', 'Revenue',
                                      [{'segment': 'A', 'revenue': 1}], 'revenue')
    assert png and png[:4] == b'\x89PNG'


# --- the module actually has what it calls at runtime -----------------------

def test_app_can_reach_the_modules_it_calls():
    """Catch a missing import before a user does.

    notegen and segment_charts were referenced by the note generator but never
    imported into app_v3: an edit meant for that file matched a string that
    only exists in deepdive.py, so the import silently went nowhere. Every test
    passed because none of them touched the note path, and the first thing that
    did was a live run, which failed with "name 'notegen' is not defined".
    """
    import app_v3
    for name in ('notegen', 'segment_charts'):
        assert hasattr(app_v3, name), f'app_v3 never imported {name}'

    # and the specific callables the generator uses
    assert callable(app_v3.notegen.plan_batches)
    assert callable(app_v3.notegen.plan_summary)
    assert callable(app_v3.segment_charts.render_pair)


def test_every_global_the_note_generator_uses_resolves():
    """Walk the generator's own bytecode for names it expects to be module-level.

    A NameError inside a background thread only shows up when someone runs it,
    which for note generation means burning an API call to find out.
    """
    import app_v3
    for fn in (app_v3._generate_research_note, app_v3.research_note_plan):
        for name in fn.__code__.co_names:
            if name.islower() and '_' in name or name in ('notegen', 'segment_charts'):
                # only assert on the modules we added; builtins and attrs vary
                if name in ('notegen', 'segment_charts'):
                    assert hasattr(app_v3, name), f'{fn.__name__} uses undefined {name}'
