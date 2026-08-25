"""Tests for signpost monitoring.

The property that matters here is restraint. This raises alerts, and an alerting
system that fires on inference gets muted -- after which the real alert arrives
muted too. So most of these pin down when it refuses to alert.
"""

import json

import pytest

import app_v3
import signposts as sp


SIGNPOSTS = [
    {'signpost': 'Gross margin', 'current': '41%', 'target': '45% by FY27',
     'why': 'the whole margin thesis'},
    {'signpost': 'Retail footprint', 'current': '9,000 stores', 'target': 'below 8,000'},
]


def _llm(payload):
    """A call_llm stand-in returning a fixed payload."""
    def call(**kwargs):
        return {'text': json.dumps(payload)}
    return call


def _extract(text):
    return json.loads(text)


@pytest.fixture
def clean_alerts():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM agent_alerts WHERE alert_type = 'signpost'")
    yield
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM agent_alerts WHERE alert_type = 'signpost'")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_a_hit_signpost_is_reported_with_its_evidence():
    out = sp.evaluate('CVS', SIGNPOSTS, 'material', _llm({'evaluations': [
        {'signpost': 'Gross margin', 'status': 'hit', 'observed': '45.2%',
         'evidence': 'gross margin of 45.2% in the quarter', 'confidence': 'high'},
    ]}), _extract)
    hit = next(e for e in out if e['signpost'] == 'Gross margin')
    assert hit['status'] == 'hit'
    assert hit['observed'] == '45.2%'
    assert hit['target'] == '45% by FY27'


def test_a_status_without_a_quote_is_downgraded_to_no_evidence():
    """A conclusion with nothing to point at is an inference, and must not alert."""
    out = sp.evaluate('CVS', SIGNPOSTS, 'material', _llm({'evaluations': [
        {'signpost': 'Gross margin', 'status': 'broken', 'observed': 'margins fell',
         'evidence': '', 'confidence': 'high'},
    ]}), _extract)
    assert next(e for e in out if e['signpost'] == 'Gross margin')['status'] == 'no_evidence'


def test_a_signpost_the_model_ignored_becomes_no_evidence():
    out = sp.evaluate('CVS', SIGNPOSTS, 'material', _llm({'evaluations': []}), _extract)
    assert {e['status'] for e in out} == {'no_evidence'}
    assert len(out) == 2


def test_an_unrecognised_status_falls_back_to_no_evidence():
    out = sp.evaluate('CVS', SIGNPOSTS, 'material', _llm({'evaluations': [
        {'signpost': 'Gross margin', 'status': 'catastrophic', 'evidence': 'x'},
    ]}), _extract)
    assert next(e for e in out if e['signpost'] == 'Gross margin')['status'] == 'no_evidence'


def test_an_llm_failure_is_silence_not_a_wave_of_alerts():
    def exploding(**kwargs):
        raise RuntimeError('model down')
    assert sp.evaluate('CVS', SIGNPOSTS, 'material', exploding, _extract) == []


def test_a_thesis_with_no_signposts_makes_no_call():
    called = []

    def call(**kwargs):
        called.append(1)
        return {'text': '{}'}
    assert sp.evaluate('CVS', [], 'material', call, _extract) == []
    assert not called


def test_confidence_defaults_to_low_when_unstated():
    out = sp.evaluate('CVS', SIGNPOSTS, 'm', _llm({'evaluations': [
        {'signpost': 'Gross margin', 'status': 'hit', 'evidence': 'q'},
    ]}), _extract)
    assert next(e for e in out if e['signpost'] == 'Gross margin')['confidence'] == 'low'


# ---------------------------------------------------------------------------
# alerting
# ---------------------------------------------------------------------------

def test_only_notable_statuses_raise_alerts(clean_db, clean_alerts):
    evaluations = [
        {'signpost': 'A', 'status': 'hit', 'observed': '45%', 'evidence': 'q'},
        {'signpost': 'B', 'status': 'on_track', 'observed': '', 'evidence': 'q'},
        {'signpost': 'C', 'status': 'no_evidence', 'observed': '', 'evidence': ''},
        {'signpost': 'D', 'status': 'broken', 'observed': 'down', 'evidence': 'q'},
    ]
    created = sp.record_alerts(app_v3.get_db, 'CVS', evaluations)
    assert {c['signpost'] for c in created} == {'A', 'D'}


def test_the_same_reading_does_not_alert_twice(clean_db, clean_alerts):
    evaluations = [{'signpost': 'A', 'status': 'hit', 'observed': '45%', 'evidence': 'q'}]
    assert len(sp.record_alerts(app_v3.get_db, 'CVS', evaluations)) == 1
    assert sp.record_alerts(app_v3.get_db, 'CVS', evaluations) == []


def test_a_signpost_that_moves_again_does_alert_again(clean_db, clean_alerts):
    """Re-checking the same reading is noise; a new reading is news."""
    sp.record_alerts(app_v3.get_db, 'CVS', [
        {'signpost': 'A', 'status': 'hit', 'observed': '45%', 'evidence': 'q'}])
    second = sp.record_alerts(app_v3.get_db, 'CVS', [
        {'signpost': 'A', 'status': 'hit', 'observed': '48%', 'evidence': 'q'}])
    assert len(second) == 1


def test_alerts_are_scoped_per_ticker(clean_db, clean_alerts):
    ev = [{'signpost': 'A', 'status': 'hit', 'observed': '45%', 'evidence': 'q'}]
    assert len(sp.record_alerts(app_v3.get_db, 'CVS', ev)) == 1
    assert len(sp.record_alerts(app_v3.get_db, 'CI', ev)) == 1


def test_alert_carries_the_evidence_into_its_detail(clean_db, clean_alerts):
    sp.record_alerts(app_v3.get_db, 'CVS', [
        {'signpost': 'Gross margin', 'status': 'hit', 'observed': '45.2%',
         'evidence': 'gross margin of 45.2%'}], source='pipeline job 7')
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT detail FROM agent_alerts WHERE alert_type='signpost'")
        detail = cur.fetchone()['detail']
    if isinstance(detail, str):
        detail = json.loads(detail)
    assert detail['evidence'] == 'gross margin of 45.2%'
    assert detail['source'] == 'pipeline job 7'


def test_summarize_counts_notables():
    counts = sp.summarize([
        {'status': 'hit'}, {'status': 'broken'},
        {'status': 'on_track'}, {'status': 'no_evidence'},
    ])
    assert counts['notable'] == 2
    assert counts['on_track'] == 1


def test_the_real_thesis_signpost_shape_is_understood():
    """Thesis signposts are {metric, target, timeframe, category}, not {signpost, current}."""
    rows = [{'metric': 'Gross margin', 'target': '45%', 'timeframe': 'FY27',
             'category': 'Financial'}]
    items, prompt = sp.build_prompt(rows, 'material', 'CVS')
    assert items[0]['signpost'] == 'Gross margin'
    assert items[0]['target'] == '45%'
    assert 'by FY27' in prompt
