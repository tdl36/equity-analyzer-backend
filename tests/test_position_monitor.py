"""The nightly position check.

The scheduler ran twelve jobs and none of them touched a thesis. This is the
thirteenth. Its whole value rests on two properties, and these tests hold both:
it costs nothing to run, and it reports changes rather than state.
"""
import json
import os
import uuid
from datetime import datetime, timedelta

import pytest

os.environ.setdefault('DATABASE_URL', 'postgresql://localhost/charlie_test')
import app_v3
import position_monitor as pm


REVIEW_STATE = {
    'price': 100.0,
    'key_question': 'Does organic growth reach 8%?',
    'scenarios': [{'name': 'bear', 'target_price': 80},
                  {'name': 'base', 'target_price': 115},
                  {'name': 'bull', 'target_price': 140}],
    'kpis': [
        {'name': 'Organic growth', 'current': 7.0, 'prior': 8.0,
         'bull_threshold': 8.0, 'bear_threshold': 7.0},          # off-thesis
        {'name': 'cRPO growth', 'current': 14.0, 'prior': 13.0,
         'bull_threshold': 14.0, 'bear_threshold': 12.0},        # on-thesis
    ],
}


def _review(created_days_ago=1, state=None):
    return {'state': state or REVIEW_STATE,
            'created_at': datetime.utcnow() - timedelta(days=created_days_ago)}


@pytest.fixture
def clean():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM agent_alerts WHERE ticker LIKE 'PM%'")
        cur.execute("DELETE FROM investment_reviews WHERE ticker LIKE 'PM%'")
        cur.execute("DELETE FROM app_settings WHERE key = %s", (pm.MONITOR_STATE_KEY,))
    yield
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM agent_alerts WHERE ticker LIKE 'PM%'")
        cur.execute("DELETE FROM investment_reviews WHERE ticker LIKE 'PM%'")
        cur.execute("DELETE FROM app_settings WHERE key = %s", (pm.MONITOR_STATE_KEY,))


def _seed_review(ticker, state=None, days_ago=1):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM investment_reviews WHERE ticker = %s", (ticker,))
        cur.execute("""INSERT INTO investment_reviews
                       (id, ticker, mode, state, review_markdown, created_at)
                       VALUES (%s,%s,'review',%s,'x', NOW() - INTERVAL '%s days')"""
                    % ('%s', '%s', '%s', days_ago),
                    (str(uuid.uuid4()), ticker, json.dumps(state or REVIEW_STATE)))


# --- the signals -----------------------------------------------------------

def test_a_big_move_since_the_review_is_reported():
    out = pm.evaluate_position('PMA', _review(), price=88.0)   # -12%
    kinds = {f['type'] for f in out}
    assert 'price_move' in kinds
    move = next(f for f in out if f['type'] == 'price_move')
    assert 'down' in move['title'] and '12%' in move['title']


def test_a_small_move_is_not_reported():
    out = pm.evaluate_position('PMA', _review(), price=104.0)  # +4%
    assert 'price_move' not in {f['type'] for f in out}


def test_crossing_the_bear_case_is_the_signal_scenarios_exist_to_give():
    out = pm.evaluate_position('PMA', _review(), price=79.0)
    breach = [f for f in out if f['type'] == 'scenario_breach']
    assert breach and 'bear case' in breach[0]['title']


def test_crossing_the_bull_case_is_reported_too():
    out = pm.evaluate_position('PMA', _review(), price=145.0)
    breach = [f for f in out if f['type'] == 'scenario_breach']
    assert breach and 'bull case' in breach[0]['title']


def test_kpi_status_is_recomputed_from_stored_thresholds():
    """No model, no new data -- status is arithmetic on what is already stored."""
    counts = pm._kpi_status_counts(REVIEW_STATE)
    assert counts['off-thesis'] == 1 and counts['on-thesis'] == 1
    out = pm.evaluate_position('PMA', _review(), price=100.0)
    off = [f for f in out if f['type'] == 'kpi_off_thesis']
    assert off and '1 KPI off-thesis' in off[0]['title']


def test_a_lower_is_better_kpi_is_not_read_backwards():
    """Attrition rising is bad; the monitor inherits that from the KPI model."""
    state = dict(REVIEW_STATE, kpis=[
        {'name': 'Attrition', 'current': 9.5, 'prior': 8.0,
         'bull_threshold': 8.0, 'bear_threshold': 9.0}])
    assert pm._kpi_status_counts(state)['off-thesis'] == 1


def test_a_position_with_nothing_wrong_produces_nothing():
    """Silence is the correct output most nights."""
    quiet = dict(REVIEW_STATE, kpis=[REVIEW_STATE['kpis'][1]])   # only the on-thesis one
    out = pm.evaluate_position('PMA', _review(state=quiet), price=101.0)
    assert out == [], out


# --- transitions, not state ------------------------------------------------

def test_the_same_condition_is_not_reported_twice(clean):
    """A position off-thesis for a month is not news every morning."""
    _seed_review('PMA')
    first = pm.run_position_monitor(['PMA'])
    assert first['alerts'] >= 1, first
    second = pm.run_position_monitor(['PMA'])
    assert second['alerts'] == 0, 'a standing condition was reported again'


def test_a_condition_that_clears_and_returns_is_reported_again(clean):
    _seed_review('PMA')
    pm.run_position_monitor(['PMA'])
    # clear it: no KPIs at all
    _seed_review('PMA', state=dict(REVIEW_STATE, kpis=[]))
    pm.run_position_monitor(['PMA'])
    # and bring it back
    _seed_review('PMA')
    again = pm.run_position_monitor(['PMA'])
    assert again['alerts'] >= 1, 'a condition that returned was suppressed'


def test_a_dry_run_writes_nothing(clean):
    _seed_review('PMB')
    out = pm.run_position_monitor(['PMB'], dry_run=True)
    assert out['findings'], 'dry run found nothing to report'
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM agent_alerts WHERE ticker='PMB'")
        assert cur.fetchone()['n'] == 0
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT COUNT(*) AS n FROM app_settings WHERE key=%s",
                    (pm.MONITOR_STATE_KEY,))
        assert cur.fetchone()['n'] == 0, 'a dry run persisted its state'


def test_a_position_with_no_review_is_skipped_not_failed(clean):
    out = pm.run_position_monitor(['PMNONE'])
    assert out['skipped'] == 1 and out['checked'] == 0


# --- cost ------------------------------------------------------------------

def test_the_monitor_calls_no_model(clean, monkeypatch):
    """The reason it can sweep the whole book nightly.

    A model in this path would cost a review per position per night and mostly
    reproduce yesterday's answer.
    """
    def forbidden(*a, **k):
        raise AssertionError('the position monitor called a model')

    monkeypatch.setattr(app_v3, '_call_pinned_long', forbidden)
    monkeypatch.setattr(app_v3, 'call_llm', forbidden)
    monkeypatch.setattr(app_v3, '_call_anthropic_stream', forbidden)
    _seed_review('PMC')
    pm.run_position_monitor(['PMC'])


def test_the_scheduler_registers_it_and_respects_the_kill_switch():
    src = open(os.path.join(os.path.dirname(__file__), '..', 'scheduler.py')).read()
    assert 'run_positions_nightly' in src
    assert "id='positions_nightly'" in src
    body = src[src.index('def run_positions_nightly'):src.index('def run_theme_scan')]
    assert '_kill_switch_on()' in body, 'the job ignores the kill switch'


# --- what makes a KPI stale ------------------------------------------------

def _add_doc(ticker, name, days_ago=0):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("""INSERT INTO document_files (ticker, filename, file_data,
                       file_type, file_size, created_at)
                       VALUES (%s,%s,'x','pdf',10, NOW() - INTERVAL '%s days')
                       ON CONFLICT (ticker, filename) DO NOTHING"""
                    % ('%s', '%s', days_ago), (ticker, name))


def test_documents_arriving_after_a_review_are_reported(clean):
    """The closest free proxy for "a KPI may have moved".

    A KPI value lives inside a filing; only a model reading that filing can
    update it. A document arriving is the moment that becomes possible, and it
    is already recorded.
    """
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM document_files WHERE ticker = 'PMD'")
    _seed_review('PMD', days_ago=5)
    _add_doc('PMD', '10q-old.pdf', days_ago=9)      # before the review
    _add_doc('PMD', 'broker-new.pdf', days_ago=1)   # after
    _add_doc('PMD', 'transcript-new.pdf', days_ago=0)

    out = pm.evaluate_position('PMD', pm._latest_review('PMD'), price=100.0)
    fresh = [f for f in out if f['type'] == 'sources_since_review']
    assert fresh, 'new documents were not reported'
    assert fresh[0]['detail']['count'] == 2, fresh[0]['detail']
    names = fresh[0]['detail']['filenames']
    assert '10q-old.pdf' not in names, 'a document predating the review was counted'
    assert 'broker-new.pdf' in names

    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM document_files WHERE ticker = 'PMD'")


def test_no_new_documents_produces_no_finding(clean):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM document_files WHERE ticker = 'PME'")
    _seed_review('PME', days_ago=1)
    _add_doc('PME', 'old.pdf', days_ago=5)
    out = pm.evaluate_position('PME', pm._latest_review('PME'), price=100.0)
    assert not [f for f in out if f['type'] == 'sources_since_review']
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM document_files WHERE ticker = 'PME'")


def test_the_kpi_count_says_it_is_as_of_the_last_review():
    """It restates what the review concluded; it is not a fresh reading.

    Labelling it honestly is the difference between a monitor and something
    that looks like one.
    """
    out = pm.evaluate_position('PMA', _review(), price=100.0)
    off = next(f for f in out if f['type'] == 'kpi_off_thesis')
    assert 'as of the last review' in off['title']
    assert 'asOf' in off['detail']


def test_the_module_says_which_signals_are_live():
    """The docstring is the first place someone checks what this can do."""
    import inspect
    # Whitespace-normalised: docstrings wrap, so a phrase can span a newline.
    doc = ' '.join(inspect.getdoc(pm).split())
    assert 'KPI *values* are not' in doc
    assert 'frozen into the review' in doc
