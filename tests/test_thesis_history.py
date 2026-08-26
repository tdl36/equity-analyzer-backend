"""Tests for the thesis change-log.

The point of this module is answering "what did I believe, and what changed my
mind" -- so what is pinned here is that revisions are recorded with real content,
that no-op refreshes do not pollute the timeline, and that a row which keeps
being reworded is detectable as drift rather than left to inference.
"""

import json

import pytest

import app_v3
import thesis_history as th


def _thesis(pillar_body, signpost_body='flat', conclusion='hold'):
    return {
        'thesis': {'pillars': [{'title': 'Margin expansion', 'body': pillar_body}]},
        'signposts': [{'signpost': 'Gross margin', 'current': signpost_body}],
        'threats': [{'title': 'Reimbursement', 'body': 'stable'}],
        'conclusion': conclusion,
    }


@pytest.fixture
def clean_revisions():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('DELETE FROM thesis_revisions')
    yield
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('DELETE FROM thesis_revisions')


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------

def test_first_save_is_recorded_even_though_nothing_changed(clean_db, clean_revisions):
    """A thesis coming into existence is the most important entry in the log."""
    rid = th.record_revision(app_v3.get_db, 'CVS', None, _thesis('a'), 'manual')
    assert rid is not None
    assert len(th.list_revisions(app_v3.get_db, 'CVS')) == 1


def test_a_refresh_that_changes_nothing_is_not_recorded(clean_db, clean_revisions):
    """Otherwise real changes get buried in a timeline of no-ops."""
    same = _thesis('a')
    th.record_revision(app_v3.get_db, 'CVS', None, same, 'manual')
    assert th.record_revision(app_v3.get_db, 'CVS', same, dict(same), 'pipeline') is None
    assert len(th.list_revisions(app_v3.get_db, 'CVS')) == 1


def test_revision_records_what_moved(clean_db, clean_revisions):
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('a'), 'manual')
    th.record_revision(app_v3.get_db, 'CVS', _thesis('a'), _thesis('b'), 'pipeline')
    revs = th.list_revisions(app_v3.get_db, 'CVS')
    assert len(revs) == 2
    assert revs[0]['source'] == 'pipeline'          # newest first
    assert revs[0]['counts'].get('changed') == 1
    assert 'reworded' in revs[0]['summary']


def test_conclusion_rewrite_counts_as_a_change(clean_db, clean_revisions):
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('a'), 'manual')
    rid = th.record_revision(app_v3.get_db, 'CVS', _thesis('a'),
                             _thesis('a', conclusion='sell'), 'manual')
    assert rid is not None
    assert th.get_revision(app_v3.get_db, rid)['counts'].get('conclusion') == 1


def test_snapshot_lets_you_read_back_what_you_believed(clean_db, clean_revisions):
    rid = th.record_revision(app_v3.get_db, 'CVS', None, _thesis('original'), 'manual')
    snap = th.get_revision(app_v3.get_db, rid)['snapshot']
    assert snap['thesis']['pillars'][0]['body'] == 'original'


def test_journalling_never_raises_on_a_broken_database(clean_db, clean_revisions):
    """A failure to journal must not take down the thesis save that triggered it."""
    def exploding_db(*a, **k):
        raise RuntimeError('database on fire')
    assert th.record_revision(exploding_db, 'CVS', None, _thesis('a'), 'manual') is None


def test_revisions_are_capped(clean_db, clean_revisions):
    for i in range(8):
        th.record_revision(app_v3.get_db, 'CVS', _thesis(f'v{i}'), _thesis(f'v{i+1}'),
                           'pipeline', max_revisions=5)
    assert len(th.list_revisions(app_v3.get_db, 'CVS', limit=100)) == 5


def test_revisions_are_scoped_per_ticker(clean_db, clean_revisions):
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('a'), 'manual')
    th.record_revision(app_v3.get_db, 'CI', None, _thesis('a'), 'manual')
    assert len(th.list_revisions(app_v3.get_db, 'CVS')) == 1
    assert len(th.list_revisions(app_v3.get_db, 'CI')) == 1


# ---------------------------------------------------------------------------
# drift -- the reason this module exists
# ---------------------------------------------------------------------------

def test_a_pillar_reworded_repeatedly_is_flagged_as_accumulating(clean_db, clean_revisions):
    """No single edit looks structural; the accumulation is the signal."""
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('v0'), 'manual')
    for i in range(4):
        th.record_revision(app_v3.get_db, 'CVS', _thesis(f'v{i}'), _thesis(f'v{i+1}'), 'pipeline')

    rows = th.drift(app_v3.get_db, 'CVS')
    pillar = next(r for r in rows if r['label'] == 'Margin expansion')
    assert pillar['timesRevised'] == 4
    assert pillar['accumulating'] is True


def test_a_stable_pillar_is_not_flagged(clean_db, clean_revisions):
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('v0'), 'manual')
    th.record_revision(app_v3.get_db, 'CVS', _thesis('v0'), _thesis('v1'), 'pipeline')
    rows = th.drift(app_v3.get_db, 'CVS')
    pillar = next(r for r in rows if r['label'] == 'Margin expansion')
    assert pillar['accumulating'] is False


def test_structural_drift_needs_the_classifier_to_have_called_it_structural(clean_db, clean_revisions):
    """The case worth surfacing: judged slow-moving, yet moving every quarter."""
    layers = {'pillars:changed:Margin expansion': {
        'layer': 'structural', 'label': 'Margin expansion', 'section': 'pillars'}}
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('v0'), 'manual')
    for i in range(4):
        th.record_revision(app_v3.get_db, 'CVS', _thesis(f'v{i}'), _thesis(f'v{i+1}'),
                           'orchestrator', layers=layers)

    pillar = next(r for r in th.drift(app_v3.get_db, 'CVS')
                  if r['label'] == 'Margin expansion')
    assert pillar['structuralDrift'] is True
    assert 'structural' in pillar['layers']


def test_drift_of_an_untouched_ticker_is_empty(clean_db, clean_revisions):
    assert th.drift(app_v3.get_db, 'ZZZZ') == []


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

def _store_thesis(ticker, analysis):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("INSERT INTO portfolio_analyses (ticker, company, analysis) "
                    "VALUES (%s, 'X', %s::jsonb) ON CONFLICT (ticker) DO UPDATE "
                    "SET analysis = EXCLUDED.analysis", (ticker, json.dumps(analysis)))


@pytest.fixture
def clean_theses():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM portfolio_analyses WHERE ticker IN ('CVS','CI','ZZZZ')")
    yield
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("DELETE FROM portfolio_analyses WHERE ticker IN ('CVS','CI','ZZZZ')")


def test_revisions_route_returns_timeline_and_drift(client, clean_revisions, clean_theses):
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('a'), 'manual')
    th.record_revision(app_v3.get_db, 'CVS', _thesis('a'), _thesis('b'), 'pipeline')
    body = client.get('/api/thesis/CVS/revisions').get_json()
    assert len(body['revisions']) == 2
    assert isinstance(body['drift'], list)


def test_revision_detail_route_rejects_a_ticker_mismatch(client, clean_revisions, clean_theses):
    """Revision ids are global, so the ticker in the path has to be enforced."""
    rid = th.record_revision(app_v3.get_db, 'CVS', None, _thesis('a'), 'manual')
    assert client.get(f'/api/thesis/CI/revisions/{rid}').status_code == 404
    assert client.get(f'/api/thesis/CVS/revisions/{rid}').status_code == 200


def test_restore_route_puts_the_thesis_back_and_stays_undoable(client, clean_revisions, clean_theses):
    _store_thesis('CVS', _thesis('original'))
    rid = th.record_revision(app_v3.get_db, 'CVS', None, _thesis('original'), 'manual')
    _store_thesis('CVS', _thesis('rewritten'))
    th.record_revision(app_v3.get_db, 'CVS', _thesis('original'), _thesis('rewritten'), 'manual')

    r = client.post(f'/api/thesis/CVS/revisions/{rid}/restore')
    assert r.status_code == 200
    assert app_v3._load_thesis('CVS')['thesis']['pillars'][0]['body'] == 'original'
    # the restore is itself journalled, so it can be walked back
    assert len(th.list_revisions(app_v3.get_db, 'CVS')) == 3


def test_signpost_status_route_is_empty_before_any_check(client, clean_theses):
    _store_thesis('CVS', {'signposts': [{'metric': 'Gross margin', 'target': '45%'}]})
    body = client.get('/api/signposts/CVS').get_json()
    assert body['evaluations'] == [] and body['signpostCount'] == 1


def test_signpost_check_refuses_a_thesis_without_signposts(client, clean_theses):
    _store_thesis('CVS', {'thesis': {'pillars': []}})
    r = client.post('/api/signposts/CVS/check', json={})
    assert r.status_code == 400
    assert 'no signposts' in r.get_json()['error']


def test_signpost_check_refuses_an_unknown_ticker(client, clean_theses):
    assert client.post('/api/signposts/ZZZZ/check', json={}).status_code == 404


def test_signpost_check_says_so_when_there_is_nothing_to_check_against(client, clean_theses):
    """No extracted figures means no material -- and guessing is how alert fatigue starts."""
    _store_thesis('CVS', {'signposts': [{'metric': 'Gross margin', 'target': '45%'}]})
    body = client.post('/api/signposts/CVS/check', json={}).get_json()
    assert body['success'] is False
    assert 'thesis update' in body['message']


def test_drift_names_the_revisions_that_moved_a_row(clean_db, clean_revisions):
    """A count nobody can investigate is not actionable; carry the ids."""
    th.record_revision(app_v3.get_db, 'CVS', None, _thesis('v0'), 'manual')
    for i in range(3):
        th.record_revision(app_v3.get_db, 'CVS', _thesis(f'v{i}'), _thesis(f'v{i+1}'), 'pipeline')

    pillar = next(r for r in th.drift(app_v3.get_db, 'CVS') if r['label'] == 'Margin expansion')
    assert len(pillar['revisionIds']) == pillar['revisions']
    # every id must resolve to a real revision the UI can open
    for rid in pillar['revisionIds']:
        assert th.get_revision(app_v3.get_db, rid) is not None
