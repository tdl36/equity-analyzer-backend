"""Route-level tests for one-pager history and the orchestrator's result merge.

These cover the two things that could not be exercised with fakes: that a saved
page's earlier versions are reachable and restorable, and that two agents
writing progress at the same time do not erase each other's fields.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import app_v3
import onepager


def _page(headline):
    return {'company': 'CVS Health', 'ticker': 'CVS', 'headline': headline}


def _save(ticker, headline, depth='standard'):
    onepager.save_onepager(ticker, _page(headline), app_v3.get_db, depth=depth)


@pytest.fixture
def clean_onepagers():
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('DELETE FROM stock_onepagers')
    yield
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('DELETE FROM stock_onepagers')


# ---------------------------------------------------------------------------
# history / restore
# ---------------------------------------------------------------------------

def test_history_is_empty_for_a_page_generated_once(client, clean_onepagers):
    _save('CVS', 'first')
    r = client.get('/api/onepager/CVS/history?depth=standard')
    assert r.status_code == 200
    assert r.get_json()['versions'] == []


def test_history_lists_previous_versions_newest_first(client, clean_onepagers):
    _save('CVS', 'first')
    _save('CVS', 'second')
    _save('CVS', 'third')
    versions = client.get('/api/onepager/CVS/history?depth=standard').get_json()['versions']
    # Two prior versions; the live one is not history.
    assert [v['index'] for v in versions] == [1, 0]


def test_history_of_an_unknown_ticker_is_empty_not_an_error(client, clean_onepagers):
    r = client.get('/api/onepager/ZZZZ/history')
    assert r.status_code == 200 and r.get_json()['versions'] == []


def test_restore_brings_back_an_earlier_page(client, clean_onepagers):
    _save('CVS', 'first')
    _save('CVS', 'second')
    r = client.post('/api/onepager/CVS/restore',
                    json={'depth': 'standard', 'index': 0})
    assert r.status_code == 200
    assert r.get_json()['onepager']['headline'] == 'first'

    live = onepager.load_onepager('CVS', app_v3.get_db, depth='standard')
    assert live['headline'] == 'first'


def test_restore_is_itself_undoable(client, clean_onepagers):
    """The page being replaced must land in history, or restore is a one-way door."""
    _save('CVS', 'first')
    _save('CVS', 'second')
    client.post('/api/onepager/CVS/restore', json={'depth': 'standard', 'index': 0})

    # 'second' was live at the moment of the restore, so it must now be in
    # history -- otherwise restoring the wrong version destroys the good one.
    r = client.get('/api/onepager/CVS/history?depth=standard')
    stored = r.get_json()['versions']
    assert len(stored) == 2
    back = client.post('/api/onepager/CVS/restore',
                       json={'depth': 'standard', 'index': max(v['index'] for v in stored)})
    assert back.get_json()['onepager']['headline'] == 'second' 


def test_restore_rejects_an_out_of_range_version(client, clean_onepagers):
    _save('CVS', 'first')
    r = client.post('/api/onepager/CVS/restore', json={'depth': 'standard', 'index': 9})
    assert r.status_code == 404


def test_restore_requires_an_index(client, clean_onepagers):
    _save('CVS', 'first')
    assert client.post('/api/onepager/CVS/restore', json={'depth': 'standard'}).status_code == 400


def test_history_is_kept_per_depth(client, clean_onepagers):
    _save('CVS', 'brief-1', depth='brief')
    _save('CVS', 'brief-2', depth='brief')
    _save('CVS', 'std-1', depth='standard')
    assert len(client.get('/api/onepager/CVS/history?depth=brief').get_json()['versions']) == 1
    assert client.get('/api/onepager/CVS/history?depth=standard').get_json()['versions'] == []


# ---------------------------------------------------------------------------
# concurrent progress writes
# ---------------------------------------------------------------------------

def test_concurrent_progress_updates_do_not_clobber_each_other(clean_db):
    """The fan-out writes from two threads; a read-modify-write lost one."""
    job_id = 'orch-race-1'
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("INSERT INTO mp_jobs (id, stage, status, result) "
                    "VALUES (%s, 'orchestrate', 'running', '{}'::jsonb)", (job_id,))

    # Four writers, well inside the connection pool, over enough keys that a
    # read-modify-write would drop several of them.
    keys = [f'agent{i}' for i in range(24)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda k: app_v3._orchestrate_set(job_id, **{k: 'done'}), keys))

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT result FROM mp_jobs WHERE id = %s', (job_id,))
        result = cur.fetchone()['result']
    if isinstance(result, str):
        result = json.loads(result)

    missing = [k for k in keys if result.get(k) != 'done']
    assert not missing, f'these writes were lost: {missing}'


def test_progress_merge_preserves_untouched_fields(clean_db):
    job_id = 'orch-merge-1'
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("INSERT INTO mp_jobs (id, stage, status, result) VALUES "
                    "(%s, 'orchestrate', 'running', %s::jsonb)",
                    (job_id, json.dumps({'diff': {'a': 1}})))
    app_v3._orchestrate_set(job_id, onepagerStep='rendering')

    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT result FROM mp_jobs WHERE id = %s', (job_id,))
        result = cur.fetchone()['result']
    if isinstance(result, str):
        result = json.loads(result)
    assert result['diff'] == {'a': 1}
    assert result['onepagerStep'] == 'rendering'
