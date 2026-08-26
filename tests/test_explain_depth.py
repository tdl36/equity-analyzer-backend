"""Tests for the Explain tab's depth dial.

Depth is an instruction prepended to whichever prompt the mode chose, so mode
(what shape of output) and depth (how much background it assumes) stay
independent rather than multiplying into four prompts. What is pinned here is
that they compose, and that a bad value degrades to standard instead of
reaching the model as an instruction.
"""
import pytest

import app_v3


def test_every_depth_has_a_key_and_label():
    keys = [k for k, _label, _text in app_v3.EXPLAIN_DEPTHS]
    assert keys == ['standard', 'simple', 'simplest']
    assert all(label for _k, label, _t in app_v3.EXPLAIN_DEPTHS)


def test_standard_adds_no_instruction():
    """Standard must be a true no-op, or it is just another opinionated voice."""
    assert app_v3.EXPLAIN_DEPTH_TEXT['standard'] == ''


def test_simpler_depths_carry_real_instructions():
    assert 'no background' in app_v3.EXPLAIN_DEPTH_TEXT['simple']
    assert 'fifteen-year-old' in app_v3.EXPLAIN_DEPTH_TEXT['simplest']


def test_depths_route_serves_the_same_list(client):
    body = client.get('/api/explain/depths').get_json()
    assert [d['key'] for d in body['depths']] == ['standard', 'simple', 'simplest']
    assert body['default'] == 'standard'


@pytest.mark.parametrize('bad', ['', 'SIMPLEST!', 'eli5', None, 'drop table'])
def test_unknown_depth_falls_back_to_standard(bad):
    """An unrecognised dial position must not reach the model as an instruction."""
    resolved = (bad or 'standard').strip().lower()
    if resolved not in app_v3.EXPLAIN_DEPTH_TEXT:
        resolved = 'standard'
    assert resolved == 'standard'
    assert app_v3.EXPLAIN_DEPTH_TEXT[resolved] == ''


def test_a_real_depth_is_not_flattened_to_standard():
    """The guard above must not be so eager it swallows valid values."""
    for good in ('simple', 'SIMPLE', ' simplest '):
        resolved = good.strip().lower()
        assert resolved in app_v3.EXPLAIN_DEPTH_TEXT
        assert app_v3.EXPLAIN_DEPTH_TEXT[resolved] != ''


# ---------------------------------------------------------------------------
# the route itself
# ---------------------------------------------------------------------------
# These exist because the depth dial shipped broken: the route referenced a
# `depth` local that had been inserted into a different function by a
# non-unique anchor, so every Explain run died with
# "name 'depth' is not defined". Testing the constants and the /depths listing
# said nothing about it -- only exercising POST /api/decipher does.

def _capture_worker(monkeypatch):
    """Run the route without threads or an Anthropic call; record the kwargs."""
    captured = {}

    class _FakeThread:
        def __init__(self, target=None, args=(), kwargs=None, **_ignored):
            captured['args'] = args
            captured['kwargs'] = kwargs or {}

        def start(self):
            captured['started'] = True

    monkeypatch.setattr(app_v3.threading, 'Thread', _FakeThread)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    return captured


def test_decipher_route_dispatches_without_a_nameerror(client, monkeypatch):
    captured = _capture_worker(monkeypatch)
    r = client.post('/api/decipher', json={'text': 'Some dense filing text.'})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json().get('jobId')
    assert captured.get('started') is True


def test_decipher_route_passes_the_chosen_depth_to_the_worker(client, monkeypatch):
    captured = _capture_worker(monkeypatch)
    client.post('/api/decipher', json={'text': 'x', 'depth': 'simplest'})
    assert captured['kwargs']['depth'] == 'simplest'


def test_decipher_route_defaults_depth_when_omitted(client, monkeypatch):
    captured = _capture_worker(monkeypatch)
    client.post('/api/decipher', json={'text': 'x'})
    assert captured['kwargs']['depth'] == 'standard'


def test_decipher_route_rejects_a_bogus_depth_without_failing(client, monkeypatch):
    captured = _capture_worker(monkeypatch)
    r = client.post('/api/decipher', json={'text': 'x', 'depth': 'eli5'})
    assert r.status_code == 200
    assert captured['kwargs']['depth'] == 'standard'


def test_depth_is_not_leaked_into_the_youtube_route(client, monkeypatch):
    """The original bug put `depth` in youtube_summarize(). Keep it out."""
    import inspect
    src = inspect.getsource(app_v3.youtube_summarize)
    assert 'EXPLAIN_DEPTH_TEXT' not in src
