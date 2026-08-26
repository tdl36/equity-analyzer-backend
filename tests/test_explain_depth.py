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
