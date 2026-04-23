"""Tests for /api/media/watchlist CRUD endpoints."""


def test_watchlist_empty(client):
    resp = client.get('/api/media/watchlist')
    assert resp.status_code == 200
    assert resp.get_json() == {'signals': [], 'total': 0}


def test_watchlist_create_ticker(client):
    resp = client.post('/api/media/watchlist', json={
        'kind': 'ticker', 'value': 'ISRG', 'associatedTicker': 'BSX', 'note': 'Competitor',
    })
    assert resp.status_code == 200
    sig = resp.get_json()['signal']
    assert sig['kind'] == 'ticker'
    assert sig['value'] == 'ISRG'
    assert sig['associatedTicker'] == 'BSX'


def test_watchlist_duplicate_returns_409(client):
    client.post('/api/media/watchlist', json={'kind': 'keyword', 'value': 'GLP-1'})
    resp = client.post('/api/media/watchlist', json={'kind': 'keyword', 'value': 'GLP-1'})
    assert resp.status_code == 409


def test_watchlist_mute(client):
    r = client.post('/api/media/watchlist', json={'kind': 'exec', 'value': 'Jensen Huang'})
    sid = r.get_json()['signal']['id']
    resp = client.patch(f'/api/media/watchlist/{sid}', json={'muted': True})
    assert resp.get_json()['signal']['muted'] is True
