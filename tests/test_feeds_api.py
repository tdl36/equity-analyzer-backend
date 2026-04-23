"""Tests for /api/media/feeds CRUD endpoints."""


def test_get_feeds_empty(client):
    resp = client.get('/api/media/feeds')
    assert resp.status_code == 200
    assert resp.get_json() == {'feeds': [], 'total': 0}


def test_create_feed(client):
    payload = {
        'name': 'Odd Lots',
        'feedUrl': 'https://feeds.bloomberg.fm/BLM7293307739',
        'sourceType': 'podcast',
        'sectorTags': ['macro'],
    }
    resp = client.post('/api/media/feeds', json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['feed']['name'] == 'Odd Lots'
    assert body['feed']['sectorTags'] == ['macro']
    assert body['feed']['muted'] is False


def test_create_feed_requires_name_and_url(client):
    resp = client.post('/api/media/feeds', json={'name': 'X'})
    assert resp.status_code == 400


def test_patch_feed_toggles_mute(client):
    resp = client.post('/api/media/feeds', json={
        'name': 'Odd Lots', 'feedUrl': 'https://example.com/rss', 'sourceType': 'podcast',
    })
    feed_id = resp.get_json()['feed']['id']
    resp = client.patch(f'/api/media/feeds/{feed_id}', json={'muted': True})
    assert resp.status_code == 200
    assert resp.get_json()['feed']['muted'] is True


def test_delete_feed(client):
    resp = client.post('/api/media/feeds', json={
        'name': 'Odd Lots', 'feedUrl': 'https://example.com/rss', 'sourceType': 'podcast',
    })
    feed_id = resp.get_json()['feed']['id']
    resp = client.delete(f'/api/media/feeds/{feed_id}')
    assert resp.status_code == 200
    resp = client.get('/api/media/feeds')
    assert resp.get_json()['total'] == 0
