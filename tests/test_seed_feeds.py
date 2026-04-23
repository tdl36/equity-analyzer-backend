"""Verify seed script inserts all feeds idempotently."""
import app_v3
from media_trackers.seed_feeds import seed, SEED_FEEDS


def test_seed_inserts_all_feeds(clean_db):
    seed()
    with app_v3.get_db() as (_conn, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_feeds WHERE source_type='podcast'")
        assert cur.fetchone()['n'] == len(SEED_FEEDS)


def test_seed_is_idempotent(clean_db):
    seed()
    seed()  # second run should not duplicate
    with app_v3.get_db() as (_conn, cur):
        cur.execute("SELECT COUNT(*) AS n FROM media_feeds WHERE source_type='podcast'")
        assert cur.fetchone()['n'] == len(SEED_FEEDS)


def test_all_feeds_have_rss_urls():
    for f in SEED_FEEDS:
        assert f['feed_url'].startswith(('http://', 'https://')), f
        assert f['name'], f
        assert 'sector_tags' in f


def test_no_duplicate_feed_urls():
    urls = [f['feed_url'] for f in SEED_FEEDS]
    assert len(urls) == len(set(urls)), "duplicate feed_url in SEED_FEEDS"
