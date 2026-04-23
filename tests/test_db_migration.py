"""Verify all media-tracker tables created by init_db()."""
import app_v3

EXPECTED_TABLES = [
    'media_feeds',
    'media_episodes',
    'media_digest_points',
    'signals_watchlist',
    'media_theme_clusters',
    'notification_prefs',
]


def test_media_tables_exist(clean_db):
    with app_v3.get_db() as (_conn, cur):
        cur.execute("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename = ANY(%s)
        """, (EXPECTED_TABLES,))
        found = {r['tablename'] for r in cur.fetchall()}
    missing = set(EXPECTED_TABLES) - found
    assert not missing, f"Missing tables: {missing}"


def test_media_feeds_has_required_columns(clean_db):
    with app_v3.get_db() as (_conn, cur):
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'media_feeds'
        """)
        cols = {r['column_name'] for r in cur.fetchall()}
    required = {'id', 'source_type', 'name', 'feed_url', 'sector_tags',
                'muted', 'last_polled_at', 'last_episode_at',
                'poll_interval_min', 'error_count', 'last_error', 'created_at'}
    assert required <= cols, f"Missing columns: {required - cols}"
