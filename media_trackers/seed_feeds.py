"""Seed the media_feeds table with Tony's default 25 podcasts.

Idempotent via ON CONFLICT DO NOTHING keyed on feed_url.
Run: python -m media_trackers.seed_feeds
"""
import uuid
import app_v3


SEED_FEEDS = [
    # Macro / general finance
    {'name': 'Odd Lots', 'feed_url': 'https://feeds.bloomberg.fm/BLM7293307739', 'sector_tags': ['macro', 'markets']},
    {'name': 'Bloomberg Surveillance', 'feed_url': 'https://feeds.bloomberg.fm/BLM4689043924', 'sector_tags': ['macro', 'markets']},
    {'name': 'Invest Like the Best', 'feed_url': 'https://feeds.simplecast.com/JGE3yC0V', 'sector_tags': ['investing']},
    {'name': 'Capital Allocators', 'feed_url': 'https://capitalallocators.libsyn.com/rss', 'sector_tags': ['investing']},
    {'name': "Grant's Current Yield Podcast", 'feed_url': 'https://feeds.megaphone.fm/grantscurrentyield', 'sector_tags': ['macro', 'credit']},
    {'name': 'Animal Spirits', 'feed_url': 'https://feeds.megaphone.fm/animalspirits', 'sector_tags': ['investing', 'macro']},
    {'name': 'The Transcript', 'feed_url': 'https://feeds.simplecast.com/UVgAJtOV', 'sector_tags': ['earnings']},
    {'name': 'Market Huddle', 'feed_url': 'https://feeds.megaphone.fm/markethuddle', 'sector_tags': ['macro', 'markets']},
    # Tech / AI / semis
    {'name': 'Acquired', 'feed_url': 'https://feeds.transistor.fm/acquired', 'sector_tags': ['tech', 'investing']},
    {'name': 'Sharp Tech with Ben Thompson', 'feed_url': 'https://sharptech.fm/feed/podcast', 'sector_tags': ['tech']},
    {'name': 'a16z Podcast', 'feed_url': 'https://feeds.simplecast.com/JGE3yC0V_a16z', 'sector_tags': ['tech', 'vc']},
    {'name': 'All-In Podcast', 'feed_url': 'https://allinchamathjason.libsyn.com/rss', 'sector_tags': ['tech', 'macro']},
    {'name': 'BG2 Pod', 'feed_url': 'https://feeds.megaphone.fm/bg2pod', 'sector_tags': ['tech', 'ai']},
    {'name': 'Big Technology Podcast', 'feed_url': 'https://feeds.megaphone.fm/bigtechnology', 'sector_tags': ['tech']},
    {'name': 'No Priors', 'feed_url': 'https://feeds.megaphone.fm/nopriors', 'sector_tags': ['tech', 'ai']},
    {'name': 'Dwarkesh Podcast', 'feed_url': 'https://feeds.transistor.fm/dwarkesh', 'sector_tags': ['tech', 'ai']},
    # Healthcare / biotech
    {'name': 'STAT First Opinion', 'feed_url': 'https://feeds.simplecast.com/5OGUyVKf', 'sector_tags': ['healthcare']},
    {'name': 'Biotech Hangout', 'feed_url': 'https://biotechhangout.libsyn.com/rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'Endpoints Weekly', 'feed_url': 'https://feeds.buzzsprout.com/1791687.rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'The Long Run', 'feed_url': 'https://timmermanreport.libsyn.com/rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'MedTech Talk', 'feed_url': 'https://feeds.buzzsprout.com/1955770.rss', 'sector_tags': ['healthcare', 'medtech']},
    # Sector / specialty
    {'name': 'Business Breakdowns', 'feed_url': 'https://feeds.simplecast.com/IZe51ENa', 'sector_tags': ['investing', 'sector']},
    {'name': 'Value Hive', 'feed_url': 'https://feeds.simplecast.com/ynrv7Y_H', 'sector_tags': ['investing']},
    {'name': 'Dealcast', 'feed_url': 'https://feeds.megaphone.fm/dealcast', 'sector_tags': ['banking', 'ma']},
    {'name': 'Hard Fork', 'feed_url': 'https://feeds.simplecast.com/l2i9YnTd', 'sector_tags': ['tech']},
]


def seed():
    """Insert default feeds. Idempotent: uses ON CONFLICT on feed_url.

    Note: feed_url uniqueness is enforced by a unique index created here
    on first run (not in init_db — optional seed-only constraint).
    """
    with app_v3.get_db(commit=True) as (_conn, cur):
        cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_media_feeds_url_unique ON media_feeds(feed_url)')
        for f in SEED_FEEDS:
            cur.execute('''
                INSERT INTO media_feeds (id, source_type, name, feed_url, sector_tags, poll_interval_min)
                VALUES (%s, 'podcast', %s, %s, %s, 30)
                ON CONFLICT (feed_url) DO NOTHING
            ''', (str(uuid.uuid4()), f['name'], f['feed_url'], f['sector_tags']))


if __name__ == '__main__':
    seed()
    print(f"Seeded up to {len(SEED_FEEDS)} feeds.")
