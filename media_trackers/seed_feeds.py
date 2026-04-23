"""Seed the media_feeds table with Tony's default 22 podcasts.

Idempotent via ON CONFLICT DO NOTHING keyed on feed_url.
Run: python -m media_trackers.seed_feeds

All URLs verified via iTunes Podcast Search API + HEAD check on 2026-04-23.
Three originally-planned feeds (a16z Podcast, Endpoints Weekly, The Transcript)
dropped due to ambiguous matches — Tony can add them via the Settings UI once
it ships.
"""
import uuid
import app_v3


SEED_FEEDS = [
    # Macro / general finance
    {'name': 'Odd Lots', 'feed_url': 'https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/982f5071-765c-403d-969d-ae27003a8d83/podcast.rss', 'sector_tags': ['macro', 'markets']},
    {'name': 'Bloomberg Surveillance', 'feed_url': 'https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8e704079-ca57-4eac-9741-ae27003e2b7f/9739700c-72c3-4176-ae55-ae27003e2b96/podcast.rss', 'sector_tags': ['macro', 'markets']},
    {'name': 'Invest Like the Best', 'feed_url': 'https://feeds.simplecast.com/JGE3yC0V', 'sector_tags': ['investing']},
    {'name': 'Capital Allocators', 'feed_url': 'https://rss.libsyn.com/shows/94820/destinations/482814.xml', 'sector_tags': ['investing']},
    {'name': "Grant's Current Yield Podcast", 'feed_url': 'https://rss.libsyn.com/shows/98027/destinations/509399.xml', 'sector_tags': ['macro', 'credit']},
    {'name': 'Animal Spirits', 'feed_url': 'https://feeds.megaphone.fm/TCP6464651487', 'sector_tags': ['investing', 'macro']},
    {'name': 'Market Huddle', 'feed_url': 'https://markethuddle.com/feed/podcast/', 'sector_tags': ['macro', 'markets']},
    # Tech / AI / semis
    {'name': 'Acquired', 'feed_url': 'https://feeds.transistor.fm/acquired', 'sector_tags': ['tech', 'investing']},
    {'name': 'Sharp Tech with Ben Thompson', 'feed_url': 'https://sharptech.fm/feed/podcast', 'sector_tags': ['tech']},
    {'name': 'All-In Podcast', 'feed_url': 'https://allinchamathjason.libsyn.com/rss', 'sector_tags': ['tech', 'macro']},
    {'name': 'BG2 Pod', 'feed_url': 'https://anchor.fm/s/f06c2370/podcast/rss', 'sector_tags': ['tech', 'ai']},
    {'name': 'Big Technology Podcast', 'feed_url': 'https://feeds.megaphone.fm/LI3617121267', 'sector_tags': ['tech']},
    {'name': 'No Priors', 'feed_url': 'https://feeds.megaphone.fm/nopriors', 'sector_tags': ['tech', 'ai']},
    {'name': 'Dwarkesh Podcast', 'feed_url': 'https://apple.dwarkesh-podcast.workers.dev/feed.rss', 'sector_tags': ['tech', 'ai']},
    # Healthcare / biotech
    {'name': 'STAT First Opinion', 'feed_url': 'https://feeds.megaphone.fm/TTSAA7504675985', 'sector_tags': ['healthcare']},
    {'name': 'Biotech Hangout', 'feed_url': 'https://anchor.fm/s/55bdff38/podcast/rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'The Long Run', 'feed_url': 'https://feeds.soundcloud.com/users/soundcloud:users:317770704/sounds.rss', 'sector_tags': ['healthcare', 'biotech']},
    {'name': 'MedTech Talk', 'feed_url': 'https://feeds.buzzsprout.com/1955770.rss', 'sector_tags': ['healthcare', 'medtech']},
    # Sector / specialty
    {'name': 'Business Breakdowns', 'feed_url': 'https://feeds.megaphone.fm/breakdowns', 'sector_tags': ['investing', 'sector']},
    {'name': 'Value Hive', 'feed_url': 'https://anchor.fm/s/11a83958/podcast/rss', 'sector_tags': ['investing']},
    {'name': 'Dealcast', 'feed_url': 'https://fast.wistia.com/channels/akudzqwqhj/rss', 'sector_tags': ['banking', 'ma']},
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
