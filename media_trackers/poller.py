"""RSS feed poller.

poll_feed(feed_id) — fetches one feed, upserts episodes, updates feed metadata.
poll_all_feeds()   — iterates non-muted feeds due for polling.
backfill_feed(id)  — first-time add helper; poll once ignoring backoff window.
"""
import re
import uuid
import time
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import app_v3


_STYLESHEET_PI_RE = re.compile(rb'<\?xml-stylesheet[^?]*\?>', re.DOTALL)


def _parse_feed_robust(body: bytes):
    """Wrap feedparser.parse with a retry that strips xml-stylesheet PIs.
    Some publishers (e.g. Market Huddle via Seriously Simple Podcasting
    plugin) emit a stylesheet PI on the same line as the XML declaration
    with no whitespace before <rss>, which feedparser's strict parser
    rejects as not-well-formed even though the feed parses fine otherwise.
    If the first parse comes back bozo with zero entries, strip the PI
    and try again.
    """
    parsed = feedparser.parse(body)
    if parsed.bozo and not parsed.entries and _STYLESHEET_PI_RE.search(body):
        cleaned = _STYLESHEET_PI_RE.sub(b'', body, count=1)
        retry = feedparser.parse(cleaned)
        if retry.entries:
            return retry
    return parsed


MAX_ERRORS_BEFORE_MUTE = 5
BACKFILL_DAYS = 7
HTTP_TIMEOUT_SEC = 30


def _parse_published(entry):
    for attr in ('published', 'updated'):
        v = entry.get(attr)
        if v:
            try:
                return parsedate_to_datetime(v).astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                pass
    return None


def _audio_url(entry):
    for enc in entry.get('enclosures') or []:
        if enc.get('type', '').startswith('audio'):
            return enc.get('href') or enc.get('url')
    return None


def _duration_sec(entry):
    v = entry.get('itunes_duration')
    if not v: return None
    try:
        parts = str(v).split(':')
        if len(parts) == 1: return int(parts[0])
        if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def _record_error(feed_id, message):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            UPDATE media_feeds
               SET error_count = error_count + 1,
                   last_error  = %s,
                   last_polled_at = NOW(),
                   muted = CASE WHEN error_count + 1 >= %s THEN TRUE ELSE muted END
             WHERE id = %s
        ''', (message, MAX_ERRORS_BEFORE_MUTE, feed_id))


def _reset_error(feed_id, max_pub):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            UPDATE media_feeds
               SET error_count = 0, last_error = NULL,
                   last_polled_at = NOW(),
                   last_episode_at = GREATEST(COALESCE(last_episode_at, '1970-01-01'::timestamp), %s)
             WHERE id = %s
        ''', (max_pub, feed_id))


def poll_feed(feed_id: str) -> None:
    with app_v3.get_db() as (_c, cur):
        cur.execute("SELECT * FROM media_feeds WHERE id=%s", (feed_id,))
        feed = cur.fetchone()
    if not feed:
        return

    try:
        resp = requests.get(feed['feed_url'], timeout=HTTP_TIMEOUT_SEC,
                            headers={'User-Agent': 'Charlie/1.0 (+https://charlie.tonydlee.com)'})
        resp.raise_for_status()
    except Exception as e:
        _record_error(feed_id, f"HTTP: {e}")
        return

    parsed = _parse_feed_robust(resp.content)
    if parsed.bozo and not parsed.entries:
        _record_error(feed_id, f"parse: {parsed.bozo_exception}")
        return

    # Backfill cutoff: new feed → 7 days ago; existing → last_episode_at.
    if feed['last_episode_at'] is None:
        cutoff = datetime.utcnow() - timedelta(days=BACKFILL_DAYS)
    else:
        cutoff = feed['last_episode_at']

    max_pub = cutoff
    with app_v3.get_db(commit=True) as (_c, cur):
        for entry in parsed.entries:
            guid = entry.get('id') or entry.get('guid') or entry.get('link')
            if not guid:
                continue
            pub = _parse_published(entry)
            if pub and pub <= cutoff:
                continue
            if pub and pub > max_pub:
                max_pub = pub
            cur.execute('''
                INSERT INTO media_episodes
                    (id, feed_id, guid, title, published_at, audio_url, source_url, show_notes, duration_sec, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'new')
                ON CONFLICT (feed_id, guid) DO NOTHING
            ''', (
                str(uuid.uuid4()), feed_id, guid, entry.get('title') or '(untitled)',
                pub, _audio_url(entry), entry.get('link'),
                entry.get('summary') or entry.get('description'),
                _duration_sec(entry),
            ))

    _reset_error(feed_id, max_pub)


def poll_all_feeds() -> None:
    """Iterate non-muted feeds that are due for polling."""
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM media_feeds
             WHERE muted = FALSE
               AND (last_polled_at IS NULL
                    OR last_polled_at < NOW() - (poll_interval_min || ' minutes')::interval)
        ''')
        ids = [r['id'] for r in cur.fetchall()]
    for feed_id in ids:
        try:
            poll_feed(feed_id)
        except Exception as e:
            _record_error(feed_id, f"poll: {e}")
        time.sleep(0.25)  # be nice to publishers


def backfill_feed(feed_id: str) -> None:
    """Force-poll a newly added feed, bypassing the normal backoff window."""
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("UPDATE media_feeds SET last_polled_at = NULL WHERE id = %s", (feed_id,))
    poll_feed(feed_id)
