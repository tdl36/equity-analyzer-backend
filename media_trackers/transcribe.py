"""Transcription pipeline for media_episodes.

Lifecycle: new -> transcribing -> extracting -> done | skipped | failed.

For each episode in status='new':
  1. If source_url matches a known publisher, try scraping the transcript page.
  2. Else (or on failure), if audio_url + duration <= 2h, download and call
     Gemini 2.0 Flash audio transcribe.
  3. Else mark transcript_source='show_notes_only' and rely on the extractor
     to use the show_notes column.
Always flip status to 'extracting' on success so the existing extract worker
picks up the episode on its next tick.
"""
import io
import os

import requests

import app_v3
from media_trackers.scrapers import try_publisher_scrape


MAX_EPISODES_PER_BATCH = 4
MAX_PARALLEL_TRANSCRIBE = 4        # threads per batch (mostly I/O wait on Gemini)
MAX_DURATION_SEC = 14400           # 4 hour cap on Gemini fallback
MAX_AUDIO_BYTES = 150 * 1024 * 1024  # 150 MB download cap
DOWNLOAD_TIMEOUT_SEC = 120

# Conservative per-second cost for gemini-2.5-flash audio input, rounded to
# 4 decimals on insert. The spec says use a safe conservative estimate.
COST_PER_AUDIO_SEC = 0.0000001


TRANSCRIBE_PROMPT = (
    "Transcribe this podcast audio verbatim into plain text. "
    "Preserve speaker turns if clearly distinguishable (e.g. 'Host:', 'Guest:') "
    "but otherwise just output clean readable prose. Do not summarize."
)


def _set_failed(episode_id: str, msg: str) -> None:
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "UPDATE media_episodes SET status='failed', error_message=%s WHERE id=%s",
            (msg[:500], episode_id))


def _set_skipped(episode_id: str, msg: str) -> None:
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "UPDATE media_episodes SET status='skipped', error_message=%s WHERE id=%s",
            (msg[:500], episode_id))


def _save_transcript(episode_id: str, transcript: str | None, source: str, cost_usd: float) -> None:
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            UPDATE media_episodes
               SET transcript = COALESCE(%s, transcript),
                   transcript_source = %s,
                   cost_usd = %s,
                   status = 'extracting'
             WHERE id = %s
        ''', (transcript, source, round(cost_usd, 4), episode_id))


def _gemini_transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    """Call Gemini 2.0 Flash audio input. Returns transcript text or raises."""
    api_key = os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
    if not api_key:
        # Fall back to app_settings-stored key, like other call sites do.
        try:
            with app_v3.get_db() as (_c, cur):
                cur.execute("SELECT value FROM app_settings WHERE key = 'geminiApiKey'")
                row = cur.fetchone()
            if row:
                api_key = row['value']
        except Exception:
            pass
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY not configured')

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    size_mb = len(audio_bytes) / (1024 * 1024)
    uploaded = None
    try:
        if size_mb > 20:
            uploaded = client.files.upload(
                file=io.BytesIO(audio_bytes),
                config=genai_types.UploadFileConfig(mime_type=mime_type, display_name='episode_audio')
            )
            # Wait for processing
            import time as _time
            wait_start = _time.time()
            def _is_active(s):
                # Gemini SDK has shipped multiple state representations:
                # 'ACTIVE', 'State.ACTIVE', 'FileState.ACTIVE', '2', and the
                # enum itself (with .name == 'ACTIVE'). Accept any of them.
                if s is None:
                    return False
                name = getattr(s, 'name', None)
                if isinstance(name, str) and name.upper() == 'ACTIVE':
                    return True
                return str(s).rsplit('.', 1)[-1].upper() == 'ACTIVE' or str(s) == '2'
            while hasattr(uploaded, 'state') and not _is_active(uploaded.state):
                if _time.time() - wait_start > 600:
                    raise RuntimeError(f'Gemini file processing timed out after 600s (size={size_mb:.1f}MB, last state={uploaded.state})')
                _time.sleep(5)
                uploaded = client.files.get(name=uploaded.name)
            audio_content = uploaded
        else:
            audio_content = genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[audio_content, TRANSCRIBE_PROMPT],
            config=genai_types.GenerateContentConfig(max_output_tokens=65536),
        )
        try:
            text = response.text
        except Exception:
            text = None
        if not text:
            # Attempt manual extraction from candidates
            if hasattr(response, 'candidates') and response.candidates:
                for c in response.candidates:
                    if hasattr(c, 'content') and c.content and hasattr(c.content, 'parts'):
                        for p in c.content.parts:
                            if hasattr(p, 'text') and p.text:
                                text = p.text
                                break
                    if text:
                        break
        if not text or not text.strip():
            raise RuntimeError('Gemini returned empty transcript')
        return text
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def _download_audio(audio_url: str) -> tuple[bytes, str]:
    """Stream-download audio, guarding against oversized files. Returns (bytes, mime_type)."""
    resp = requests.get(audio_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC,
                        headers={'User-Agent': 'Charlie/1.0'})
    resp.raise_for_status()
    mime_type = resp.headers.get('Content-Type', 'audio/mpeg').split(';')[0].strip() or 'audio/mpeg'
    # Some hosts (Dwarkesh's Cloudflare worker, others) serve audio as
    # binary/octet-stream which Gemini rejects. Fall back to the URL's
    # extension or audio/mpeg.
    if mime_type in ('binary/octet-stream', 'application/octet-stream', 'application/binary'):
        url_lower = audio_url.lower().split('?')[0]
        if   url_lower.endswith('.mp3'): mime_type = 'audio/mpeg'
        elif url_lower.endswith('.m4a'): mime_type = 'audio/mp4'
        elif url_lower.endswith('.aac'): mime_type = 'audio/aac'
        elif url_lower.endswith('.ogg') or url_lower.endswith('.oga'): mime_type = 'audio/ogg'
        elif url_lower.endswith('.wav'): mime_type = 'audio/wav'
        elif url_lower.endswith('.flac'): mime_type = 'audio/flac'
        else: mime_type = 'audio/mpeg'

    content_length = resp.headers.get('Content-Length')
    if content_length:
        try:
            if int(content_length) > MAX_AUDIO_BYTES:
                raise RuntimeError(f'audio too large: {content_length} bytes')
        except ValueError:
            pass

    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 16):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_AUDIO_BYTES:
            raise RuntimeError('audio exceeded 150MB cap')
        chunks.append(chunk)
    return b''.join(chunks), mime_type


def transcribe_episode(episode_id: str) -> None:
    """Claim a 'new' episode, run transcript acquisition, flip status to 'extracting'."""
    # Atomic claim — only one worker can transition new -> transcribing.
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute(
            "UPDATE media_episodes SET status='transcribing' WHERE id=%s AND status='new' RETURNING *",
            (episode_id,))
        ep = cur.fetchone()
    if not ep:
        return

    try:
        source_url = ep.get('source_url')
        audio_url = ep.get('audio_url')
        duration_sec = ep.get('duration_sec')

        # 1) Try publisher scrape
        if source_url:
            text, src = try_publisher_scrape(source_url, audio_url)
            if text and src:
                _save_transcript(episode_id, text, src, 0.0)
                return

        # 2) Gemini fallback — requires audio and <=2h duration
        if audio_url:
            if duration_sec is not None and duration_sec > MAX_DURATION_SEC:
                _set_skipped(episode_id, f'duration {duration_sec}s exceeds {MAX_DURATION_SEC}s cap')
                return
            try:
                audio_bytes, mime_type = _download_audio(audio_url)
            except Exception as e:
                # Audio download failed — fall through to show_notes_only rather than failing hard.
                _save_transcript(episode_id, None, 'show_notes_only', 0.0)
                return
            try:
                transcript = _gemini_transcribe_audio(audio_bytes, mime_type)
            except Exception as e:
                _set_failed(episode_id, f'gemini: {e}')
                return
            cost = (duration_sec or 0) * COST_PER_AUDIO_SEC
            _save_transcript(episode_id, transcript, 'gemini', cost)
            return

        # 3) No source_url transcript, no audio — let extractor use show_notes.
        _save_transcript(episode_id, None, 'show_notes_only', 0.0)
    except Exception as e:
        _set_failed(episode_id, str(e))


def process_transcribe_batch() -> None:
    """Pick up to MAX_EPISODES_PER_BATCH 'new' episodes and transcribe each."""
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT id FROM media_episodes
             WHERE status = 'new'
             ORDER BY published_at DESC NULLS LAST
             LIMIT %s
        ''', (MAX_EPISODES_PER_BATCH,))
        ids = [r['id'] for r in cur.fetchall()]
    if not ids:
        return
    # Transcription is mostly waiting on Gemini upload+processing+generate;
    # ThreadPoolExecutor lets us hold MAX_PARALLEL_TRANSCRIBE in flight at
    # once. Each thread holds one episode's audio bytes (10-150MB), so cap
    # parallelism rather than spawning unbounded.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _safe(eid):
        try:
            transcribe_episode(eid)
        except Exception as e:
            try: _set_failed(eid, str(e))
            except Exception: pass
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRANSCRIBE) as ex:
        futures = [ex.submit(_safe, eid) for eid in ids]
        for _ in as_completed(futures):
            pass
