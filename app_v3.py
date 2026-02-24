"""
TDL Equity Analyzer - Backend API with PostgreSQL
Cross-device sync for portfolio analyses and overviews
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import json
import base64
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from datetime import datetime
import anthropic
import openai
from google import genai
from google.genai import types as genai_types

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
CORS(app, origins=[
    "https://equity-analyzer.tonydlee.workers.dev",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
])


# ============================================
# IN-MEMORY CACHE
# ============================================
import time
import threading

class SimpleCache:
    """Thread-safe in-memory cache with TTL and manual invalidation."""
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._data.get(key)
            if entry and time.time() < entry['expires']:
                return entry['value']
            if entry:
                del self._data[key]
            return None

    def set(self, key, value, ttl=300):
        with self._lock:
            self._data[key] = {'value': value, 'expires': time.time() + ttl}

    def invalidate(self, *keys):
        with self._lock:
            for key in keys:
                self._data.pop(key, None)

cache = SimpleCache()

# ============================================
# MULTI-MODEL LLM FALLBACK
# ============================================

class LLMError(Exception):
    """Raised when all LLM providers fail."""
    def __init__(self, errors):
        self.errors = errors  # list of (provider, model, exception)
        messages = [f"{p}/{m}: {e}" for p, m, e in errors]
        super().__init__(f"All LLM providers failed: {'; '.join(messages)}")

MODEL_TIERS = {
    "fast": [
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("gemini",    "gemini-2.0-flash"),
        ("openai",    "gpt-4o-mini"),
    ],
    "standard": [
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("gemini",    "gemini-2.0-flash"),
        ("openai",    "gpt-4o"),
    ],
    "advanced": [
        ("anthropic", "claude-opus-4-6"),
        ("gemini",    "gemini-2.0-pro"),
        ("openai",    "gpt-4o"),
    ],
}

def _get_api_keys(anthropic_api_key="", gemini_api_key="", openai_api_key=""):
    """Resolve API keys from explicit params or environment variables."""
    return {
        "anthropic": anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        "gemini":    gemini_api_key or os.environ.get("GEMINI_API_KEY", ""),
        "openai":    openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
    }

def _is_retryable(provider, error):
    """Determine if an error should trigger fallback to next provider."""
    if provider == "anthropic":
        if isinstance(error, anthropic.AuthenticationError):
            return False
        if isinstance(error, (anthropic.RateLimitError, anthropic.APIConnectionError)):
            return True
        if isinstance(error, anthropic.APIStatusError):
            return error.status_code in (429, 500, 502, 503, 529)
        if isinstance(error, requests.Timeout):
            return True
        return True  # Network errors etc.
    elif provider == "gemini":
        err_str = str(error)
        return "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "500" in err_str or "503" in err_str or isinstance(error, (ConnectionError, TimeoutError))
    elif provider == "openai":
        if isinstance(error, openai.AuthenticationError):
            return False
        if isinstance(error, (openai.RateLimitError, openai.APIConnectionError)):
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code in (429, 500, 502, 503)
        return True
    return True

def _call_anthropic(*, messages, system, model, max_tokens, timeout, api_key):
    """Call Anthropic API using the SDK. Returns normalized response dict."""
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text
    return {
        "text": text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "provider": "anthropic",
        "model": model,
    }

def _call_gemini(*, messages, system, model, max_tokens, timeout, api_key):
    """Call Google Gemini API. Returns normalized response dict."""
    client = genai.Client(api_key=api_key)
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        if isinstance(msg["content"], list):
            parts = []
            for block in msg["content"]:
                if block.get("type") == "text":
                    parts.append(genai_types.Part.from_text(text=block["text"]))
                elif block.get("type") == "image":
                    source = block["source"]
                    parts.append(genai_types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    ))
                elif block.get("type") == "document":
                    source = block["source"]
                    parts.append(genai_types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source.get("media_type", "application/pdf"),
                    ))
            contents.append(genai_types.Content(role=role, parts=parts))
        else:
            contents.append(genai_types.Content(
                role=role,
                parts=[genai_types.Part.from_text(text=msg["content"])]
            ))
    config = genai_types.GenerateContentConfig(max_output_tokens=max_tokens)
    if system:
        config.system_instruction = system
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    text = response.text or ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage["input_tokens"] = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        usage["output_tokens"] = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
    return {
        "text": text,
        "usage": usage,
        "provider": "gemini",
        "model": model,
    }

def _call_openai(*, messages, system, model, max_tokens, timeout, api_key):
    """Call OpenAI API. Returns normalized response dict."""
    client = openai.OpenAI(api_key=api_key, timeout=timeout)
    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    for msg in messages:
        role = msg["role"]
        if isinstance(msg["content"], list):
            oai_content = []
            for block in msg["content"]:
                if block.get("type") == "text":
                    oai_content.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    source = block["source"]
                    data_uri = f"data:{source['media_type']};base64,{source['data']}"
                    oai_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
                elif block.get("type") == "document":
                    # OpenAI chat completions don't support inline PDFs — skip
                    oai_content.append({"type": "text", "text": "[PDF document — content not available for this provider]"})
            oai_messages.append({"role": role, "content": oai_content})
        else:
            oai_messages.append({"role": role, "content": msg["content"]})
    response = client.chat.completions.create(
        model=model,
        messages=oai_messages,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    return {
        "text": text,
        "usage": {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        },
        "provider": "openai",
        "model": model,
    }

_LLM_ADAPTERS = {
    "anthropic": _call_anthropic,
    "gemini":    _call_gemini,
    "openai":    _call_openai,
}

def call_llm(*, messages, system="", tier="standard", max_tokens=4096,
             timeout=120, anthropic_api_key="", gemini_api_key="", openai_api_key=""):
    """Call LLM with automatic multi-provider fallback.

    Returns dict with keys: text, usage, provider, model.
    Raises LLMError if all providers fail.
    """
    api_keys = _get_api_keys(anthropic_api_key, gemini_api_key, openai_api_key)
    chain = MODEL_TIERS.get(tier, MODEL_TIERS["standard"])
    errors = []
    for provider, model in chain:
        key = api_keys.get(provider, "")
        if not key:
            continue
        try:
            print(f"[LLM Fallback] Trying {provider}/{model}...")
            result = _LLM_ADAPTERS[provider](
                messages=messages, system=system, model=model,
                max_tokens=max_tokens, timeout=timeout, api_key=key,
            )
            print(f"[LLM Fallback] Success with {provider}/{model}")
            return result
        except Exception as e:
            print(f"[LLM Fallback] {provider}/{model} failed: {type(e).__name__}: {e}")
            errors.append((provider, model, e))
            if not _is_retryable(provider, e):
                break
    raise LLMError(errors)

def call_llm_stream(*, messages, system="", tier="standard", max_tokens=16384,
                    anthropic_api_key="", gemini_api_key="", openai_api_key=""):
    """Generator: yields keep-alive spaces, then final result dict.

    Usage:
        for chunk in call_llm_stream(...):
            if isinstance(chunk, dict):
                llm_result = chunk  # final result
            else:
                yield chunk  # keep-alive space
    """
    api_keys = _get_api_keys(anthropic_api_key, gemini_api_key, openai_api_key)
    chain = MODEL_TIERS.get(tier, MODEL_TIERS["standard"])
    errors = []
    for provider, model in chain:
        key = api_keys.get(provider, "")
        if not key:
            continue
        try:
            print(f"[LLM Stream Fallback] Trying {provider}/{model}...")
            if provider == "anthropic":
                client = anthropic.Anthropic(api_key=key)
                result_text = ""
                kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
                if system:
                    kwargs["system"] = system
                with client.messages.stream(**kwargs) as stream:
                    for text in stream.text_stream:
                        result_text += text
                        yield " "
                    response = stream.get_final_message()
                print(f"[LLM Stream Fallback] Success with {provider}/{model}")
                yield {
                    "text": result_text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "provider": provider,
                    "model": model,
                }
                return
            else:
                # Gemini/OpenAI: run non-streaming in background thread, yield keep-alive
                future_result = {}
                future_error = {}
                def _run_call(p=provider, m=model, k=key):
                    try:
                        future_result["data"] = _LLM_ADAPTERS[p](
                            messages=messages, system=system, model=m,
                            max_tokens=max_tokens, timeout=300, api_key=k,
                        )
                    except Exception as exc:
                        future_error["err"] = exc
                thread = threading.Thread(target=_run_call)
                thread.start()
                while thread.is_alive():
                    yield " "
                    thread.join(timeout=2.0)
                if "err" in future_error:
                    raise future_error["err"]
                print(f"[LLM Stream Fallback] Success with {provider}/{model}")
                yield future_result["data"]
                return
        except Exception as e:
            print(f"[LLM Stream Fallback] {provider}/{model} failed: {type(e).__name__}: {e}")
            errors.append((provider, model, e))
            if not _is_retryable(provider, e):
                break
    raise LLMError(errors)

# ============================================
# DATABASE CONNECTION
# ============================================

from contextlib import contextmanager

# Connection pool — initialized lazily, one per Gunicorn worker process
_pool = None

def _get_database_url():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception('DATABASE_URL environment variable not set')
    # Render uses postgres:// but psycopg2 needs postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    return database_url

def _get_pool():
    """Get or create the connection pool (thread-safe, lazy init)."""
    global _pool
    if _pool is None or _pool.closed:
        # 3 workers × 2 threads = 6 handlers; pool up to 10 per worker
        _pool = ThreadedConnectionPool(
            minconn=2, maxconn=10,
            dsn=_get_database_url(),
            cursor_factory=RealDictCursor
        )
        print(f"DB connection pool created (min=2, max=10)")
    return _pool

def get_db_connection():
    """Get a connection from the pool."""
    return _get_pool().getconn()

@contextmanager
def get_db(commit=False):
    """Context manager for pooled database connections. Returns connection to pool on exit."""
    pool = _get_pool()
    conn = pool.getconn()
    cur = conn.cursor()
    try:
        yield conn, cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        # Roll back any implicit transaction from reads before returning to pool
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            if conn.closed:
                pool.putconn(conn, close=True)
            else:
                pool.putconn(conn)
        except Exception:
            pass

def init_db():
    """Initialize database tables"""
    try:
        with get_db(commit=True) as (_, cur):
            # Portfolio Analyses table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_analyses (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) UNIQUE NOT NULL,
                    company VARCHAR(255),
                    analysis JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Stock Overviews table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS stock_overviews (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) UNIQUE NOT NULL,
                    company_name VARCHAR(255),
                    company_overview TEXT,
                    business_model TEXT,
                    business_mix TEXT,
                    opportunities TEXT,
                    risks TEXT,
                    conclusion TEXT,
                    raw_content TEXT,
                    history JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Add business_mix column if it doesn't exist (migration)
            cur.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='stock_overviews' AND column_name='business_mix') THEN
                        ALTER TABLE stock_overviews ADD COLUMN business_mix TEXT;
                    END IF;
                END $$;
            ''')

            # Chat Histories table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS chat_histories (
                    id VARCHAR(100) PRIMARY KEY,
                    title VARCHAR(255),
                    messages JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Meeting Summaries table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS meeting_summaries (
                    id VARCHAR(100) PRIMARY KEY,
                    title VARCHAR(255),
                    raw_notes TEXT,
                    summary TEXT,
                    questions TEXT,
                    topic VARCHAR(100) DEFAULT 'General',
                    topic_type VARCHAR(20) DEFAULT 'other',
                    source_type VARCHAR(20) DEFAULT 'paste',
                    source_files JSONB DEFAULT '[]',
                    doc_type VARCHAR(50) DEFAULT 'other',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Add columns if they don't exist (migration)
            cur.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='meeting_summaries' AND column_name='topic') THEN
                        ALTER TABLE meeting_summaries ADD COLUMN topic VARCHAR(100) DEFAULT 'General';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='meeting_summaries' AND column_name='topic_type') THEN
                        ALTER TABLE meeting_summaries ADD COLUMN topic_type VARCHAR(20) DEFAULT 'other';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='meeting_summaries' AND column_name='source_type') THEN
                        ALTER TABLE meeting_summaries ADD COLUMN source_type VARCHAR(20) DEFAULT 'paste';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='meeting_summaries' AND column_name='source_files') THEN
                        ALTER TABLE meeting_summaries ADD COLUMN source_files JSONB DEFAULT '[]';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='meeting_summaries' AND column_name='doc_type') THEN
                        ALTER TABLE meeting_summaries ADD COLUMN doc_type VARCHAR(50) DEFAULT 'other';
                    END IF;
                END $$;
            ''')

            # Document Files table - stores actual document content for re-analysis
            cur.execute('''
                CREATE TABLE IF NOT EXISTS document_files (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    file_data TEXT NOT NULL,
                    file_type VARCHAR(50),
                    mime_type VARCHAR(100),
                    metadata JSONB DEFAULT '{}',
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, filename)
                )
            ''')

            # Research Categories (tickers + topics)
            cur.execute('''
                CREATE TABLE IF NOT EXISTS research_categories (
                    id VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    type VARCHAR(20) DEFAULT 'ticker',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Research Documents (files/text under categories)
            cur.execute('''
                CREATE TABLE IF NOT EXISTS research_documents (
                    id VARCHAR(100) PRIMARY KEY,
                    category_id VARCHAR(100) REFERENCES research_categories(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    content TEXT,
                    file_names JSONB DEFAULT '[]',
                    smart_name VARCHAR(500),
                    original_filename VARCHAR(500),
                    published_date VARCHAR(100),
                    doc_type VARCHAR(50) DEFAULT 'other',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Migration: Add new columns if they don't exist
            cur.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='research_documents' AND column_name='smart_name') THEN
                        ALTER TABLE research_documents ADD COLUMN smart_name VARCHAR(500);
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='research_documents' AND column_name='original_filename') THEN
                        ALTER TABLE research_documents ADD COLUMN original_filename VARCHAR(500);
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='research_documents' AND column_name='published_date') THEN
                        ALTER TABLE research_documents ADD COLUMN published_date VARCHAR(100);
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='research_documents' AND column_name='has_stored_files') THEN
                        ALTER TABLE research_documents ADD COLUMN has_stored_files BOOLEAN DEFAULT FALSE;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='research_documents' AND column_name='doc_type') THEN
                        ALTER TABLE research_documents ADD COLUMN doc_type VARCHAR(50) DEFAULT 'other';
                    END IF;
                END $$;
            ''')

            # Research Document Files (stored PDFs/files for re-analysis)
            cur.execute('''
                CREATE TABLE IF NOT EXISTS research_document_files (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(100) REFERENCES research_documents(id) ON DELETE CASCADE,
                    filename VARCHAR(500) NOT NULL,
                    file_type VARCHAR(100),
                    file_data TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Summary Files (stored PDFs/files for summaries)
            cur.execute('''
                CREATE TABLE IF NOT EXISTS summary_files (
                    id SERIAL PRIMARY KEY,
                    summary_id VARCHAR(100) REFERENCES meeting_summaries(id) ON DELETE CASCADE,
                    filename VARCHAR(500) NOT NULL,
                    file_type VARCHAR(100),
                    file_data TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Migration: Add has_stored_files to meeting_summaries if not exists
            cur.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                  WHERE table_name='meeting_summaries' AND column_name='has_stored_files') THEN
                        ALTER TABLE meeting_summaries ADD COLUMN has_stored_files BOOLEAN DEFAULT FALSE;
                    END IF;
                END $$;
            ''')

            # Research Analyses (framework results under documents)
            cur.execute('''
                CREATE TABLE IF NOT EXISTS research_analyses (
                    id VARCHAR(100) PRIMARY KEY,
                    document_id VARCHAR(100) REFERENCES research_documents(id) ON DELETE CASCADE,
                    prompt_id VARCHAR(100),
                    prompt_name VARCHAR(255),
                    prompt_icon VARCHAR(10),
                    result TEXT,
                    usage JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index for faster lookups
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_document_files_ticker
                ON document_files(ticker)
            ''')

            # ============================================
            # MEETING PREP TABLES
            # ============================================

            cur.execute('''
                CREATE TABLE IF NOT EXISTS mp_companies (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    sector VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS mp_meetings (
                    id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES mp_companies(id),
                    meeting_date DATE,
                    meeting_type VARCHAR(50) DEFAULT 'other',
                    status VARCHAR(20) DEFAULT 'draft',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_mp_meetings_company ON mp_meetings(company_id)')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS mp_documents (
                    id SERIAL PRIMARY KEY,
                    meeting_id INTEGER REFERENCES mp_meetings(id) ON DELETE CASCADE,
                    filename VARCHAR(500) NOT NULL,
                    file_data TEXT,
                    doc_type VARCHAR(50) DEFAULT 'other',
                    doc_date VARCHAR(20),
                    page_count INTEGER,
                    token_estimate INTEGER,
                    extracted_text TEXT,
                    upload_order INTEGER,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_mp_documents_meeting ON mp_documents(meeting_id)')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS mp_question_sets (
                    id SERIAL PRIMARY KEY,
                    meeting_id INTEGER REFERENCES mp_meetings(id) ON DELETE CASCADE,
                    version INTEGER DEFAULT 1,
                    status VARCHAR(20) DEFAULT 'ready',
                    topics_json TEXT,
                    synthesis_json TEXT,
                    generation_model VARCHAR(100),
                    generation_tokens INTEGER,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_mp_question_sets_meeting ON mp_question_sets(meeting_id)')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS mp_past_questions (
                    id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES mp_companies(id),
                    meeting_id INTEGER REFERENCES mp_meetings(id) ON DELETE SET NULL,
                    question TEXT NOT NULL,
                    topic VARCHAR(255),
                    response_notes TEXT,
                    status VARCHAR(20) DEFAULT 'asked',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_mp_past_questions_company ON mp_past_questions(company_id)')

            # ============================================
            # SLIDE GENERATOR TABLES
            # ============================================

            cur.execute('''
                CREATE TABLE IF NOT EXISTS slide_projects (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20),
                    title VARCHAR(255) NOT NULL,
                    theme VARCHAR(50) DEFAULT 'sketchnote',
                    status VARCHAR(20) DEFAULT 'draft',
                    total_slides INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS slide_items (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES slide_projects(id) ON DELETE CASCADE,
                    slide_number INTEGER NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    type VARCHAR(30) DEFAULT 'content',
                    content TEXT,
                    illustration_hints JSONB DEFAULT '[]',
                    no_header BOOLEAN DEFAULT FALSE,
                    image_data TEXT,
                    content_hash VARCHAR(64),
                    status VARCHAR(20) DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_slide_items_project ON slide_items(project_id)')
            cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_slide_items_project_num ON slide_items(project_id, slide_number)')

            # ============================================
            # STUDIO TABLES
            # ============================================

            cur.execute('''
                CREATE TABLE IF NOT EXISTS studio_design_themes (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    style_prompt TEXT NOT NULL,
                    preview_image TEXT,
                    is_default BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS studio_outputs (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    type VARCHAR(30) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    theme_id INTEGER REFERENCES studio_design_themes(id) ON DELETE SET NULL,
                    source_config JSONB DEFAULT '{}',
                    settings JSONB DEFAULT '{}',
                    content JSONB DEFAULT '{}',
                    image_data TEXT,
                    progress_current INTEGER DEFAULT 0,
                    progress_total INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS studio_slide_images (
                    id SERIAL PRIMARY KEY,
                    output_id INTEGER REFERENCES studio_outputs(id) ON DELETE CASCADE,
                    slide_number INTEGER NOT NULL,
                    image_data TEXT,
                    content_hash VARCHAR(64),
                    status VARCHAR(20) DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_studio_slide_images_output ON studio_slide_images(output_id)')

            # Seed default studio themes from SLIDE_THEMES if table is empty
            cur.execute('SELECT COUNT(*) as cnt FROM studio_design_themes')
            if cur.fetchone()['cnt'] == 0:
                for key, theme in SLIDE_THEMES.items():
                    cur.execute('''
                        INSERT INTO studio_design_themes (name, description, style_prompt, is_default)
                        VALUES (%s, %s, %s, %s)
                    ''', (theme['name'], theme.get('illustration_guidance', ''), theme['style_prefix'], True))

        print("Database tables initialized")
    except Exception as e:
        print(f"Database init error (may be normal on first run): {e}")

# Initialize database on startup
try:
    init_db()
except:
    pass  # Will init when DATABASE_URL is available


# ============================================
# PORTFOLIO ANALYSES ENDPOINTS
# ============================================

@app.route('/api/analyses', methods=['GET'])
def get_analyses():
    """Get all saved portfolio analyses"""
    try:
        cached = cache.get('analyses')
        if cached is not None:
            return jsonify(cached)

        with get_db() as (conn, cur):
            cur.execute('''
                SELECT ticker, company, analysis, updated_at
                FROM portfolio_analyses
                ORDER BY ticker ASC
            ''')
            rows = cur.fetchall()

        result = []
        for row in rows:
            result.append({
                'ticker': row['ticker'],
                'company': row['company'],
                'analysis': row['analysis'],
                'updated': row['updated_at'].isoformat() if row['updated_at'] else None
            })

        cache.set('analyses', result, ttl=300)
        return jsonify(result)
    except Exception as e:
        print(f"Error getting analyses: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<ticker>', methods=['GET'])
def get_analysis(ticker):
    """Get a specific portfolio analysis by ticker"""
    try:
        with get_db() as (conn, cur):
            cur.execute('''
                SELECT ticker, company, analysis, updated_at
                FROM portfolio_analyses
                WHERE ticker = %s
            ''', (ticker.upper(),))
            row = cur.fetchone()

        if not row:
            return jsonify({'error': 'Analysis not found'}), 404

        return jsonify({
            'ticker': row['ticker'],
            'company': row['company'],
            'analysis': row['analysis'],
            'updated': row['updated_at'].isoformat() if row['updated_at'] else None
        })
    except Exception as e:
        print(f"Error getting analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-analysis', methods=['POST'])
def save_analysis():
    """Save or update a portfolio analysis"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        company = data.get('companyName', data.get('company', ''))
        analysis = data.get('analysis', {})
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        with get_db(commit=True) as (conn, cur):
            # Upsert - insert or update
            cur.execute('''
                INSERT INTO portfolio_analyses (ticker, company, analysis, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker)
                DO UPDATE SET
                    company = EXCLUDED.company,
                    analysis = EXCLUDED.analysis,
                    updated_at = EXCLUDED.updated_at
                RETURNING ticker
            ''', (ticker, company, json.dumps(analysis), datetime.utcnow()))

            result = cur.fetchone()

        cache.invalidate('analyses')
        return jsonify({'success': True, 'ticker': result['ticker']})
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-analysis', methods=['POST'])
def delete_analysis():
    """Delete a portfolio analysis"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM portfolio_analyses WHERE ticker = %s', (ticker,))

        cache.invalidate('analyses')
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting analysis: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# STOCK OVERVIEWS ENDPOINTS
# ============================================

@app.route('/api/overviews', methods=['GET'])
def get_overviews():
    """Get all saved stock overviews"""
    try:
        cached = cache.get('overviews')
        if cached is not None:
            return jsonify(cached)

        with get_db() as (conn, cur):
            cur.execute('''
                SELECT ticker, company_name, company_overview, business_model, business_mix,
                       opportunities, risks, conclusion, raw_content, history, updated_at
                FROM stock_overviews
                ORDER BY ticker ASC
            ''')
            rows = cur.fetchall()

        result = []
        for row in rows:
            result.append({
                'ticker': row['ticker'],
                'companyName': row['company_name'],
                'companyOverview': row['company_overview'],
                'businessModel': row['business_model'],
                'businessMix': row.get('business_mix', ''),
                'opportunities': row['opportunities'],
                'risks': row['risks'],
                'conclusion': row['conclusion'],
                'rawContent': row['raw_content'],
                'history': row['history'] or [],
                'updatedAt': row['updated_at'].isoformat() if row['updated_at'] else None
            })

        cache.set('overviews', result, ttl=300)
        return jsonify(result)
    except Exception as e:
        print(f"Error getting overviews: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-overview', methods=['POST'])
def save_overview():
    """Save or update a stock overview"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        with get_db(commit=True) as (conn, cur):
            # Upsert
            cur.execute('''
                INSERT INTO stock_overviews (
                    ticker, company_name, company_overview, business_model, business_mix,
                    opportunities, risks, conclusion, raw_content, history, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker)
                DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    company_overview = EXCLUDED.company_overview,
                    business_model = EXCLUDED.business_model,
                    business_mix = EXCLUDED.business_mix,
                    opportunities = EXCLUDED.opportunities,
                    risks = EXCLUDED.risks,
                    conclusion = EXCLUDED.conclusion,
                    raw_content = EXCLUDED.raw_content,
                    history = EXCLUDED.history,
                    updated_at = EXCLUDED.updated_at
                RETURNING ticker
            ''', (
                ticker,
                data.get('companyName', ''),
                data.get('companyOverview', ''),
                data.get('businessModel', ''),
                data.get('businessMix', ''),
                data.get('opportunities', ''),
                data.get('risks', ''),
                data.get('conclusion', ''),
                data.get('rawContent', ''),
                json.dumps(data.get('history', [])),
                datetime.utcnow()
            ))

            result = cur.fetchone()

        cache.invalidate('overviews')
        return jsonify({'success': True, 'ticker': result['ticker']})
    except Exception as e:
        print(f"Error saving overview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-overview', methods=['POST'])
def delete_overview():
    """Delete a stock overview"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM stock_overviews WHERE ticker = %s', (ticker,))

        cache.invalidate('overviews')
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting overview: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# CHAT HISTORY ENDPOINTS
# ============================================

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Get all chat histories"""
    try:
        with get_db() as (conn, cur):
            cur.execute('''
                SELECT id, title, messages, updated_at
                FROM chat_histories
                ORDER BY updated_at DESC
            ''')
            rows = cur.fetchall()

        result = []
        for row in rows:
            result.append({
                'id': row['id'],
                'title': row['title'],
                'messages': row['messages'] or [],
                'updatedAt': row['updated_at'].isoformat() if row['updated_at'] else None
            })

        return jsonify(result)
    except Exception as e:
        print(f"Error getting chats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-chat', methods=['POST'])
def save_chat():
    """Save or update a chat history"""
    try:
        data = request.json
        chat_id = data.get('id', '')
        
        if not chat_id:
            return jsonify({'error': 'Chat ID is required'}), 400

        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                INSERT INTO chat_histories (id, title, messages, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    messages = EXCLUDED.messages,
                    updated_at = EXCLUDED.updated_at
                RETURNING id
            ''', (
                chat_id,
                data.get('title', 'New Chat'),
                json.dumps(data.get('messages', [])),
                datetime.utcnow()
            ))

            result = cur.fetchone()

        return jsonify({'success': True, 'id': result['id']})
    except Exception as e:
        print(f"Error saving chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-chat', methods=['POST'])
def delete_chat():
    """Delete a chat history"""
    try:
        data = request.json
        chat_id = data.get('id', '')
        
        if not chat_id:
            return jsonify({'error': 'Chat ID is required'}), 400

        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM chat_histories WHERE id = %s', (chat_id,))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# MEETING SUMMARY ENDPOINTS
# ============================================

@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    """Get all meeting summaries"""
    try:
        cached = cache.get('summaries')
        if cached is not None:
            return jsonify(cached)

        with get_db() as (conn, cur):
            cur.execute('''
                SELECT id, title, raw_notes, summary, questions, topic, topic_type, source_type, source_files, doc_type, has_stored_files, created_at
                FROM meeting_summaries
                ORDER BY created_at DESC
            ''')
            rows = cur.fetchall()

        result = []
        for row in rows:
            result.append({
                'id': row['id'],
                'title': row['title'],
                'rawNotes': row['raw_notes'],
                'summary': row['summary'],
                'questions': row['questions'],
                'topic': row.get('topic') or 'General',
                'topicType': row.get('topic_type') or 'other',
                'sourceType': row.get('source_type') or 'paste',
                'sourceFiles': row.get('source_files') or [],
                'docType': row.get('doc_type') or 'other',
                'hasStoredFiles': row.get('has_stored_files') or False,
                'createdAt': row['created_at'].isoformat() if row['created_at'] else None
            })

        cache.set('summaries', result, ttl=300)
        return jsonify(result)
    except Exception as e:
        print(f"Error getting summaries: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-summary', methods=['POST'])
def save_summary():
    """Save or update a meeting summary"""
    try:
        data = request.json
        summary_id = data.get('id', '')
        
        if not summary_id:
            return jsonify({'error': 'Summary ID is required'}), 400
        
        # Convert sourceFiles list to JSON
        source_files = data.get('sourceFiles', [])
        if isinstance(source_files, list):
            source_files = json.dumps(source_files)

        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                INSERT INTO meeting_summaries (id, title, raw_notes, summary, questions, topic, topic_type, source_type, source_files, doc_type, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    raw_notes = EXCLUDED.raw_notes,
                    summary = EXCLUDED.summary,
                    questions = EXCLUDED.questions,
                    topic = EXCLUDED.topic,
                    topic_type = EXCLUDED.topic_type,
                    source_type = EXCLUDED.source_type,
                    source_files = EXCLUDED.source_files,
                    doc_type = EXCLUDED.doc_type
                RETURNING id
            ''', (
                summary_id,
                data.get('title', 'Meeting Summary'),
                data.get('rawNotes', ''),
                data.get('summary', ''),
                data.get('questions', ''),
                data.get('topic', 'General'),
                data.get('topicType', 'other'),
                data.get('sourceType', 'paste'),
                source_files,
                data.get('docType', 'other'),
                data.get('createdAt', datetime.utcnow().isoformat())
            ))

            result = cur.fetchone()

        cache.invalidate('summaries')
        return jsonify({'success': True, 'id': result['id']})
    except Exception as e:
        print(f"Error saving summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-summary', methods=['POST'])
def delete_summary():
    """Delete a meeting summary"""
    try:
        data = request.json
        summary_id = data.get('id', '')

        if not summary_id:
            return jsonify({'error': 'Summary ID is required'}), 400

        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM summary_files WHERE summary_id = %s', (summary_id,))
            cur.execute('DELETE FROM meeting_summaries WHERE id = %s', (summary_id,))

        cache.invalidate('summaries')
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting summary: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# SUMMARY FILES ENDPOINTS
# ============================================

@app.route('/api/summary-files/<summary_id>', methods=['GET'])
def get_summary_files(summary_id):
    """Get stored files for a summary"""
    try:
        with get_db() as (conn, cur):
            cur.execute('''
                SELECT id, filename, file_type, file_data, file_size, created_at
                FROM summary_files
                WHERE summary_id = %s
                ORDER BY created_at ASC
            ''', (summary_id,))

            files = []
            for row in cur.fetchall():
                files.append({
                    'id': row['id'],
                    'filename': row['filename'],
                    'fileType': row['file_type'],
                    'fileData': row['file_data'],
                    'fileSize': row['file_size'],
                    'createdAt': row['created_at'].isoformat() if row['created_at'] else None
                })

        return jsonify(files)
    except Exception as e:
        print(f"Error getting summary files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary-files/<summary_id>', methods=['POST'])
def save_summary_files(summary_id):
    """Save files for a summary"""
    try:
        data = request.json
        files = data.get('files', [])
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400

        with get_db(commit=True) as (conn, cur):
            saved_count = 0
            for file_data in files:
                cur.execute('''
                    INSERT INTO summary_files (summary_id, filename, file_type, file_data, file_size)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (
                    summary_id,
                    file_data.get('filename', 'document.pdf'),
                    file_data.get('fileType', 'application/pdf'),
                    file_data.get('fileData', ''),
                    file_data.get('fileSize', 0)
                ))
                saved_count += 1

            # Update has_stored_files flag on the summary
            cur.execute('''
                UPDATE meeting_summaries
                SET has_stored_files = TRUE
                WHERE id = %s
            ''', (summary_id,))

        return jsonify({'success': True, 'savedCount': saved_count})
    except Exception as e:
        print(f"Error saving summary files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary-files/<summary_id>', methods=['DELETE'])
def delete_summary_files(summary_id):
    """Delete all stored files for a summary"""
    try:
        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM summary_files WHERE summary_id = %s', (summary_id,))

            # Update has_stored_files flag
            cur.execute('''
                UPDATE meeting_summaries
                SET has_stored_files = FALSE
                WHERE id = %s
            ''', (summary_id,))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting summary files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/email-summary-section', methods=['POST'])
def email_summary_section():
    """Email a specific section of a summary (takeaways or questions)"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import re
    
    try:
        data = request.json
        email = data.get('email', '')
        subject = data.get('subject', 'Summary Notes')
        section = data.get('section', '')  # 'takeaways' or 'questions'
        content = data.get('content', '')
        title = data.get('title', 'Meeting Summary')
        topic = data.get('topic', 'General')
        smtp_config = data.get('smtpConfig', {})
        
        if not email or not content:
            return jsonify({'error': 'Email and content are required'}), 400
        
        # Get SMTP configuration
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required. Please set them in Settings.'}), 400
        
        # Convert HTML to plain text
        plain_text = re.sub(r'<[^>]+>', '', content)
        plain_text = plain_text.replace('&nbsp;', ' ').replace('&amp;', '&')
        
        # Format the section label
        section_label = "Key Takeaways" if section == 'takeaways' else "Follow-up Questions"
        header_color = "#0d9488" if section == 'takeaways' else "#d97706"
        
        # Build HTML email
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: {header_color};
                    border-bottom: 2px solid {header_color};
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #374151;
                    margin-top: 24px;
                }}
                h3 {{
                    color: #4b5563;
                }}
                ul, ol {{
                    padding-left: 24px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                strong {{
                    color: #111;
                }}
                .header {{
                    background: linear-gradient(135deg, {header_color} 0%, {'#0891b2' if section == 'takeaways' else '#ea580c'} 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 24px;
                }}
                .topic-badge {{
                    display: inline-block;
                    background: rgba(255,255,255,0.2);
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    margin-top: 8px;
                }}
                .content {{
                    background: #f9fafb;
                    padding: 24px;
                    border-radius: 8px;
                    border: 1px solid #e5e7eb;
                }}
                .footer {{
                    margin-top: 24px;
                    padding-top: 16px;
                    border-top: 1px solid #e5e7eb;
                    font-size: 12px;
                    color: #6b7280;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="color: white; border: none; margin: 0;">{section_label}</h1>
                <p style="margin: 8px 0 0 0; opacity: 0.9;">{title}</p>
                <span class="topic-badge">{topic}</span>
            </div>
            <div class="content">
                {content}
            </div>
            <div class="footer">
                Generated by TDL Equity Analyzer
            </div>
        </body>
        </html>
        """
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = email
        msg['Subject'] = subject
        
        # Attach both plain text and HTML versions
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send via Gmail SMTP
        if use_gmail:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        print(f"Error sending summary email: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract-summary-text', methods=['POST'])
def extract_summary_text():
    """Extract text from uploaded files (PDF, DOCX, images, TXT) for summary generation.
    Accepts either multipart/form-data or JSON with base64-encoded files."""
    try:
        # Support both JSON (base64 files) and multipart/form-data uploads
        file_items = []  # list of (filename, file_content_bytes)

        if request.is_json or (request.content_type and 'application/json' in request.content_type):
            # JSON mode: files sent as base64
            data = request.get_json()
            if not data or 'files' not in data or not data['files']:
                return jsonify({'error': 'No files provided'}), 400
            api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('apiKey', '')
            for f in data['files']:
                file_content = base64.b64decode(f['fileData'])
                file_items.append((f['filename'], file_content))
            print(f"extract-summary-text (JSON): {len(file_items)} files, API key present: {bool(api_key)}")
        elif 'files' in request.files:
            # Multipart mode: standard file upload
            files = request.files.getlist('files')
            if not files or files[0].filename == '':
                return jsonify({'error': 'No files selected'}), 400
            api_key = os.environ.get('ANTHROPIC_API_KEY', '') or request.form.get('apiKey', '')
            for file in files:
                file_items.append((file.filename, file.read()))
            print(f"extract-summary-text (multipart): {len(file_items)} files, API key present: {bool(api_key)}")
        else:
            return jsonify({'error': 'No files provided'}), 400

        all_text = []
        first_filename = file_items[0][0]

        for orig_filename, file_content in file_items:
            filename = orig_filename.lower()
            extracted_text = ''
            
            try:
                # Handle PDF files
                if filename.endswith('.pdf'):
                    try:
                        import io
                        from PyPDF2 import PdfReader
                        pdf_reader = PdfReader(io.BytesIO(file_content))
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                extracted_text += text + '\n\n'
                    except ImportError:
                        # Fallback: try pdfplumber
                        try:
                            import pdfplumber
                            import io
                            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                                for page in pdf.pages:
                                    text = page.extract_text()
                                    if text:
                                        extracted_text += text + '\n\n'
                        except ImportError:
                            return jsonify({'error': 'PDF processing libraries not available'}), 500
                
                # Handle Word documents
                elif filename.endswith('.docx') or filename.endswith('.doc'):
                    try:
                        import io
                        from docx import Document
                        doc = Document(io.BytesIO(file_content))
                        for para in doc.paragraphs:
                            if para.text.strip():
                                extracted_text += para.text + '\n'
                        # Also extract from tables
                        for table in doc.tables:
                            for row in table.rows:
                                row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
                                if row_text:
                                    extracted_text += row_text + '\n'
                    except ImportError:
                        return jsonify({'error': 'Word document processing library not available'}), 500
                
                # Handle plain text files
                elif filename.endswith('.txt'):
                    try:
                        extracted_text = file_content.decode('utf-8')
                    except UnicodeDecodeError:
                        extracted_text = file_content.decode('latin-1')
                
                # Handle images (use Claude Vision API for OCR)
                elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    # Use API key from form data or environment
                    if api_key:
                        try:
                            ext = filename.split('.')[-1].lower()
                            media_types = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif', 'bmp': 'image/bmp', 'webp': 'image/webp'}
                            media_type = media_types.get(ext, 'image/jpeg')
                            image_base64 = base64.b64encode(file_content).decode('utf-8')

                            ocr_result = call_llm(
                                messages=[{
                                    'role': 'user',
                                    'content': [
                                        {
                                            'type': 'image',
                                            'source': {
                                                'type': 'base64',
                                                'media_type': media_type,
                                                'data': image_base64
                                            }
                                        },
                                        {
                                            'type': 'text',
                                            'text': 'Please extract ALL text from this image exactly as it appears. Preserve the original formatting, paragraphs, and structure. Output ONLY the extracted text, nothing else. If this is a screenshot of an article, extract the full article text.'
                                        }
                                    ]
                                }],
                                tier="fast",
                                max_tokens=8000,
                                timeout=60,
                                anthropic_api_key=api_key,
                            )
                            extracted_text = ocr_result["text"]
                            print(f"OCR extracted {len(extracted_text)} chars from {orig_filename} via {ocr_result['provider']}/{ocr_result['model']}")

                        except LLMError as llm_err:
                            print(f"OCR failed across all providers: {llm_err}")
                            extracted_text = f"[Image file: {orig_filename} - OCR failed: {str(llm_err)[:150]}]"
                        except Exception as ocr_error:
                            print(f"OCR error: {ocr_error}")
                            extracted_text = f"[Image file: {orig_filename} - OCR error: {str(ocr_error)[:150]}]"
                    else:
                        # Fallback to pytesseract if no API key
                        try:
                            import pytesseract
                            from PIL import Image
                            import io
                            image = Image.open(io.BytesIO(file_content))
                            extracted_text = pytesseract.image_to_string(image)
                        except ImportError:
                            extracted_text = f"[Image file: {orig_filename} - OCR not available. Please set your API key in Settings.]"
                        except Exception as ocr_error:
                            extracted_text = f"[Image file: {orig_filename} - Could not extract text: {str(ocr_error)}]"

                else:
                    # Try to read as text
                    try:
                        extracted_text = file_content.decode('utf-8')
                    except:
                        extracted_text = f"[Unsupported file type: {orig_filename}]"

                if extracted_text.strip():
                    # Add filename header if multiple files
                    if len(file_items) > 1:
                        all_text.append(f"=== {orig_filename} ===\n{extracted_text}")
                    else:
                        all_text.append(extracted_text)

            except Exception as file_error:
                print(f"Error processing file {orig_filename}: {file_error}")
                all_text.append(f"[Error processing {orig_filename}: {str(file_error)}]")
        
        combined_text = '\n\n'.join(all_text)
        
        if not combined_text.strip():
            return jsonify({'error': 'Could not extract any text from the uploaded files'}), 400
        
        # Check if OCR failed for images (all we got was placeholder text)
        if '[Image file:' in combined_text:
            # Count how many images failed
            failed_count = combined_text.count('[Image file:')
            if failed_count == len(file_items):
                return jsonify({
                    'error': f'Could not extract text from {failed_count} image(s). Please ensure your API key is set in Settings, or use PDFs/text files instead.'
                }), 400

        return jsonify({
            'success': True,
            'text': combined_text,
            'filename': first_filename,
            'fileCount': len(file_items)
        })
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# AUDIO TRANSCRIPTION ENDPOINT
# ============================================

@app.route('/api/transcribe-audio', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file using Google Gemini API"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get Gemini API key (prefer env var, fallback to frontend)
        gemini_api_key = os.environ.get('GEMINI_API_KEY', '') or request.form.get('geminiApiKey', '')
        if not gemini_api_key:
            return jsonify({'error': 'Gemini API key is required for audio transcription. Please add it in Settings.'}), 400

        # Validate file extension
        allowed_extensions = ('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.ogg', '.flac')
        filename_lower = file.filename.lower()
        if not filename_lower.endswith(allowed_extensions):
            return jsonify({'error': f'Unsupported audio format. Supported: {", ".join(allowed_extensions)}'}), 400

        # Map extensions to MIME types
        mime_map = {
            '.mp3': 'audio/mpeg', '.mp4': 'audio/mp4', '.mpeg': 'audio/mpeg',
            '.mpga': 'audio/mpeg', '.m4a': 'audio/mp4', '.wav': 'audio/wav',
            '.webm': 'audio/webm', '.ogg': 'audio/ogg', '.flac': 'audio/flac'
        }
        file_ext = '.' + filename_lower.rsplit('.', 1)[-1]
        mime_type = mime_map.get(file_ext, 'audio/mpeg')

        # Read file content
        file_content = file.read()
        file_size_mb = len(file_content) / (1024 * 1024)

        print(f"Transcribing audio with Gemini: {file.filename} ({file_size_mb:.1f}MB)")

        # Initialize Gemini client
        client = genai.Client(api_key=gemini_api_key)

        # Upload file to Gemini
        import io
        uploaded_file = client.files.upload(
            file=io.BytesIO(file_content),
            config={'mime_type': mime_type, 'display_name': file.filename}
        )

        print(f"File uploaded to Gemini: {uploaded_file.name}")

        # Transcribe with Gemini (with retry for rate limits)
        transcription_prompt = """Please provide a complete, word-for-word professional transcription of this audio recording.

Requirements:
- Transcribe EVERY word spoken, do not summarize or skip any content
- Identify different speakers where possible (e.g., "Speaker 1:", "Speaker 2:", or use names if mentioned)
- Include filler words like "um", "uh", "you know" for accuracy
- Use proper punctuation and paragraph breaks for readability
- If a speaker's name is mentioned or identifiable, use their name as the label
- Start each speaker's turn on a new line with their label
- Do NOT add any commentary, headers, timestamps, or notes - just the pure transcription"""

        import time
        models_to_try = ['gemini-3-flash-preview', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
        transcript_text = None
        last_error = None

        for model_name in models_to_try:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    print(f"Trying {model_name} (attempt {attempt + 1}/{max_retries})...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[uploaded_file, transcription_prompt]
                    )
                    transcript_text = response.text
                    print(f"Success with {model_name}")
                    break
                except Exception as retry_err:
                    last_error = retry_err
                    err_str = str(retry_err)
                    if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 10
                            print(f"Rate limited on {model_name}, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"{model_name} exhausted, falling back to next model...")
                            break
                    else:
                        raise
            if transcript_text:
                break

        if transcript_text is None:
            raise last_error or Exception("Transcription failed across all models")

        # Clean up the uploaded file
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass  # Non-critical if cleanup fails

        if not transcript_text or not transcript_text.strip():
            return jsonify({'error': 'Transcription returned empty result. The audio may be silent or unrecognizable.'}), 400

        print(f"Transcription complete: {len(transcript_text)} characters from {file.filename}")

        return jsonify({
            'success': True,
            'text': transcript_text,
            'filename': file.filename,
            'fileSizeMb': round(file_size_mb, 1),
            'charCount': len(transcript_text)
        })

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        err_str = str(e)
        if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
            return jsonify({'error': 'Gemini API rate limit reached. Please wait 1-2 minutes and try again. If this persists, check your API quota at console.cloud.google.com.'}), 429
        return jsonify({'error': f'Transcription failed: {err_str}'}), 500


@app.route('/api/text-to-docx', methods=['POST'])
def text_to_docx():
    """Convert transcript text to a .docx file and return as base64"""
    try:
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        import base64
        import io

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        title = data.get('title', 'Transcript')

        doc = Document()

        # Set default font
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)

        # Add title
        heading = doc.add_heading(title, level=1)
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # Add transcript content, preserving line breaks
        for line in text.split('\n'):
            if line.strip():
                doc.add_paragraph(line)
            else:
                doc.add_paragraph('')

        # Save to bytes
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        docx_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return jsonify({
            'success': True,
            'fileData': docx_base64,
            'fileSize': len(buffer.getvalue())
        })

    except Exception as e:
        print(f"Error creating docx: {e}")
        return jsonify({'error': f'Failed to create document: {str(e)}'}), 500


# ============================================
# DOCUMENT STORAGE ENDPOINTS
# ============================================

@app.route('/api/documents/<ticker>', methods=['GET'])
def get_documents(ticker):
    """Get all stored documents for a ticker"""
    try:
        with get_db() as (_, cur):
            cur.execute('''
                SELECT filename, file_type, mime_type, metadata, file_size, created_at
                FROM document_files
                WHERE ticker = %s
                ORDER BY created_at DESC
            ''', (ticker.upper(),))
            docs = cur.fetchall()

        return jsonify({
            'documents': [{
                'filename': d['filename'],
                'fileType': d['file_type'],
                'mimeType': d['mime_type'],
                'metadata': d['metadata'] or {},
                'fileSize': d['file_size'],
                'createdAt': d['created_at'].isoformat() if d['created_at'] else None,
                'stored': True
            } for d in docs]
        })
    except Exception as e:
        print(f"Error getting documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/<ticker>/content', methods=['GET'])
def get_documents_with_content(ticker):
    """Get all stored documents with file content for re-analysis"""
    try:
        with get_db() as (_, cur):
            cur.execute('''
                SELECT filename, file_data, file_type, mime_type, metadata
                FROM document_files
                WHERE ticker = %s
            ''', (ticker.upper(),))
            docs = cur.fetchall()

        return jsonify({
            'documents': [{
                'filename': d['filename'],
                'fileData': d['file_data'],
                'fileType': d['file_type'],
                'mimeType': d['mime_type'],
                'metadata': d['metadata'] or {},
                'stored': True
            } for d in docs]
        })
    except Exception as e:
        print(f"Error getting document content: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/save', methods=['POST'])
def save_documents():
    """Save document files to database for a ticker"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        documents = data.get('documents', [])
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        if not documents:
            return jsonify({'error': 'No documents provided'}), 400

        with get_db(commit=True) as (_, cur):
            saved_count = 0
            for doc in documents:
                filename = doc.get('filename')
                file_data = doc.get('fileData')
                file_type = doc.get('fileType', 'pdf')
                mime_type = doc.get('mimeType', 'application/pdf')
                metadata = doc.get('metadata', {})

                if not filename or not file_data:
                    continue

                # Calculate approximate file size (base64 is ~1.33x original)
                file_size = len(file_data) * 3 // 4

                cur.execute('''
                    INSERT INTO document_files (ticker, filename, file_data, file_type, mime_type, metadata, file_size)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, filename)
                    DO UPDATE SET
                        file_data = EXCLUDED.file_data,
                        file_type = EXCLUDED.file_type,
                        mime_type = EXCLUDED.mime_type,
                        metadata = EXCLUDED.metadata,
                        file_size = EXCLUDED.file_size
                ''', (ticker, filename, file_data, file_type, mime_type, json.dumps(metadata), file_size))
                saved_count += 1

        return jsonify({'success': True, 'savedCount': saved_count})
    except Exception as e:
        print(f"Error saving documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/delete', methods=['POST'])
def delete_document():
    """Delete a specific document for a ticker"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        filename = data.get('filename')
        
        if not ticker or not filename:
            return jsonify({'error': 'Ticker and filename are required'}), 400

        with get_db(commit=True) as (_, cur):
            cur.execute('''
                DELETE FROM document_files
                WHERE ticker = %s AND filename = %s
            ''', (ticker, filename))
            deleted = cur.rowcount > 0

        return jsonify({'success': True, 'deleted': deleted})
    except Exception as e:
        print(f"Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/delete-all', methods=['POST'])
def delete_all_documents():
    """Delete all documents for a ticker"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        with get_db(commit=True) as (_, cur):
            cur.execute('DELETE FROM document_files WHERE ticker = %s', (ticker,))
            deleted_count = cur.rowcount

        return jsonify({'success': True, 'deletedCount': deleted_count})
    except Exception as e:
        print(f"Error deleting all documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/storage-stats', methods=['GET'])
def get_storage_stats():
    """Get storage statistics"""
    try:
        with get_db() as (_, cur):
            cur.execute('''
                SELECT
                    ticker,
                    COUNT(*) as doc_count,
                    SUM(file_size) as total_size
                FROM document_files
                GROUP BY ticker
                ORDER BY total_size DESC
            ''')
            stats = cur.fetchall()

            cur.execute('SELECT SUM(file_size) as total FROM document_files')
            total = cur.fetchone()

        return jsonify({
            'byTicker': [{
                'ticker': s['ticker'],
                'docCount': s['doc_count'],
                'totalSize': s['total_size'] or 0
            } for s in stats],
            'totalSize': total['total'] or 0 if total else 0
        })
    except Exception as e:
        print(f"Error getting storage stats: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# ANTHROPIC API PROXY ENDPOINTS
# ============================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """Proxy chat requests to Anthropic API"""
    try:
        data = request.json
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('api_key', '')
        messages = data.get('messages', [])
        system = data.get('system', '')

        if not api_key:
            return jsonify({'error': 'No API key provided. Please add your API key in Settings.'}), 400

        result = call_llm(
            messages=messages,
            system=system,
            tier="standard",
            max_tokens=4096,
            timeout=120,
            anthropic_api_key=api_key,
        )

        return jsonify({
            'response': result["text"],
            'usage': result["usage"]
        })

    except LLMError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-multi', methods=['POST'])
def analyze_multi():
    """Analyze multiple PDF documents and generate investment thesis"""
    try:
        data = request.json
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('apiKey', '')
        documents = data.get('documents', [])
        existing_analysis = data.get('existingAnalysis')
        historical_weights = data.get('historicalWeights', [])
        weighting_config = data.get('weightingConfig', {})

        if not api_key:
            return jsonify({'error': 'No API key provided. Please add your API key in Settings.'}), 400
        
        if not documents:
            return jsonify({'error': 'No documents provided'}), 400
        
        # Filter enabled documents
        enabled_docs = [d for d in documents if d.get('enabled', True)]
        
        if not enabled_docs:
            return jsonify({'error': 'No enabled documents'}), 400
        
        # Build the content array for Claude
        content = []
        
        # Check if using simple weighting mode
        simple_mode = weighting_config.get('mode') == 'simple'
        existing_weight = weighting_config.get('existingAnalysisWeight', 70) if simple_mode else None
        new_docs_weight = weighting_config.get('newDocsWeight', 30) if simple_mode else None
        
        # Build weighting information string
        weight_info = ""
        
        # Pre-categorize documents for simple mode
        truly_new_docs = [d for d in enabled_docs if d.get('isNew', True)]
        stored_existing_docs = [d for d in enabled_docs if not d.get('isNew', True)]
        
        if simple_mode and existing_analysis:
            # Simple mode: clear instruction about preservation vs updates
            weight_info = f"""=== ANALYSIS UPDATE MODE: SIMPLE WEIGHTING ===

PRESERVATION RATIO: {existing_weight}% existing / {new_docs_weight}% new

This means:
- PRESERVE {existing_weight}% of the existing thesis, pillars, signposts, and threats
- Allow only {new_docs_weight}% worth of modifications from the new document(s)
- The new document(s) are SUPPLEMENTARY, not replacements

"""
            if stored_existing_docs:
                weight_info += "EXISTING DOCUMENTS (re-uploaded for reference, part of the preserved analysis):\n"
                for doc in stored_existing_docs:
                    doc_name = doc.get('filename', 'document.pdf')
                    weight_info += f"- {doc_name}\n"
                weight_info += "\n"
            
            if truly_new_docs:
                per_new_doc_weight = new_docs_weight / len(truly_new_docs)
                weight_info += f"NEW DOCUMENTS (sharing the {new_docs_weight}% update allocation):\n"
                for doc in truly_new_docs:
                    doc_name = doc.get('filename', 'document.pdf')
                    weight_info += f"- {doc_name} ({round(per_new_doc_weight)}% weight)\n"
                weight_info += "\n"
            
            weight_info += f"""Remember: With {existing_weight}% preservation, you should keep most existing content intact.
Only add minor refinements, new data points, or small additions from the new document(s).
Do NOT rewrite or fundamentally change the existing analysis.

"""
        elif simple_mode and not existing_analysis:
            # Simple mode but new analysis - just list documents
            weight_info = "DOCUMENT WEIGHTING:\n\n"
            weight_info += "NEW DOCUMENTS (being analyzed now):\n"
            for doc in enabled_docs:
                doc_name = doc.get('filename', 'document.pdf')
                weight_info += f"- {doc_name}\n"
            weight_info += "\n"
        elif not simple_mode:
            # Advanced mode: per-document weights
            # Calculate total weight including both new and historical docs
            new_doc_weight = sum(doc.get('weight', 1) for doc in enabled_docs)
            hist_doc_weight = sum(hw.get('weight', 1) for hw in historical_weights)
            total_weight = new_doc_weight + hist_doc_weight
            
            # Historical documents (from existing analysis)
            if historical_weights:
                weight_info += "PREVIOUSLY ANALYZED DOCUMENTS (their insights are in the existing analysis):\n"
                for hw in historical_weights:
                    hw_name = hw.get('filename', 'document')
                    hw_weight = hw.get('weight', 1)
                    hw_pct = round((hw_weight / total_weight) * 100) if total_weight > 0 else 0
                    weight_info += f"- {hw_name}: {hw_pct}% weight\n"
                weight_info += "\n"
            
            # New documents being analyzed now
            weight_info += "NEW DOCUMENTS (being analyzed now):\n"
            for doc in enabled_docs:
                doc_name = doc.get('filename', 'document.pdf')
                doc_weight = doc.get('weight', 1)
                doc_pct = round((doc_weight / total_weight) * 100) if total_weight > 0 else 0
                weight_info += f"- {doc_name}: {doc_pct}% weight\n"
            
            weight_info += "\nWhen synthesizing the analysis:\n"
            weight_info += "- Give MORE emphasis to higher-weighted documents\n"
            weight_info += "- If updating existing analysis, respect the weights of previously analyzed documents\n"
            weight_info += "- Higher-weighted historical docs = keep more of their conclusions in the existing analysis\n"
        
        content.append({
            "type": "text",
            "text": weight_info
        })
        
        # Calculate total weight for document headers (use simple mode weight or calculated weight)
        if simple_mode and existing_analysis:
            # In simple mode, only truly NEW documents share the new_docs_weight
            # Stored existing documents are re-uploaded for context but shouldn't count as "new"
            truly_new_docs = [d for d in enabled_docs if d.get('isNew', True)]
            stored_docs = [d for d in enabled_docs if not d.get('isNew', True)]
            
            per_new_doc_weight = new_docs_weight / len(truly_new_docs) if truly_new_docs else 0
        else:
            new_doc_weight = sum(doc.get('weight', 1) for doc in enabled_docs)
            hist_doc_weight = sum(hw.get('weight', 1) for hw in historical_weights)
            total_weight = new_doc_weight + hist_doc_weight
        
        # Add each document
        for doc in enabled_docs:
            doc_content = doc.get('fileData', '')
            doc_name = doc.get('filename', 'document.pdf')
            doc_type = doc.get('fileType', 'pdf')
            mime_type = doc.get('mimeType', 'application/pdf')
            is_new = doc.get('isNew', True)
            
            if simple_mode and existing_analysis:
                if is_new:
                    doc_pct = round(per_new_doc_weight)
                    doc_header = f"\n=== NEW DOCUMENT (Supplementary - {doc_pct}% update weight): {doc_name} ==="
                else:
                    # Stored existing document - re-uploaded for reference, part of the existing analysis
                    doc_header = f"\n=== EXISTING DOCUMENT (Reference - part of {existing_weight}% preserved analysis): {doc_name} ==="
            else:
                doc_weight = doc.get('weight', 1)
                doc_pct = round((doc_weight / total_weight) * 100) if total_weight > 0 else 0
                doc_header = f"\n=== DOCUMENT: {doc_name} (Weight: {doc_pct}%) ==="
            
            if not doc_content:
                continue
            
            # Add document header with weight
            content.append({
                "type": "text",
                "text": doc_header
            })
                
            if doc_type == 'pdf':
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": doc_content
                    }
                })
            elif doc_type == 'image':
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type or "image/png",
                        "data": doc_content
                    }
                })
            else:
                try:
                    decoded_text = base64.b64decode(doc_content).decode('utf-8')
                    content.append({
                        "type": "text",
                        "text": decoded_text
                    })
                except:
                    continue
        
        if not content:
            return jsonify({'error': 'No valid documents to analyze'}), 400
        
        # Add the analysis prompt
        analysis_prompt = """Analyze these broker research documents and create a comprehensive investment analysis.

Return a JSON object with this exact structure:
{
    "ticker": "STOCK_TICKER",
    "company": "Company Name",
    "thesis": {
        "summary": "2-3 sentence investment thesis summary",
        "pillars": [
            {"title": "Pillar 1 Title", "description": "Detailed explanation", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]},
            {"title": "Pillar 2 Title", "description": "Detailed explanation", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]}
        ]
    },
    "signposts": [
        {"metric": "Key metric or KPI name", "target": "Target value or outcome", "timeframe": "When to expect", "category": "Financial/Operational/Strategic/Market", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]},
        {"metric": "Another metric name", "target": "Target", "timeframe": "Timeframe", "category": "Financial/Operational/Strategic/Market", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]}
    ],
    "threats": [
        {"threat": "Risk factor description", "likelihood": "High/Medium/Low", "impact": "High/Medium/Low", "triggerPoints": "What to watch for - early warning signs", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]},
        {"threat": "Another risk", "likelihood": "Medium", "impact": "High", "triggerPoints": "Monitoring triggers", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]}
    ],
    "documentMetadata": [
        {"filename": "exact_filename.pdf", "docType": "broker_report", "source": "Citi", "publishDate": "YYYY-MM-DD", "authors": ["Analyst Name"], "title": "Report Title"},
        {"filename": "transcript.pdf", "docType": "earnings_call", "source": "Company Name", "publishDate": "YYYY-MM-DD", "quarter": "Q3 2025", "title": "Q3 2025 Earnings Call"},
        {"filename": "email_screenshot.png", "docType": "email", "source": "Sender Name/Org", "publishDate": "YYYY-MM-DD", "title": "Email Subject"}
    ]
}

DOCUMENT METADATA EXTRACTION (CRITICAL):
For EACH document provided, identify the document type and extract appropriate metadata:

**For Broker Reports:**
- "docType": "broker_report"
- "source": Investment bank/broker name (e.g., "Citi", "Morgan Stanley", "Goldman Sachs", "Wolfe Research")
- "publishDate": Report date in YYYY-MM-DD format
- "authors": Array of analyst names
- "title": Report title/headline

**For Earnings Call Transcripts:**
- "docType": "earnings_call"  
- "source": Company name (e.g., "Union Pacific", "Apple Inc.")
- "publishDate": Call date in YYYY-MM-DD format
- "quarter": Fiscal quarter (e.g., "Q3 2025", "FY 2025")
- "title": e.g., "Q3 2025 Earnings Call Transcript"

**For SEC Filings (10-K, 10-Q, 8-K):**
- "docType": "sec_filing"
- "source": Company name
- "publishDate": Filing date in YYYY-MM-DD format
- "filingType": "10-K", "10-Q", "8-K", etc.
- "title": Filing description

**For Emails/Email Screenshots:**
- "docType": "email"
- "source": Sender name or organization
- "publishDate": Email date in YYYY-MM-DD format
- "title": Email subject line if visible

**For Company Presentations:**
- "docType": "presentation"
- "source": Company name or presenting organization
- "publishDate": Presentation date in YYYY-MM-DD format
- "title": Presentation title

**For News Articles:**
- "docType": "news"
- "source": Publication name (e.g., "Wall Street Journal", "Reuters")
- "publishDate": Article date in YYYY-MM-DD format
- "authors": Array of journalist names if visible
- "title": Article headline

**For Screenshots/Images (if content type unclear):**
- "docType": "screenshot"
- "source": Infer source if visible in image
- "publishDate": Infer date if visible, otherwise null
- "title": Brief description of content

Always include "filename" with the exact filename provided.

IMPORTANT STYLE RULES:
- Do NOT reference any sellside broker names (e.g., "Goldman Sachs believes...", "According to Morgan Stanley...")
- Do NOT reference specific analyst names
- Do NOT include specific broker price targets
- Write as independent analysis that synthesizes the information without attribution to sources in the prose
- The output should read like original independent research, not a summary of broker views

DOCUMENT WEIGHTING:
- Each document has an assigned weight percentage shown at the start
- Give MORE emphasis to higher-weighted documents when forming conclusions
- Higher-weighted documents should have more influence on the thesis, signposts, and threats
- If documents conflict, prefer the view from the higher-weighted document

Focus on:
1. Why own this stock? (Investment Thesis) - include confidence level and source citations
2. What are we looking for? (Signposts - specific KPIs, events, milestones with metric names)
3. Where can we be wrong? (Threats - bear case scenarios with likelihood, impact, and trigger points)

For each pillar, signpost, and threat, include:
- "sources": Array of source documents that support this point, with filename and a brief excerpt
- Use the actual document filenames provided in the analysis

Return ONLY valid JSON, no markdown, no explanation."""

        if existing_analysis:
            # Build weighting instruction specific to the mode
            if simple_mode:
                weighting_instruction = f"""
CRITICAL WEIGHTING INSTRUCTION (SIMPLE MODE):
You MUST preserve {existing_weight}% of the existing analysis. The new documents can only contribute {new_docs_weight}% worth of changes.

What this means:
- KEEP {existing_weight}% of the existing thesis, pillars, signposts, and threats UNCHANGED
- Only make MINOR refinements or additions based on the new document(s)
- Do NOT fundamentally rewrite or replace the existing analysis
- Do NOT treat the new document as a "primary source" - it is a SUPPLEMENTARY source
- The new document should ADD to or SLIGHTLY REFINE the existing analysis, not replace it

Example of correct behavior with {existing_weight}% existing / {new_docs_weight}% new:
- If existing thesis has 3 pillars, keep all 3, maybe slightly update wording or add a 4th minor pillar
- If existing has 5 signposts, keep them mostly intact, maybe add 1-2 new ones or update targets slightly
- Do NOT remove or majorly rewrite existing content unless it's factually contradicted

In the "changes" array, describe what minor updates were made, NOT that you've rewritten the analysis.
"""
            else:
                weighting_instruction = """
DOCUMENT WEIGHTING:
- Each document has an assigned weight percentage shown at the start
- Give MORE emphasis to higher-weighted documents when forming conclusions
- Higher-weighted documents should have more influence on the thesis, signposts, and threats
- If documents conflict, prefer the view from the higher-weighted document
"""
            
            analysis_prompt = f"""Update this existing analysis with new information from the documents.

Existing Analysis:
{json.dumps(existing_analysis, indent=2)}

{weighting_instruction}

Review the new documents and:
1. Update or confirm the investment thesis (respecting the weighting above)
2. Add any new signposts or update existing ones
3. Add any new threats or update existing ones
4. Note what has changed in the "changes" array
5. Update sources for each point based on all documents analyzed
6. Extract metadata for ALL documents (both new and from existing analysis)

DOCUMENT METADATA EXTRACTION (CRITICAL):
For EACH document (new AND previously analyzed), identify the document type and extract appropriate metadata:

**For Broker Reports:** docType="broker_report", source=Broker name, publishDate, authors=Analyst names, title
**For Earnings Calls:** docType="earnings_call", source=Company name, publishDate, quarter, title
**For SEC Filings:** docType="sec_filing", source=Company, publishDate, filingType, title
**For Emails:** docType="email", source=Sender, publishDate, title=Subject
**For Presentations:** docType="presentation", source=Company/Org, publishDate, title
**For News:** docType="news", source=Publication, publishDate, authors, title
**For Screenshots:** docType="screenshot", source=Inferred source, publishDate=if visible, title=description

For previously analyzed documents in the existing analysis, use the filenames from documentHistory and extract what metadata you can infer from the existing analysis context.

IMPORTANT STYLE RULES:
- Do NOT reference any sellside broker names (e.g., "Goldman Sachs believes...", "According to Morgan Stanley...")
- Do NOT reference specific analyst names
- Do NOT include specific broker price targets
- Write as independent analysis that synthesizes the information without attribution to sources in the prose
- The output should read like original independent research, not a summary of broker views

For each pillar, signpost, and threat, include:
- "sources": Array of source documents that support this point, with filename and a brief excerpt
- "confidence": High/Medium/Low for pillars and signposts
- Use the actual document filenames provided

Return the updated analysis as JSON with the same structure (including "documentMetadata" array), plus a "changes" array describing what minor updates were made.

Return ONLY valid JSON, no markdown, no explanation."""

        content.append({
            "type": "text",
            "text": analysis_prompt
        })
        
        result = call_llm(
            messages=[{'role': 'user', 'content': content}],
            system='You are an expert equity research analyst. Analyze documents thoroughly and provide institutional-quality investment analysis. Always respond with valid JSON only.',
            tier="standard",
            max_tokens=8192,
            timeout=180,
            anthropic_api_key=api_key,
        )
        assistant_content = result["text"]
        
        # Parse the JSON response
        try:
            cleaned = assistant_content.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1]
            if cleaned.endswith('```'):
                cleaned = cleaned.rsplit('\n', 1)[0]
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            
            analysis = json.loads(cleaned)
            changes = analysis.pop('changes', [])
            document_metadata = analysis.pop('documentMetadata', [])
            
            return jsonify({
                'analysis': analysis,
                'changes': changes,
                'documentMetadata': document_metadata,
                'usage': result["usage"]
            })

        except json.JSONDecodeError as e:
            return jsonify({
                'error': f'Failed to parse analysis: {str(e)}',
                'raw_response': assistant_content
            }), 500

    except LLMError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        import traceback
        print(f"Error in analyze-multi: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/parse', methods=['POST'])
def parse():
    """Use Claude to intelligently parse stock analysis into sections"""
    try:
        data = request.json
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('api_key', '')
        content = data.get('content', '')

        if not api_key:
            return jsonify({'error': 'No API key provided. Please add your API key in Settings.'}), 400

        result = call_llm(
            messages=[{'role': 'user', 'content': content}],
            system='You are a precise JSON extractor. Extract content into the exact JSON format requested. Return ONLY valid JSON with no markdown formatting, no code blocks, no explanation - just the raw JSON object.',
            tier="fast",
            max_tokens=4096,
            timeout=120,
            anthropic_api_key=api_key,
        )

        return jsonify({
            'response': result["text"],
            'usage': result["usage"]
        })

    except LLMError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# EMAIL ENDPOINTS
# ============================================

@app.route('/api/email', methods=['POST'])
def send_email():
    """Send email via SMTP"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = request.json
        smtp_server = data.get('smtp_server')
        smtp_port = data.get('smtp_port', 587)
        email = data.get('email')
        password = data.get('password')
        recipient = data.get('recipient')
        subject = data.get('subject')
        body = data.get('body')
        
        if not all([smtp_server, email, password, recipient, subject, body]):
            return jsonify({'error': 'Missing required email fields'}), 400
        
        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email, password)
            server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'SMTP authentication failed. Check your email and password/app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/email-overview', methods=['POST'])
def send_overview_email():
    """Send Overview email via SMTP with HTML support"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = request.json
        ticker = data.get('ticker', '')
        company_name = data.get('companyName', '')
        html_body = data.get('htmlBody', '')
        recipient = data.get('email')
        subject = data.get('customSubject', f'{ticker} - Stock Overview')
        smtp_config = data.get('smtpConfig', {})
        
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required'}), 400
        
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        plain_text = html_body.replace('<h1>', '\n').replace('</h1>', '\n' + '='*50 + '\n')
        plain_text = plain_text.replace('<h2>', '\n\n').replace('</h2>', '\n' + '-'*30 + '\n')
        plain_text = plain_text.replace('<p>', '').replace('</p>', '\n')
        plain_text = plain_text.replace('<br>', '\n').replace('<em>', '').replace('</em>', '')
        
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        if use_gmail:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Overview email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/email-analysis', methods=['POST'])
def send_analysis_email():
    """Send Analysis email via SMTP with HTML formatting"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = request.json
        analysis = data.get('analysis', {})
        recipient = data.get('email')
        smtp_config = data.get('smtpConfig', {})
        
        ticker = analysis.get('ticker', 'Stock')
        company = analysis.get('company', '')
        
        # Default subject if not provided
        default_subject = f"{ticker} - Investment Analysis"
        subject = data.get('customSubject') or default_subject
        
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required'}), 400
        
        thesis = analysis.get('thesis', {})
        signposts = analysis.get('signposts', [])
        threats = analysis.get('threats', [])
        
        # Build HTML email
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 700px;">
    <h1 style="color: #1a365d; border-bottom: 2px solid #2c5282; padding-bottom: 10px;">{ticker} - {company}</h1>
    
    <h2 style="color: #2c5282; margin-top: 25px;">1. Investment Thesis</h2>
    <p style="margin-left: 20px;">{thesis.get('summary', 'N/A')}</p>
"""
        
        if thesis.get('pillars'):
            html_body += '<ul style="margin-left: 20px;">'
            for pillar in thesis['pillars']:
                title = pillar.get('pillar', pillar.get('title', ''))
                desc = pillar.get('detail', pillar.get('description', ''))
                html_body += f'<li style="margin-bottom: 8px;"><strong>{title}:</strong> {desc}</li>'
            html_body += '</ul>'
        
        html_body += '<h2 style="color: #2c5282; margin-top: 25px;">2. Signposts (What We\'re Watching)</h2>'
        html_body += '<ul style="margin-left: 20px;">'
        for sp in signposts:
            metric = sp.get('metric', sp.get('signpost', ''))
            target = sp.get('target', '')
            timeframe = sp.get('timeframe', '')
            html_body += f'<li style="margin-bottom: 8px;"><strong>{metric}:</strong> {target}'
            if timeframe:
                html_body += f' <em>({timeframe})</em>'
            html_body += '</li>'
        html_body += '</ul>'
        
        html_body += '<h2 style="color: #2c5282; margin-top: 25px;">3. Thesis Threats (Where We Can Be Wrong)</h2>'
        html_body += '<ul style="margin-left: 20px;">'
        for threat in threats:
            threat_desc = threat.get('threat', '')
            likelihood = threat.get('likelihood', '')
            impact = threat.get('impact', '')
            triggers = threat.get('triggerPoints', '')
            html_body += f'<li style="margin-bottom: 10px;"><strong>{threat_desc}</strong>'
            if likelihood or impact:
                html_body += f'<br><span style="color: #666; font-size: 0.9em;">Likelihood: {likelihood} | Impact: {impact}</span>'
            if triggers:
                html_body += f'<br><span style="color: #666; font-size: 0.9em;">Watch for: {triggers}</span>'
            html_body += '</li>'
        html_body += '</ul>'
        
        html_body += """
</body>
</html>
"""
        
        # Plain text version
        plain_text = f"{ticker} - {company}\n\n"
        plain_text += "1. INVESTMENT THESIS\n"
        plain_text += f"{thesis.get('summary', 'N/A')}\n\n"
        
        if thesis.get('pillars'):
            for pillar in thesis['pillars']:
                title = pillar.get('pillar', pillar.get('title', ''))
                desc = pillar.get('detail', pillar.get('description', ''))
                plain_text += f"  - {title}: {desc}\n"
        
        plain_text += "\n2. SIGNPOSTS\n"
        for sp in signposts:
            metric = sp.get('metric', sp.get('signpost', ''))
            target = sp.get('target', '')
            plain_text += f"  - {metric}: {target}\n"
        
        plain_text += "\n3. THESIS THREATS\n"
        for threat in threats:
            plain_text += f"  - {threat.get('threat', '')}\n"
        
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        if use_gmail:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Analysis email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# PDF EXTRACTION ENDPOINT
# ============================================

@app.route('/api/extract-pdf', methods=['POST'])
def extract_pdf():
    """
    Extract text from uploaded PDF file.
    Used by Research tab for document analysis.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read PDF and extract text
        from PyPDF2 import PdfReader
        import io
        
        pdf_bytes = file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        text_content = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        
        full_text = '\n\n'.join(text_content)
        
        if not full_text.strip():
            return jsonify({'error': 'Could not extract text from PDF. It may be scanned/image-based.'}), 400
        
        return jsonify({
            'text': full_text,
            'pages': len(pdf_reader.pages),
            'filename': file.filename
        })
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return jsonify({'error': f'Failed to extract PDF: {str(e)}'}), 500


# ============================================
# RESEARCH ANALYSIS ENDPOINT
# ============================================

@app.route('/api/research-analyze', methods=['POST'])
def research_analyze():
    """
    Deep analysis of sell-side research using customizable prompt frameworks.
    Calls Anthropic API directly for better control and reliability.
    
    Request body:
    {
        "text": "Full prompt with document content",
        "promptId": "executive-brief",
        "promptName": "Executive Brief",
        "apiKey": "sk-ant-..." (from user's Settings)
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        prompt_id = data.get('promptId', '')
        prompt_name = data.get('promptName', '')
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('apiKey', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not api_key:
            return jsonify({'error': 'No API key provided. Please add your API key in Settings.'}), 400

        result = call_llm(
            messages=[{"role": "user", "content": text}],
            tier="standard",
            max_tokens=4096,
            anthropic_api_key=api_key,
        )

        return jsonify({
            'result': result["text"],
            'promptId': prompt_id,
            'promptName': prompt_name,
            'usage': result["usage"]
        })

    except LLMError as e:
        print(f"Research analysis LLM error: {e}")
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        print(f"Research analysis error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# RESEARCH HIERARCHICAL ENDPOINTS
# ============================================

# --- Categories ---
@app.route('/api/research-categories', methods=['GET'])
def get_research_categories():
    """Get all research categories"""
    try:
        cached = cache.get('research_categories')
        if cached is not None:
            return jsonify(cached)

        with get_db() as (_, cur):
            cur.execute('SELECT id, name, type, created_at FROM research_categories ORDER BY created_at DESC')
            rows = cur.fetchall()

        result = [{
            'id': row['id'],
            'name': row['name'],
            'type': row['type'],
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows]
        cache.set('research_categories', result, ttl=600)
        return jsonify(result)
    except Exception as e:
        print(f"Error getting research categories: {e}")
        return jsonify([])


@app.route('/api/save-research-category', methods=['POST'])
def save_research_category():
    """Save a research category"""
    try:
        data = request.json
        cat_id = data.get('id', '')
        
        if not cat_id:
            return jsonify({'error': 'Category ID is required'}), 400
        
        with get_db(commit=True) as (_, cur):
            cur.execute('''
                INSERT INTO research_categories (id, name, type)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, type = EXCLUDED.type
                RETURNING id
            ''', (cat_id, data.get('name', ''), data.get('type', 'ticker')))

        cache.invalidate('research_categories')
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving research category: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-research-category', methods=['POST'])
def delete_research_category():
    """Delete a research category and all its documents/analyses"""
    try:
        data = request.json
        cat_id = data.get('id', '')

        if not cat_id:
            return jsonify({'error': 'Category ID is required'}), 400

        with get_db(commit=True) as (_, cur):
            cur.execute('DELETE FROM research_categories WHERE id = %s', (cat_id,))

        cache.invalidate('research_categories')
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research category: {e}")
        return jsonify({'error': str(e)}), 500


# --- Documents ---
@app.route('/api/research-documents', methods=['GET'])
def get_research_documents():
    """Get all research documents"""
    try:
        with get_db() as (_, cur):
            cur.execute('SELECT id, category_id, name, content, file_names, smart_name, original_filename, published_date, doc_type, has_stored_files, created_at FROM research_documents ORDER BY created_at DESC')
            rows = cur.fetchall()

        return jsonify([{
            'id': row['id'],
            'categoryId': row['category_id'],
            'name': row['name'],
            'content': row['content'],
            'fileNames': row['file_names'] or [],
            'smartName': row['smart_name'],
            'originalFilename': row['original_filename'],
            'publishedDate': row['published_date'],
            'docType': row.get('doc_type') or 'other',
            'hasStoredFiles': row['has_stored_files'] or False,
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows])
    except Exception as e:
        print(f"Error getting research documents: {e}")
        return jsonify([])


@app.route('/api/save-research-document', methods=['POST'])
def save_research_document():
    """Save a research document"""
    try:
        data = request.json
        doc_id = data.get('id', '')
        
        print(f"📄 save_research_document: id={doc_id}, name={data.get('name', '')[:50]}")
        
        if not doc_id:
            return jsonify({'error': 'Document ID is required'}), 400
        
        with get_db(commit=True) as (_, cur):
            cur.execute('''
                INSERT INTO research_documents (id, category_id, name, content, file_names, smart_name, original_filename, published_date, doc_type, has_stored_files)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    content = EXCLUDED.content,
                    file_names = EXCLUDED.file_names,
                    smart_name = EXCLUDED.smart_name,
                    original_filename = EXCLUDED.original_filename,
                    published_date = EXCLUDED.published_date,
                    doc_type = EXCLUDED.doc_type,
                    has_stored_files = EXCLUDED.has_stored_files
                RETURNING id
            ''', (
                doc_id,
                data.get('categoryId', ''),
                data.get('name', ''),
                data.get('content', ''),
                json.dumps(data.get('fileNames', [])),
                data.get('smartName'),
                data.get('originalFilename'),
                data.get('publishedDate'),
                data.get('docType', 'other'),
                data.get('hasStoredFiles', False)
            ))

        print(f"✅ Document saved: {doc_id}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Error saving research document: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-research-file', methods=['POST'])
def save_research_file():
    """Save a file for a research document"""
    try:
        data = request.json
        if not data:
            print("❌ No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400

        document_id = data.get('documentId', '')
        filename = data.get('filename', '')
        file_type = data.get('fileType', '')
        file_data = data.get('fileData', '')
        file_size = data.get('fileSize', 0)

        print(f"📁 save_research_file: docId={document_id}, filename={filename}, fileType={file_type}, dataLen={len(file_data) if file_data else 0}, fileSize={file_size}")

        if not document_id or not filename:
            print(f"❌ Missing required fields: docId={document_id}, filename={filename}")
            return jsonify({'error': 'Document ID and filename are required'}), 400

        if not file_data:
            print(f"❌ No file data provided for {filename}")
            return jsonify({'error': 'No file data provided'}), 400

        with get_db(commit=True) as (conn, cur):
            # Check if document exists first
            cur.execute('SELECT id FROM research_documents WHERE id = %s', (document_id,))
            doc_exists = cur.fetchone()
            if not doc_exists:
                print(f"❌ Document {document_id} does not exist in research_documents table")
                return jsonify({'error': f'Document {document_id} not found - must save document first'}), 400

            print(f"✅ Document {document_id} exists, proceeding with file save")

            cur.execute('''
                INSERT INTO research_document_files (document_id, filename, file_type, file_data, file_size)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            ''', (document_id, filename, file_type, file_data, file_size))

            result = cur.fetchone()
            if result is None:
                print(f"❌ INSERT did not return an id")
                conn.rollback()
                return jsonify({'error': 'Insert failed - no id returned'}), 500

            inserted_id = result['id']

        print(f"✅ File saved successfully: id={inserted_id}, filename={filename}")
        return jsonify({'success': True, 'id': inserted_id})
    except Exception as e:
        print(f"❌ Error saving research file: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500


@app.route('/api/research-document-files/<document_id>', methods=['GET'])
def get_research_document_files(document_id):
    """Get stored files for a research document"""
    try:
        with get_db() as (_, cur):
            cur.execute('''
                SELECT id, filename, file_type, file_data, file_size, created_at
                FROM research_document_files
                WHERE document_id = %s
                ORDER BY created_at
            ''', (document_id,))
            rows = cur.fetchall()

        return jsonify([{
            'id': row['id'],
            'filename': row['filename'],
            'fileType': row['file_type'],
            'fileData': row['file_data'],
            'fileSize': row['file_size'],
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows])
    except Exception as e:
        print(f"Error getting research document files: {e}")
        return jsonify([])


@app.route('/api/delete-research-file/<int:file_id>', methods=['DELETE'])
def delete_research_file(file_id):
    """Delete a stored research file"""
    try:
        with get_db(commit=True) as (_, cur):
            cur.execute('DELETE FROM research_document_files WHERE id = %s', (file_id,))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-research-document', methods=['POST'])
def delete_research_document():
    """Delete a research document and all its analyses"""
    try:
        data = request.json
        doc_id = data.get('id', '')
        
        if not doc_id:
            return jsonify({'error': 'Document ID is required'}), 400
        
        with get_db(commit=True) as (_, cur):
            # CASCADE will delete analyses
            cur.execute('DELETE FROM research_documents WHERE id = %s', (doc_id,))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research document: {e}")
        return jsonify({'error': str(e)}), 500


# --- Analyses ---
@app.route('/api/research-analyses', methods=['GET'])
def get_research_analyses():
    """Get all research analyses"""
    try:
        with get_db() as (_, cur):
            cur.execute('SELECT id, document_id, prompt_id, prompt_name, prompt_icon, result, usage, created_at FROM research_analyses ORDER BY created_at DESC')
            rows = cur.fetchall()

        return jsonify([{
            'id': row['id'],
            'documentId': row['document_id'],
            'promptId': row['prompt_id'],
            'promptName': row['prompt_name'],
            'promptIcon': row['prompt_icon'],
            'result': row['result'],
            'usage': row['usage'],
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows])
    except Exception as e:
        print(f"Error getting research analyses: {e}")
        return jsonify([])


@app.route('/api/save-research-analysis', methods=['POST'])
def save_research_analysis():
    """Save a research analysis"""
    try:
        data = request.json
        analysis_id = data.get('id', '')
        
        if not analysis_id:
            return jsonify({'error': 'Analysis ID is required'}), 400
        
        with get_db(commit=True) as (_, cur):
            cur.execute('''
                INSERT INTO research_analyses (id, document_id, prompt_id, prompt_name, prompt_icon, result, usage)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    result = EXCLUDED.result,
                    usage = EXCLUDED.usage
                RETURNING id
            ''', (
                analysis_id,
                data.get('documentId', ''),
                data.get('promptId', ''),
                data.get('promptName', ''),
                data.get('promptIcon', ''),
                data.get('result', ''),
                json.dumps(data.get('usage', {}))
            ))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving research analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-research-analysis', methods=['POST'])
def delete_research_analysis():
    """Delete a research analysis"""
    try:
        data = request.json
        analysis_id = data.get('id', '')
        
        if not analysis_id:
            return jsonify({'error': 'Analysis ID is required'}), 400
        
        with get_db(commit=True) as (_, cur):
            cur.execute('DELETE FROM research_analyses WHERE id = %s', (analysis_id,))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/email-research', methods=['POST'])
def email_research():
    """Email a research result"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import markdown
    import re
    
    def preprocess_bullets(text):
        """Convert bullet characters to standard markdown format"""
        # Split into lines for processing
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Handle lines that start with bullet characters (with optional whitespace)
            line = re.sub(r'^(\s*)[•·▪▸►‣⁃]\s*', r'\1- ', line)
            
            # Handle inline bullets (mid-line) - add newline before them
            # This catches patterns like "text • more text"
            line = re.sub(r'\s+[•·▪▸►‣⁃]\s+', '\n- ', line)
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    try:
        data = request.json
        recipient = data.get('email', '')
        subject = data.get('subject', 'Research Analysis')
        content = data.get('content', '')
        prompt_name = data.get('promptName', '')
        ticker = data.get('ticker', '')
        topic = data.get('topic', '')
        smtp_config = data.get('smtpConfig', {})
        minimal = data.get('minimal', False)

        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400

        # Preprocess bullet characters before markdown conversion
        processed_content = preprocess_bullets(content)

        # Convert markdown to HTML with nl2br for line break preservation
        try:
            content_html = markdown.markdown(
                processed_content,
                extensions=['tables', 'fenced_code', 'nl2br']
            )
        except:
            content_html = f"<pre style='white-space: pre-wrap;'>{content}</pre>"

        if minimal:
            # Clean email — just content with basic styling, no header/footer
            html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #1a1a2e; margin-top: 1.5em; margin-bottom: 0.5em; }}
                ul {{ margin: 10px 0; padding-left: 25px; }}
                li {{ margin-bottom: 8px; line-height: 1.5; }}
                ul ul {{ margin-top: 8px; }}
                p {{ margin: 0.8em 0; }}
                strong {{ color: #1e293b; }}
                hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 1.5em 0; }}
            </style>
        </head>
        <body>
            {content_html}
        </body>
        </html>
            """
        else:
            # Full decorated email with header/footer
            header_info = []
            if ticker:
                header_info.append(f"<strong>Ticker:</strong> {ticker}")
            if topic:
                header_info.append(f"<strong>Topic:</strong> {topic}")
            if prompt_name:
                header_info.append(f"<strong>Framework:</strong> {prompt_name}")
            header_html = " | ".join(header_info) if header_info else ""

            html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #1a1a2e; margin-top: 1.5em; margin-bottom: 0.5em; }}
                .header {{ background: linear-gradient(135deg, #0f172a, #1e293b); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .header h1 {{ margin: 0 0 10px 0; color: white; }}
                .header-meta {{ font-size: 14px; opacity: 0.9; }}
                .content {{ background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; }}
                ul {{ margin: 10px 0; padding-left: 25px; }}
                li {{ margin-bottom: 8px; line-height: 1.5; }}
                ul ul {{ margin-top: 8px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
                th {{ background: #f1f5f9; }}
                code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 14px; }}
                pre {{ background: #1e293b; color: #e2e8f0; padding: 15px; border-radius: 8px; overflow-x: auto; }}
                p {{ margin: 0.8em 0; }}
                strong {{ color: #1e293b; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Research Analysis</h1>
                <div class="header-meta">{header_html}</div>
            </div>
            <div class="content">
                {content_html}
            </div>
        </body>
        </html>
            """
        
        # Send email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['To'] = recipient
        
        # Plain text version
        msg.attach(MIMEText(content, 'plain'))
        # HTML version
        msg.attach(MIMEText(html_body, 'html'))
        
        if smtp_config.get('use_gmail') and smtp_config.get('gmail_user') and smtp_config.get('gmail_app_password'):
            msg['From'] = smtp_config['gmail_user']
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(smtp_config['gmail_user'], smtp_config['gmail_app_password'])
                server.send_message(msg)
        else:
            return jsonify({'error': 'Email not configured. Please set up Gmail in Settings.'}), 400
        
        return jsonify({'success': True, 'message': 'Research email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        print(f"Error sending research email: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# MEETING PREP - HELPERS & PROMPTS
# ============================================

import re as _re

def parse_mp_json(text):
    """Parse JSON from AI response, handling markdown fencing and truncation."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for sc, ec in [("[", "]"), ("{", "}")]:
        s = text.find(sc)
        e = text.rfind(ec)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    # Try to repair truncated JSON by closing open brackets/braces
    for sc, ec in [("{", "}"), ("[", "]")]:
        s = text.find(sc)
        if s != -1:
            fragment = text[s:]
            # Count open vs close brackets
            for attempt in range(5):
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    # Try closing with appropriate bracket
                    open_braces = fragment.count('{') - fragment.count('}')
                    open_brackets = fragment.count('[') - fragment.count(']')
                    # Remove trailing partial content after last comma or complete value
                    last_comma = fragment.rfind(',')
                    if last_comma > 0:
                        fragment = fragment[:last_comma]
                    fragment += '}' * max(0, open_braces) + ']' * max(0, open_brackets)
                    try:
                        return json.loads(fragment)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not parse JSON from response: {text[:300]}")


MP_DOC_TYPE_PATTERNS = {
    "earnings_transcript": [
        r"earnings\s*(call|transcript)",
        r"q[1-4]\s*\d{4}\s*(call|transcript|results)",
        r"(quarterly|annual)\s*results\s*call",
    ],
    "conference_transcript": [
        r"conference\s*(transcript|presentation)",
        r"investor\s*(day|conference|presentation)",
        r"fireside\s*chat",
    ],
    "broker_report": [
        r"(initiat|maintain|reiterat|upgrad|downgrad|price\s*target)",
        r"(buy|sell|hold|overweight|underweight|neutral|outperform)\s*(rating)?",
        r"(equity\s*research|research\s*report|analyst\s*note)",
    ],
    "press_release": [
        r"press\s*release",
        r"(announces|reports)\s*(q[1-4]|quarterly|annual|full.year)",
    ],
    "filing": [
        r"(10-[kq]|8-k|def\s*14a|proxy|annual\s*report)",
        r"securities\s*and\s*exchange\s*commission",
    ],
}

def classify_mp_document(filename, text_sample=""):
    """Auto-classify document type from filename and first 2000 chars of content."""
    combined = f"{filename} {text_sample[:2000]}".lower()
    for doc_type, patterns in MP_DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if _re.search(pattern, combined, _re.IGNORECASE):
                return doc_type
    return "other"


MP_ANALYSIS_PROMPT = """You are a senior equity research analyst assistant preparing for a management meeting.

Analyze the following {doc_type} for {ticker} ({company_name}).

Extract and organize the following into a structured JSON response:

1. **key_metrics**: Array of objects with {{metric, value, change, period}} — revenue, margins, EPS, guidance, segment data, KPIs. Be specific with numbers.
2. **management_claims**: Array of strings — specific commitments, promises, strategic statements management made. Quote where possible.
3. **guidance_changes**: Array of objects with {{metric, old_guidance, new_guidance, direction}} — any changes to forward guidance.
4. **risks_concerns**: Array of strings — risks flagged by management, analysts, or evident from data.
5. **catalysts**: Array of strings — upcoming events, product launches, regulatory decisions, etc. that could move the stock.
6. **contradictions**: Array of strings — anything that contradicts prior statements, guidance, or consensus.
7. **notable_quotes**: Array of objects with {{quote, speaker, context}} — important verbatim quotes worth referencing.
8. **key_numbers**: Object — the most important 5-10 data points someone should know before a meeting.

Return ONLY valid JSON, no markdown fencing."""

MP_SYNTHESIS_PROMPT = """You are preparing a senior equity research analyst for a management meeting with {ticker} ({company_name}, {sector} sector).

You have analyses of {doc_count} documents spanning the {timeframe} period. Your task is to synthesize these into a coherent picture that identifies what the analyst MUST explore in the meeting.

{analyses_text}

{past_questions_text}

Synthesize into a JSON object with:

1. **narrative_arc**: 3-5 sentences describing the story across these documents — what's the trajectory? What's changed?
2. **key_themes**: Array of objects with {{theme, description, supporting_evidence}} — the 4-7 major themes emerging.
3. **contradictions**: Array of objects with {{claim_1, source_1, claim_2, source_2, significance}} — where documents or statements conflict.
4. **information_gaps**: Array of strings — what's NOT being discussed that should be? What data is missing?
5. **tone_shifts**: Array of objects with {{topic, old_tone, new_tone, source}} — where management messaging has shifted.
6. **unresolved_from_prior**: Array of objects with {{question, original_date, why_still_relevant}} — past questions that need follow-up.
7. **consensus_vs_reality**: Array of strings — where consensus expectations seem misaligned with what documents reveal.

Return ONLY valid JSON, no markdown fencing."""

MP_QUESTION_PROMPT = """You are helping a senior equity research analyst prepare 25-30 sophisticated questions for a management meeting with {ticker} ({company_name}, {sector} sector).

Context from document synthesis:
{synthesis_text}

CRITICAL RULES FOR QUESTION TEXT:
- Questions must sound 100% proprietary — as if the analyst developed them entirely from their own deep, independent research on the company
- The ONLY acceptable references in question text are: the company's own filings, earnings calls, press releases, management's own statements/quotes, and publicly reported financial data
- NEVER reference or allude to sell-side in ANY form. This includes:
  - Direct: "Wells Fargo notes...", "according to Deutsche Bank...", "Jefferies estimates..."
  - Indirect: "analysts are cutting estimates", "the Street expects", "consensus is...", "the investment community", "market participants", "some observers note", "there's skepticism among investors"
- Instead, ground every question in the company's OWN words, data, and disclosures: "You guided to X but reported Y", "Your 10-K shows margin compression from Z to W", "On the Q3 call you said X, but Q4 results suggest otherwise"
- The analyst's edge comes from connecting dots across the company's own disclosures — not from citing what other analysts think
- The "source" and "context" fields ARE where you indicate which broker report or document the question was derived from — that metadata is for the analyst's private reference only, never surfaced in the question

Generate questions that are:
- **Proprietary-sounding**: Every question reads as if the analyst personally identified the issue through their own deep research
- **Specific**: Reference concrete data points, quotes, or metrics — but attribute them to the company's own disclosures, not to broker commentary
- **Probing**: Push management beyond their prepared talking points — ask about inconsistencies, gaps, and changes
- **Organized**: Group by dynamically determined topics relevant to THIS company/sector (NOT a generic template)
- **Prioritized**: Mark each as high (must-ask, 8-10 questions), medium (important, 10-12 questions), or low (if time permits, 5-8 questions)
- **Strategic**: Include questions about capital allocation, competitive dynamics, and forward catalysts

For each question, also provide:
- **context**: Why this question matters — what data point or observation prompted it. THIS is where you can reference the sell-side source for the analyst's private notes (1-2 sentences)
- **source**: Which document(s) the question draws from (e.g., "Wells Fargo report, p.3; Q4 Earnings Transcript, p.12"). This is private metadata for the analyst.
- **follow_up_angle**: What to ask if management gives an evasive or generic answer

{unresolved_text}

Return a JSON array of topic groups:
[
  {{
    "topic": "Revenue Growth Trajectory",
    "description": "Brief description of why this topic matters for this company",
    "questions": [
      {{
        "question": "Proprietary-sounding question with NO broker/analyst attribution",
        "context": "Private note: sourced from Wells Fargo report highlighting X — important because Y",
        "source": "Wells Fargo report p.3, Q4 Earnings Transcript p.12",
        "priority": "high",
        "follow_up_angle": "If they deflect, ask about..."
      }}
    ]
  }}
]

Return ONLY valid JSON, no markdown fencing."""


# ============================================
# MEETING PREP - MEETING ENDPOINTS
# ============================================

@app.route('/api/mp/meetings', methods=['POST'])
def mp_create_meeting():
    """Create a new meeting prep session."""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper().strip()
        company_name = data.get('companyName', '')
        sector = data.get('sector', '')
        meeting_date = data.get('meetingDate') or None
        meeting_type = data.get('meetingType', 'other')
        notes = data.get('notes', '')

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        with get_db(commit=True) as (_, cur):
            # Upsert company
            cur.execute('''
                INSERT INTO mp_companies (ticker, name, sector)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    name = COALESCE(NULLIF(EXCLUDED.name, ''), mp_companies.name),
                    sector = COALESCE(NULLIF(EXCLUDED.sector, ''), mp_companies.sector)
                RETURNING id, ticker, name, sector
            ''', (ticker, company_name, sector))
            company = dict(cur.fetchone())

            # Create meeting
            cur.execute('''
                INSERT INTO mp_meetings (company_id, meeting_date, meeting_type, notes)
                VALUES (%s, %s, %s, %s)
                RETURNING id, company_id, meeting_date, meeting_type, status, notes, created_at, updated_at
            ''', (company['id'], meeting_date, meeting_type, notes))
            meeting = dict(cur.fetchone())
            meeting['ticker'] = company['ticker']
            meeting['company_name'] = company['name']
            meeting['sector'] = company['sector']

        return jsonify(meeting)
    except Exception as e:
        print(f"Error creating meeting: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/meetings', methods=['GET'])
def mp_list_meetings():
    """List all meeting prep sessions."""
    try:
        with get_db() as (_, cur):
            cur.execute('''
                SELECT m.*, c.ticker, c.name as company_name, c.sector,
                       (SELECT COUNT(*) FROM mp_documents WHERE meeting_id = m.id) as doc_count,
                       (SELECT COUNT(*) FROM mp_question_sets WHERE meeting_id = m.id AND status = 'ready') as qs_count
                FROM mp_meetings m
                JOIN mp_companies c ON m.company_id = c.id
                ORDER BY m.created_at DESC
            ''')
            rows = cur.fetchall()

        result = []
        for r in rows:
            result.append({
                'id': r['id'],
                'company_id': r['company_id'],
                'ticker': r['ticker'],
                'company_name': r['company_name'],
                'sector': r['sector'],
                'meeting_date': str(r['meeting_date']) if r['meeting_date'] else None,
                'meeting_type': r['meeting_type'],
                'status': r['status'],
                'notes': r['notes'],
                'doc_count': r['doc_count'],
                'qs_count': r['qs_count'],
                'created_at': r['created_at'].isoformat() if r['created_at'] else None,
            })

        return jsonify(result)
    except Exception as e:
        print(f"Error listing meetings: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/meetings/<int:meeting_id>', methods=['GET'])
def mp_get_meeting(meeting_id):
    """Get a meeting with its documents and latest question set."""
    try:
        with get_db() as (_, cur):
            # Get meeting
            cur.execute('''
                SELECT m.*, c.ticker, c.name as company_name, c.sector
                FROM mp_meetings m
                JOIN mp_companies c ON m.company_id = c.id
                WHERE m.id = %s
            ''', (meeting_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({'error': 'Meeting not found'}), 404

            meeting = dict(row)
            meeting['meeting_date'] = str(meeting['meeting_date']) if meeting['meeting_date'] else None
            meeting['created_at'] = meeting['created_at'].isoformat() if meeting['created_at'] else None
            meeting['updated_at'] = meeting['updated_at'].isoformat() if meeting['updated_at'] else None

            # Get documents (without file_data to keep response small)
            cur.execute('''
                SELECT id, meeting_id, filename, doc_type, doc_date, page_count, token_estimate,
                       upload_order, file_size, created_at
                FROM mp_documents WHERE meeting_id = %s ORDER BY upload_order
            ''', (meeting_id,))
            docs = []
            for d in cur.fetchall():
                dd = dict(d)
                dd['created_at'] = dd['created_at'].isoformat() if dd['created_at'] else None
                docs.append(dd)

            # Get latest question set
            cur.execute('''
                SELECT * FROM mp_question_sets
                WHERE meeting_id = %s ORDER BY version DESC LIMIT 1
            ''', (meeting_id,))
            qs_row = cur.fetchone()
            question_set = None
            if qs_row:
                question_set = dict(qs_row)
                if question_set['topics_json']:
                    question_set['topics'] = json.loads(question_set['topics_json'])
                else:
                    question_set['topics'] = []
                question_set['created_at'] = question_set['created_at'].isoformat() if question_set['created_at'] else None

        return jsonify({
            'meeting': meeting,
            'documents': docs,
            'questionSet': question_set,
        })
    except Exception as e:
        print(f"Error getting meeting: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/meetings/<int:meeting_id>', methods=['DELETE'])
def mp_delete_meeting(meeting_id):
    """Delete a meeting and all related data."""
    try:
        with get_db(commit=True) as (_, cur):
            cur.execute('DELETE FROM mp_meetings WHERE id = %s', (meeting_id,))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting meeting: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# MEETING PREP - DOCUMENT ENDPOINTS
# ============================================

@app.route('/api/mp/meetings/<int:meeting_id>/documents', methods=['POST'])
def mp_upload_documents(meeting_id):
    """Upload PDF documents for a meeting. Expects JSON with base64 file data."""
    try:
        from PyPDF2 import PdfReader
        import io

        data = request.json
        documents = data.get('documents', [])

        if not documents:
            return jsonify({'error': 'No documents provided'}), 400

        with get_db(commit=True) as (_, cur):
            # Verify meeting exists
            cur.execute('SELECT id FROM mp_meetings WHERE id = %s', (meeting_id,))
            if not cur.fetchone():
                return jsonify({'error': 'Meeting not found'}), 404

            # Get current max upload_order
            cur.execute('SELECT COALESCE(MAX(upload_order), 0) AS max_order FROM mp_documents WHERE meeting_id = %s', (meeting_id,))
            order = cur.fetchone()['max_order']

            results = []
            for doc in documents:
                order += 1
                filename = doc.get('filename', 'unknown.pdf')
                file_data = doc.get('fileData', '')
                extracted_text = doc.get('extractedText', '')
                page_count = doc.get('pageCount')

                # If no extracted text provided, try extracting from base64 PDF
                if not extracted_text and file_data:
                    try:
                        pdf_bytes = base64.b64decode(file_data)
                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        pages = []
                        for page in reader.pages:
                            t = page.extract_text()
                            if t:
                                pages.append(t)
                        extracted_text = '\n\n'.join(pages)
                        if page_count is None:
                            page_count = len(reader.pages)
                    except Exception as ex:
                        print(f"PDF extraction error for {filename}: {ex}")

                # Classify and estimate tokens
                doc_type = classify_mp_document(filename, extracted_text)
                token_estimate = len(extracted_text) // 4 if extracted_text else 0
                file_size = len(file_data) * 3 // 4 if file_data else 0

                cur.execute('''
                    INSERT INTO mp_documents (meeting_id, filename, file_data, doc_type, doc_date,
                        page_count, token_estimate, extracted_text, upload_order, file_size)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, filename, doc_type, doc_date, page_count, token_estimate, upload_order, file_size, created_at
                ''', (meeting_id, filename, file_data, doc_type, doc.get('docDate'),
                      page_count, token_estimate, extracted_text, order, file_size))
                row = dict(cur.fetchone())
                row['created_at'] = row['created_at'].isoformat() if row['created_at'] else None
                results.append(row)

        return jsonify(results)
    except Exception as e:
        print(f"Error uploading documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/meetings/<int:meeting_id>/documents/<int:doc_id>', methods=['DELETE'])
def mp_delete_document(meeting_id, doc_id):
    """Delete a document from a meeting."""
    try:
        with get_db(commit=True) as (_, cur):
            cur.execute('DELETE FROM mp_documents WHERE id = %s AND meeting_id = %s', (doc_id, meeting_id))

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# MEETING PREP - PIPELINE ENDPOINTS
# ============================================

@app.route('/api/mp/analyze-document', methods=['POST'])
def mp_analyze_document():
    """Step 1: Analyze a single document. Uses streaming to avoid Render timeout."""
    try:
        data = request.json
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('apiKey', '')
        if not api_key:
            return jsonify({'error': 'No API key provided. Please add your API key in Settings.'}), 400

        ticker = data.get('ticker', '')
        company_name = data.get('companyName', ticker)
        doc_type = data.get('docType', 'document')
        filename = data.get('filename', '')
        extracted_text = data.get('extractedText', '')

        if not extracted_text:
            return jsonify({'error': 'No document text provided'}), 400

        if len(extracted_text) > 400000:
            extracted_text = extracted_text[:400000] + "\n\n[... document truncated for length ...]"

        prompt = MP_ANALYSIS_PROMPT.format(
            doc_type=doc_type, ticker=ticker, company_name=company_name
        )
        user_msg = f"Document: {filename}\n\n{extracted_text}"

        def generate():
            try:
                llm_result = None
                for chunk in call_llm_stream(
                    messages=[{"role": "user", "content": user_msg}],
                    system=prompt,
                    tier="standard",
                    max_tokens=16384,
                    anthropic_api_key=api_key,
                ):
                    if isinstance(chunk, dict):
                        llm_result = chunk
                    else:
                        yield chunk

                if llm_result is None:
                    raise Exception("No result from LLM")

                analysis = parse_mp_json(llm_result["text"])
                tokens_used = llm_result["usage"]["input_tokens"] + llm_result["usage"]["output_tokens"]
                yield "\n" + json.dumps({'analysis': analysis, 'tokensUsed': tokens_used, 'filename': filename})
            except LLMError as e:
                print(f"MP analyze LLM error: {e}")
                yield "\n" + json.dumps({'error': str(e)})
            except Exception as e:
                print(f"MP analyze stream error: {e}")
                yield "\n" + json.dumps({'error': str(e)})

        return app.response_class(generate(), mimetype='text/plain')

    except Exception as e:
        print(f"MP analyze error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/synthesize', methods=['POST'])
def mp_synthesize():
    """Step 2: Cross-reference all document analyses. Uses streaming to avoid timeout."""
    try:
        data = request.json
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('apiKey', '')
        if not api_key:
            return jsonify({'error': 'No API key provided.'}), 400

        ticker = data.get('ticker', '')
        company_name = data.get('companyName', ticker)
        sector = data.get('sector', 'unknown')
        analyses = data.get('analyses', [])
        past_questions = data.get('pastQuestions', [])
        timeframe = data.get('timeframe', 'recent')

        if not analyses:
            return jsonify({'error': 'No analyses provided'}), 400

        analyses_parts = []
        for i, a in enumerate(analyses):
            analyses_parts.append(f"### Document {i+1}: {a.get('_source_filename', 'unknown')}\n{json.dumps(a, indent=2)}")
        analyses_text = "\n\n".join(analyses_parts)

        past_q_text = ""
        if past_questions:
            pq_items = []
            for pq in past_questions[:30]:
                status_note = f" [STATUS: {pq.get('status', '')}]" if pq.get('status') != 'asked' else ""
                response = f" — Response: {pq.get('response_notes', '')}" if pq.get('response_notes') else ""
                pq_items.append(f"- [{pq.get('meeting_date', '?')}] {pq.get('question', '')}{status_note}{response}")
            past_q_text = "PAST QUESTIONS FROM PRIOR MEETINGS (reference these and flag unresolved items):\n" + "\n".join(pq_items)

        prompt = MP_SYNTHESIS_PROMPT.format(
            ticker=ticker, company_name=company_name, sector=sector,
            doc_count=len(analyses), timeframe=timeframe,
            analyses_text=analyses_text, past_questions_text=past_q_text,
        )

        def generate():
            try:
                llm_result = None
                for chunk in call_llm_stream(
                    messages=[{"role": "user", "content": "Synthesize the above document analyses."}],
                    system=prompt,
                    tier="standard",
                    max_tokens=16384,
                    anthropic_api_key=api_key,
                ):
                    if isinstance(chunk, dict):
                        llm_result = chunk
                    else:
                        yield chunk

                if llm_result is None:
                    raise Exception("No result from LLM")

                synthesis = parse_mp_json(llm_result["text"])
                tokens_used = llm_result["usage"]["input_tokens"] + llm_result["usage"]["output_tokens"]
                yield "\n" + json.dumps({'synthesis': synthesis, 'tokensUsed': tokens_used})
            except LLMError as e:
                print(f"MP synthesize LLM error: {e}")
                yield "\n" + json.dumps({'error': str(e)})
            except Exception as e:
                print(f"MP synthesize stream error: {e}")
                yield "\n" + json.dumps({'error': str(e)})

        return app.response_class(generate(), mimetype='text/plain')

    except Exception as e:
        print(f"MP synthesize error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/generate-questions', methods=['POST'])
def mp_generate_questions():
    """Step 3: Generate questions from synthesis. Uses streaming to avoid timeout."""
    try:
        data = request.json
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('apiKey', '')
        if not api_key:
            return jsonify({'error': 'No API key provided.'}), 400

        ticker = data.get('ticker', '')
        company_name = data.get('companyName', ticker)
        sector = data.get('sector', 'unknown')
        synthesis = data.get('synthesis', {})
        unresolved = data.get('unresolvedQuestions', [])

        if not synthesis:
            return jsonify({'error': 'No synthesis provided'}), 400

        synthesis_text = json.dumps(synthesis, indent=2)

        unresolved_text = ""
        if unresolved:
            items = [f"- {q.get('question', '')} (from {q.get('meeting_date', '?')})" for q in unresolved[:15]]
            unresolved_text = "UNRESOLVED QUESTIONS FROM PRIOR MEETINGS (include follow-ups for these):\n" + "\n".join(items)

        prompt = MP_QUESTION_PROMPT.format(
            ticker=ticker, company_name=company_name, sector=sector,
            synthesis_text=synthesis_text, unresolved_text=unresolved_text,
        )

        def generate():
            try:
                llm_result = None
                for chunk in call_llm_stream(
                    messages=[{"role": "user", "content": "Generate the meeting preparation questions."}],
                    system=prompt,
                    tier="standard",
                    max_tokens=16384,
                    anthropic_api_key=api_key,
                ):
                    if isinstance(chunk, dict):
                        llm_result = chunk
                    else:
                        yield chunk

                if llm_result is None:
                    raise Exception("No result from LLM")

                topics = parse_mp_json(llm_result["text"])
                tokens_used = llm_result["usage"]["input_tokens"] + llm_result["usage"]["output_tokens"]
                yield "\n" + json.dumps({'topics': topics, 'tokensUsed': tokens_used})
            except LLMError as e:
                print(f"MP generate questions LLM error: {e}")
                yield "\n" + json.dumps({'error': str(e)})
            except Exception as e:
                print(f"MP generate questions stream error: {e}")
                yield "\n" + json.dumps({'error': str(e)})

        return app.response_class(generate(), mimetype='text/plain')

    except Exception as e:
        print(f"MP generate questions error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/save-results', methods=['POST'])
def mp_save_results():
    """Save pipeline results: question set + past questions."""
    try:
        data = request.json
        meeting_id = data.get('meetingId')
        topics = data.get('topics', [])
        synthesis_json = data.get('synthesisJson')
        total_tokens = data.get('totalTokens', 0)
        model = data.get('model', 'claude-sonnet-4-5-20250929')

        if not meeting_id:
            return jsonify({'error': 'meetingId is required'}), 400

        with get_db(commit=True) as (_, cur):
            # Get next version
            cur.execute('SELECT COALESCE(MAX(version), 0) + 1 AS next_ver FROM mp_question_sets WHERE meeting_id = %s', (meeting_id,))
            version = cur.fetchone()['next_ver']

            # Insert question set
            cur.execute('''
                INSERT INTO mp_question_sets (meeting_id, version, status, topics_json, synthesis_json, generation_model, generation_tokens)
                VALUES (%s, %s, 'ready', %s, %s, %s, %s)
                RETURNING id, version
            ''', (meeting_id, version, json.dumps(topics), json.dumps(synthesis_json) if synthesis_json else None, model, total_tokens))
            qs = dict(cur.fetchone())

            # Update meeting status
            cur.execute("UPDATE mp_meetings SET status = 'ready', updated_at = CURRENT_TIMESTAMP WHERE id = %s", (meeting_id,))

            # Save questions to past_questions
            cur.execute('SELECT company_id FROM mp_meetings WHERE id = %s', (meeting_id,))
            company_row = cur.fetchone()
            if company_row:
                company_id = company_row['company_id']
                for topic in (topics if isinstance(topics, list) else []):
                    topic_name = topic.get('topic', '') if isinstance(topic, dict) else ''
                    questions = topic.get('questions', []) if isinstance(topic, dict) else []
                    for q in questions:
                        q_text = q.get('question', '') if isinstance(q, dict) else ''
                        if q_text:
                            cur.execute('''
                                INSERT INTO mp_past_questions (company_id, meeting_id, question, topic, status)
                                VALUES (%s, %s, %s, %s, 'asked')
                            ''', (company_id, meeting_id, q_text, topic_name))

        return jsonify({'questionSetId': qs['id'], 'version': qs['version']})
    except Exception as e:
        print(f"Error saving results: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# MEETING PREP - HISTORY ENDPOINTS
# ============================================

@app.route('/api/mp/companies/<ticker>/past-questions', methods=['GET'])
def mp_get_past_questions(ticker):
    """Get past questions for a company."""
    try:
        with get_db() as (_, cur):
            cur.execute('SELECT * FROM mp_companies WHERE ticker = %s', (ticker.upper(),))
            company = cur.fetchone()
            if not company:
                return jsonify({'company': None, 'pastQuestions': []})

            cur.execute('''
                SELECT pq.*, m.meeting_date, m.meeting_type
                FROM mp_past_questions pq
                LEFT JOIN mp_meetings m ON pq.meeting_id = m.id
                WHERE pq.company_id = %s
                ORDER BY pq.created_at DESC
                LIMIT 100
            ''', (company['id'],))
            rows = cur.fetchall()

        pqs = []
        for r in rows:
            d = dict(r)
            d['created_at'] = d['created_at'].isoformat() if d['created_at'] else None
            d['meeting_date'] = str(d['meeting_date']) if d['meeting_date'] else None
            pqs.append(d)

        return jsonify({'company': dict(company), 'pastQuestions': pqs})
    except Exception as e:
        print(f"Error getting past questions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/past-questions/<int:pq_id>/note', methods=['POST'])
def mp_update_past_question(pq_id):
    """Update notes/status on a past question."""
    try:
        data = request.json

        updates = []
        params = []
        if 'responseNotes' in data:
            updates.append('response_notes = %s')
            params.append(data['responseNotes'])
        if 'status' in data:
            updates.append('status = %s')
            params.append(data['status'])

        if updates:
            params.append(pq_id)
            with get_db(commit=True) as (_, cur):
                cur.execute(f"UPDATE mp_past_questions SET {', '.join(updates)} WHERE id = %s", params)

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating past question: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mp/meetings/<int:meeting_id>/documents/<int:doc_id>/text', methods=['GET'])
def mp_get_document_text(meeting_id, doc_id):
    """Get extracted text for a document (needed by frontend pipeline)."""
    try:
        with get_db() as (_, cur):
            cur.execute('''
                SELECT id, filename, doc_type, extracted_text
                FROM mp_documents WHERE id = %s AND meeting_id = %s
            ''', (doc_id, meeting_id))
            row = cur.fetchone()

        if not row:
            return jsonify({'error': 'Document not found'}), 404

        return jsonify({
            'id': row['id'],
            'filename': row['filename'],
            'docType': row['doc_type'],
            'extractedText': row['extracted_text'] or '',
        })
    except Exception as e:
        print(f"Error getting document text: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# MEETING PREP — GOOGLE DRIVE INTEGRATION
# ============================================

@app.route('/api/mp/search-drive', methods=['POST'])
def mp_search_drive():
    """Search Google Drive 'Research Reports Char' folder for documents matching a ticker."""
    try:
        from datetime import datetime, timedelta
        import requests as http_requests

        data = request.json
        access_token = data.get('accessToken', '')
        ticker = data.get('ticker', '')
        time_range = data.get('timeRange', '3months')
        keyword = data.get('keyword', '').strip()

        if not access_token:
            return jsonify({'error': 'Google access token required'}), 400
        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400

        # Sanitize inputs to prevent Drive API query injection
        import re
        def sanitize_drive_query(s):
            """Escape single quotes and strip non-printable chars for Drive API queries"""
            return re.sub(r"['\\\x00-\x1f]", '', s.strip())[:100]

        ticker = sanitize_drive_query(ticker)
        keyword = sanitize_drive_query(keyword)

        headers = {'Authorization': f'Bearer {access_token}'}
        drive_api = 'https://www.googleapis.com/drive/v3/files'

        # Calculate date cutoff
        ranges = {
            'day': 1, 'week': 7, 'month': 30, '3months': 90,
            '6months': 180, 'year': 365, '3years': 1095
        }
        days = ranges.get(time_range, 90)
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%S')

        # Find "Research Reports Char" folder
        folder_query = "name = 'Research Reports Char' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        folder_resp = http_requests.get(drive_api, headers=headers, params={
            'q': folder_query, 'fields': 'files(id, name)', 'pageSize': 5
        }, timeout=15)
        if folder_resp.status_code == 401:
            return jsonify({'error': 'Google authentication expired. Please re-authenticate.'}), 401
        folder_resp.raise_for_status()
        folders = folder_resp.json().get('files', [])

        if not folders:
            return jsonify({'error': 'Folder "Research Reports Char" not found in your Google Drive'}), 404

        folder_id = folders[0]['id']

        # Search for files matching ticker (and optional keyword) in that folder
        search_term = keyword if keyword else ticker
        file_query = (
            f"'{folder_id}' in parents and trashed = false "
            f"and modifiedTime > '{cutoff}' "
            f"and (name contains '{search_term}' or fullText contains '{search_term}')"
        )
        file_resp = http_requests.get(drive_api, headers=headers, params={
            'q': file_query,
            'fields': 'files(id, name, mimeType, modifiedTime, size)',
            'pageSize': 50,
            'orderBy': 'modifiedTime desc'
        }, timeout=15)
        file_resp.raise_for_status()
        files = file_resp.json().get('files', [])

        return jsonify({'files': files, 'folderId': folder_id, 'folderName': folders[0]['name']})

    except Exception as e:
        error_msg = str(e)
        print(f"Drive search error: {error_msg}")
        if '401' in error_msg:
            return jsonify({'error': 'Google authentication expired. Please re-authenticate.'}), 401
        return jsonify({'error': error_msg}), 500


@app.route('/api/mp/preview-zip', methods=['POST'])
def mp_preview_zip():
    """Download a zip file from Google Drive and list the PDF files inside."""
    try:
        import requests as http_requests
        import io
        import zipfile

        data = request.json
        access_token = data.get('accessToken', '')
        file_id = data.get('fileId', '')

        if not access_token or not file_id:
            return jsonify({'error': 'accessToken and fileId are required'}), 400

        headers = {'Authorization': f'Bearer {access_token}'}
        drive_api = 'https://www.googleapis.com/drive/v3/files'

        dl_resp = http_requests.get(f'{drive_api}/{file_id}', headers=headers, params={'alt': 'media'}, timeout=60)
        if dl_resp.status_code == 401:
            return jsonify({'error': 'Google authentication expired. Please re-authenticate.'}), 401
        dl_resp.raise_for_status()

        try:
            with zipfile.ZipFile(io.BytesIO(dl_resp.content)) as zf:
                pdfs = []
                for info in zf.infolist():
                    if info.filename.lower().endswith('.pdf') and not info.filename.startswith('__MACOSX'):
                        name = info.filename.split('/')[-1] if '/' in info.filename else info.filename
                        pdfs.append({
                            'zipPath': info.filename,
                            'name': name,
                            'size': info.file_size
                        })
                return jsonify({'pdfs': pdfs})
        except zipfile.BadZipFile:
            return jsonify({'error': 'Invalid or corrupted zip file'}), 400

    except Exception as e:
        error_msg = str(e)
        print(f"Zip preview error: {error_msg}")
        if '401' in error_msg:
            return jsonify({'error': 'Google authentication expired. Please re-authenticate.'}), 401
        return jsonify({'error': error_msg}), 500


@app.route('/api/mp/import-drive-files', methods=['POST'])
def mp_import_drive_files():
    """Download files from Google Drive and import into a meeting. Handles zip files by extracting PDFs."""
    try:
        import requests as http_requests
        from PyPDF2 import PdfReader
        import io
        import zipfile

        data = request.json
        access_token = data.get('accessToken', '')
        meeting_id = data.get('meetingId')
        files_to_import = data.get('files', [])

        if not access_token or not meeting_id or not files_to_import:
            return jsonify({'error': 'accessToken, meetingId, and files are required'}), 400

        headers = {'Authorization': f'Bearer {access_token}'}
        drive_api = 'https://www.googleapis.com/drive/v3/files'

        with get_db(commit=True) as (_, cur):
            # Verify meeting exists
            cur.execute('SELECT id FROM mp_meetings WHERE id = %s', (meeting_id,))
            if not cur.fetchone():
                return jsonify({'error': 'Meeting not found'}), 404

            cur.execute('SELECT COALESCE(MAX(upload_order), 0) AS max_order FROM mp_documents WHERE meeting_id = %s', (meeting_id,))
            order = cur.fetchone()['max_order']

            def import_pdf(pdf_bytes, filename, doc_date, cur, meeting_id, order):
                """Extract text from PDF bytes and insert into mp_documents."""
                extracted_text = ''
                page_count = None
                try:
                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    pages = []
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            pages.append(t)
                    extracted_text = '\n\n'.join(pages)
                    page_count = len(reader.pages)
                except Exception as ex:
                    print(f"PDF extraction error for {filename}: {ex}")

                doc_type = classify_mp_document(filename, extracted_text)
                token_estimate = len(extracted_text) // 4 if extracted_text else 0
                file_size = len(pdf_bytes)

                cur.execute('''
                    INSERT INTO mp_documents (meeting_id, filename, file_data, doc_type, doc_date,
                        page_count, token_estimate, extracted_text, upload_order, file_size)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, filename, doc_type, doc_date, page_count, token_estimate, upload_order, file_size, created_at
                ''', (meeting_id, filename, '', doc_type, doc_date,
                      page_count, token_estimate, extracted_text, order, file_size))
                row = dict(cur.fetchone())
                row['created_at'] = row['created_at'].isoformat() if row['created_at'] else None
                return row

            results = []
            for file_info in files_to_import:
                file_id = file_info.get('id')
                filename = file_info.get('name', 'unknown')
                mime_type = file_info.get('mimeType', '')
                doc_date = file_info.get('modifiedTime', '')[:10] if file_info.get('modifiedTime') else None

                try:
                    # Download file content via REST API
                    if mime_type in ('application/vnd.google-apps.document', 'application/vnd.google-apps.spreadsheet'):
                        dl_resp = http_requests.get(f'{drive_api}/{file_id}/export', headers=headers, params={'mimeType': 'application/pdf'}, timeout=60)
                    else:
                        dl_resp = http_requests.get(f'{drive_api}/{file_id}', headers=headers, params={'alt': 'media'}, timeout=60)
                    dl_resp.raise_for_status()
                    file_content = dl_resp.content

                    is_zip = (mime_type == 'application/zip' or
                              mime_type == 'application/x-zip-compressed' or
                              filename.lower().endswith('.zip'))

                    if is_zip:
                        # Extract selected (or all) PDFs from the zip file
                        selected_pdfs = set(file_info.get('selectedPdfs', []))
                        try:
                            with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
                                pdf_names = [n for n in zf.namelist()
                                             if n.lower().endswith('.pdf') and not n.startswith('__MACOSX')]
                                if selected_pdfs:
                                    pdf_names = [n for n in pdf_names if n in selected_pdfs]
                                if not pdf_names:
                                    results.append({'filename': filename, 'error': 'No matching PDF files found inside zip'})
                                    continue
                                for pdf_name in pdf_names:
                                    order += 1
                                    pdf_bytes = zf.read(pdf_name)
                                    pdf_filename = pdf_name.split('/')[-1] if '/' in pdf_name else pdf_name
                                    row = import_pdf(pdf_bytes, pdf_filename, doc_date, cur, meeting_id, order)
                                    row['fromZip'] = filename
                                    results.append(row)
                        except zipfile.BadZipFile:
                            results.append({'filename': filename, 'error': 'Invalid or corrupted zip file'})
                    else:
                        # Regular PDF file
                        order += 1
                        row = import_pdf(file_content, filename, doc_date, cur, meeting_id, order)
                        results.append(row)

                except Exception as ex:
                    print(f"Error importing {filename}: {ex}")
                    results.append({'filename': filename, 'error': str(ex)})

        return jsonify(results)

    except Exception as e:
        error_msg = str(e)
        print(f"Drive import error: {error_msg}")
        if 'invalid_grant' in error_msg.lower() or '401' in error_msg:
            return jsonify({'error': 'Google authentication expired. Please re-authenticate.'}), 401
        return jsonify({'error': error_msg}), 500


# ============================================
# GENERIC GOOGLE DRIVE DOWNLOAD (for Research, Summary, Portfolio tabs)
# ============================================

@app.route('/api/drive/download-files', methods=['POST'])
def drive_download_files():
    """Download files from Google Drive and extract text. Returns extracted data without storing in DB."""
    try:
        import requests as http_requests
        from PyPDF2 import PdfReader
        import io
        import zipfile
        import base64

        data = request.json
        access_token = data.get('accessToken', '')
        files_to_import = data.get('files', [])

        if not access_token or not files_to_import:
            return jsonify({'error': 'accessToken and files are required'}), 400

        headers = {'Authorization': f'Bearer {access_token}'}
        drive_api = 'https://www.googleapis.com/drive/v3/files'

        def extract_pdf(pdf_bytes, filename):
            """Extract text from PDF bytes and return data dict."""
            extracted_text = ''
            page_count = None
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                extracted_text = '\n\n'.join(pages)
                page_count = len(reader.pages)
            except Exception as ex:
                print(f"PDF extraction error for {filename}: {ex}")

            return {
                'filename': filename,
                'extractedText': extracted_text,
                'pageCount': page_count,
                'fileSize': len(pdf_bytes),
                'fileData': base64.b64encode(pdf_bytes).decode('utf-8')
            }

        results = []
        for file_info in files_to_import:
            file_id = file_info.get('id')
            filename = file_info.get('name', 'unknown')
            mime_type = file_info.get('mimeType', '')

            try:
                if mime_type in ('application/vnd.google-apps.document', 'application/vnd.google-apps.spreadsheet'):
                    dl_resp = http_requests.get(f'{drive_api}/{file_id}/export', headers=headers, params={'mimeType': 'application/pdf'}, timeout=60)
                else:
                    dl_resp = http_requests.get(f'{drive_api}/{file_id}', headers=headers, params={'alt': 'media'}, timeout=60)
                dl_resp.raise_for_status()
                file_content = dl_resp.content

                is_zip = (mime_type == 'application/zip' or
                          mime_type == 'application/x-zip-compressed' or
                          filename.lower().endswith('.zip'))

                if is_zip:
                    selected_pdfs = set(file_info.get('selectedPdfs', []))
                    try:
                        with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
                            pdf_names = [n for n in zf.namelist()
                                         if n.lower().endswith('.pdf') and not n.startswith('__MACOSX')]
                            if selected_pdfs:
                                pdf_names = [n for n in pdf_names if n in selected_pdfs]
                            if not pdf_names:
                                results.append({'filename': filename, 'error': 'No matching PDF files found inside zip'})
                                continue
                            for pdf_name in pdf_names:
                                pdf_bytes = zf.read(pdf_name)
                                pdf_filename = pdf_name.split('/')[-1] if '/' in pdf_name else pdf_name
                                row = extract_pdf(pdf_bytes, pdf_filename)
                                row['fromZip'] = filename
                                results.append(row)
                    except zipfile.BadZipFile:
                        results.append({'filename': filename, 'error': 'Invalid or corrupted zip file'})
                else:
                    row = extract_pdf(file_content, filename)
                    results.append(row)

            except Exception as ex:
                print(f"Error downloading {filename}: {ex}")
                results.append({'filename': filename, 'error': str(ex)})

        return jsonify({'files': results})

    except Exception as e:
        error_msg = str(e)
        print(f"Drive download error: {error_msg}")
        if 'invalid_grant' in error_msg.lower() or '401' in error_msg:
            return jsonify({'error': 'Google authentication expired. Please re-authenticate.'}), 401
        return jsonify({'error': error_msg}), 500


# ============================================
# SLIDE GENERATOR ENDPOINTS
# ============================================

# --- Theme constants ---
SLIDE_THEMES = {
    'sketchnote': {
        'name': 'Sketchnote',
        'style_prefix': 'Background: Warm beige/cream colored textured paper (like aged notebook paper)\nIllustrations: Cute hand-drawn cartoon style with colorful doodles\nTitle text: Large, colorful, hand-lettered style typography (rounded, playful, multicolor gradients)\nBody text: Clean, readable text in dark gray/brown\nDecorations: Small stars, sparkles, arrows, dots, swirls scattered around\nLayout: Professional yet friendly, like a designer\'s sketchnote\nAspect ratio: 16:9 widescreen slide\nResolution: High quality, crisp text\nALL text MUST be in English\nDO NOT include any watermarks or AI generation notices',
        'illustration_guidance': 'Use topic-appropriate icons and characters. DO NOT include robots unless topic is AI/robotics.',
        'negative_guidance': 'DO NOT include robots or bot characters unless the topic is specifically about AI/robotics.',
    },
    'healthcare': {
        'name': 'Healthcare',
        'style_prefix': 'Background: Warm beige/cream colored textured paper\nIllustrations: Cute hand-drawn cartoon style with medical-themed doodles\nTitle text: Large, colorful, hand-lettered typography (soft blues, greens, whites)\nBody text: Clean, readable text in dark gray/brown\nDecorations: Stars, sparkles, medical icons scattered around\nLayout: Professional yet friendly sketchnote style\nAspect ratio: 16:9 widescreen\nResolution: High quality, crisp text\nALL text MUST be in English\nDO NOT include any watermarks',
        'illustration_guidance': 'Use medical icons: stethoscopes, hearts, hospitals, pill capsules, doctors, nurses, medical charts, DNA helixes.',
        'negative_guidance': 'DO NOT include robots unless topic is health AI. DO NOT include violent or graphic medical imagery.',
    },
    'technology': {
        'name': 'Technology',
        'style_prefix': 'Background: Warm beige/cream colored textured paper\nIllustrations: Cute hand-drawn cartoon style with tech-themed doodles\nTitle text: Large, colorful, hand-lettered typography (electric blues, purples, neon greens)\nBody text: Clean, readable text in dark gray/brown\nDecorations: Stars, sparkles, circuit-like arrows, dots\nLayout: Professional yet friendly sketchnote style\nAspect ratio: 16:9 widescreen\nResolution: High quality, crisp text\nALL text MUST be in English\nDO NOT include any watermarks',
        'illustration_guidance': 'Use tech icons: laptops, smartphones, cloud symbols, servers, code brackets, AI brain icons. Robot/AI characters are appropriate.',
        'negative_guidance': 'DO NOT include medical or finance-specific imagery unless warranted.',
    },
    'finance': {
        'name': 'Finance',
        'style_prefix': 'Background: Warm beige/cream colored textured paper\nIllustrations: Cute hand-drawn cartoon style with finance-themed doodles\nTitle text: Large, colorful, hand-lettered typography (deep blues, golds, greens)\nBody text: Clean, readable text in dark gray/brown\nDecorations: Stars, sparkles, dollar signs scattered around\nLayout: Professional yet friendly sketchnote style\nAspect ratio: 16:9 widescreen\nResolution: High quality, crisp text\nALL text MUST be in English\nDO NOT include any watermarks',
        'illustration_guidance': 'Use finance icons: stock charts, bank buildings, money bags, bull/bear characters, trading terminals.',
        'negative_guidance': 'DO NOT include robots unless discussing fintech/algo trading.',
    },
    'general': {
        'name': 'General',
        'style_prefix': 'Background: Warm beige/cream colored textured paper\nIllustrations: Cute hand-drawn cartoon style with colorful doodles\nTitle text: Large, colorful, hand-lettered typography\nBody text: Clean, readable text in dark gray/brown\nDecorations: Stars, sparkles, arrows scattered around\nLayout: Professional yet friendly sketchnote style\nAspect ratio: 16:9 widescreen\nResolution: High quality, crisp text\nALL text MUST be in English\nDO NOT include any watermarks',
        'illustration_guidance': 'Use topic-appropriate icons and characters.',
        'negative_guidance': 'DO NOT include robots unless topic is about AI/robotics.',
    },
}

SECTOR_THEME_MAP = {
    'Healthcare': 'healthcare',
    'Health Care': 'healthcare',
    'Technology': 'technology',
    'Information Technology': 'technology',
    'Communication Services': 'technology',
    'Financials': 'finance',
    'Financial Services': 'finance',
    'Energy': 'general',
    'Consumer Discretionary': 'general',
    'Consumer Staples': 'general',
    'Industrials': 'general',
    'Materials': 'general',
    'Real Estate': 'finance',
    'Utilities': 'general',
}


def _build_slide_prompt(slide_data, theme_name, project_title, total_slides):
    """Build the complete Gemini prompt for a single slide."""
    theme = SLIDE_THEMES.get(theme_name, SLIDE_THEMES['sketchnote'])
    parts = ["You are generating a presentation slide image.\n"]
    parts.append("VISUAL STYLE (MUST follow exactly):")
    parts.append(theme['style_prefix'])
    if theme.get('illustration_guidance'):
        parts.append("\nILLUSTRATION GUIDANCE:")
        parts.append(theme['illustration_guidance'])
    if theme.get('negative_guidance'):
        parts.append("\nIMPORTANT RESTRICTIONS:")
        parts.append(theme['negative_guidance'])
    if not slide_data.get('no_header', False):
        parts.append(f'\nAt the top-left corner, show small header text: "{project_title}"')
        parts.append(f'At the top-right corner, show: "Slide {slide_data["slide_number"]} / {total_slides}"')
    else:
        parts.append("\nDo NOT include any header or slide number on this slide.")
    parts.append(f"\nSLIDE CONTENT:\n{slide_data['content'].strip()}")
    hints = slide_data.get('illustration_hints', [])
    if hints:
        parts.append("\nILLUSTRATIONS TO INCLUDE:")
        for h in hints:
            parts.append(f"- {h}")
    return "\n".join(parts)


def _generate_slide_image(prompt, api_key=None):
    """Generate a slide image via Gemini API. Returns base64 PNG string or None."""
    key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
    if not key:
        return None
    client = genai.Client(api_key=key)
    model = "gemini-3-pro-image-preview"
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=genai_types.ImageConfig(aspect_ratio="16:9"),
                ),
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    return base64.b64encode(part.inline_data.data).decode('utf-8')
            import time; time.sleep(2)
        except Exception as e:
            print(f"Slide generation attempt {attempt+1} failed: {e}")
            if attempt < 2:
                import time; time.sleep((attempt + 1) * 5)
    return None


def _compute_content_hash(slide_data):
    """Compute SHA-256 hash of slide content for change detection."""
    import hashlib
    hints = ','.join(slide_data.get('illustration_hints', []))
    payload = f"{slide_data.get('title','')}|{slide_data.get('type','')}|{slide_data.get('content','')}|{hints}|{slide_data.get('no_header', False)}"
    return hashlib.sha256(payload.encode()).hexdigest()


@app.route('/api/slides/themes', methods=['GET'])
def get_slide_themes():
    """List available slide themes."""
    result = []
    for key, theme in SLIDE_THEMES.items():
        result.append({'id': key, 'name': theme['name']})
    return jsonify(result)


@app.route('/api/slides/projects', methods=['GET'])
def get_slide_projects():
    """List all slide projects."""
    try:
        with get_db() as (conn, cur):
            cur.execute('''
                SELECT id, ticker, title, theme, status, total_slides, created_at, updated_at
                FROM slide_projects ORDER BY updated_at DESC
            ''')
            rows = cur.fetchall()
        result = []
        for row in rows:
            result.append({
                'id': row['id'],
                'ticker': row['ticker'],
                'title': row['title'],
                'theme': row['theme'],
                'status': row['status'],
                'total_slides': row['total_slides'],
                'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
            })
        return jsonify(result)
    except Exception as e:
        print(f"Error getting slide projects: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects', methods=['POST'])
def create_slide_project():
    """Create a new slide project."""
    try:
        data = request.json
        title = data.get('title', '')
        ticker = data.get('ticker', '').upper() if data.get('ticker') else None
        theme = data.get('theme', 'sketchnote')
        if not title:
            return jsonify({'error': 'Title is required'}), 400
        if theme not in SLIDE_THEMES:
            theme = 'sketchnote'
        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                INSERT INTO slide_projects (ticker, title, theme, status, total_slides)
                VALUES (%s, %s, %s, 'draft', 0)
                RETURNING id
            ''', (ticker, title, theme))
            project_id = cur.fetchone()['id']
        return jsonify({'success': True, 'id': project_id})
    except Exception as e:
        print(f"Error creating slide project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>', methods=['GET'])
def get_slide_project(project_id):
    """Get a slide project with all its slides."""
    try:
        with get_db() as (conn, cur):
            cur.execute('SELECT * FROM slide_projects WHERE id = %s', (project_id,))
            project = cur.fetchone()
            if not project:
                return jsonify({'error': 'Project not found'}), 404
            cur.execute('''
                SELECT id, slide_number, title, type, content, illustration_hints,
                       no_header, image_data, content_hash, status
                FROM slide_items WHERE project_id = %s ORDER BY slide_number
            ''', (project_id,))
            slides = cur.fetchall()
        result = {
            'id': project['id'],
            'ticker': project['ticker'],
            'title': project['title'],
            'theme': project['theme'],
            'status': project['status'],
            'total_slides': project['total_slides'],
            'created_at': project['created_at'].isoformat() if project['created_at'] else None,
            'updated_at': project['updated_at'].isoformat() if project['updated_at'] else None,
            'slides': [],
        }
        for s in slides:
            result['slides'].append({
                'id': s['id'],
                'slide_number': s['slide_number'],
                'title': s['title'],
                'type': s['type'],
                'content': s['content'],
                'illustration_hints': s['illustration_hints'] or [],
                'no_header': s['no_header'],
                'has_image': bool(s['image_data']),
                'status': s['status'],
            })
        return jsonify(result)
    except Exception as e:
        print(f"Error getting slide project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>', methods=['DELETE'])
def delete_slide_project(project_id):
    """Delete a slide project and all its slides."""
    try:
        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM slide_projects WHERE id = %s', (project_id,))
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting slide project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/slides/<int:slide_num>', methods=['PUT'])
def update_slide(project_id, slide_num):
    """Update a slide's content."""
    try:
        data = request.json
        with get_db(commit=True) as (conn, cur):
            updates = []
            params = []
            for field in ['title', 'content', 'type', 'no_header']:
                if field in data:
                    updates.append(f"{field} = %s")
                    params.append(data[field])
            if 'illustration_hints' in data:
                updates.append("illustration_hints = %s")
                params.append(json.dumps(data['illustration_hints']))
            if not updates:
                return jsonify({'error': 'No fields to update'}), 400
            # Recompute hash and mark as edited
            new_hash = _compute_content_hash({
                'title': data.get('title', ''),
                'type': data.get('type', 'content'),
                'content': data.get('content', ''),
                'illustration_hints': data.get('illustration_hints', []),
                'no_header': data.get('no_header', False),
            })
            updates.append("content_hash = %s")
            params.append(new_hash)
            updates.append("status = 'edited'")
            updates.append("updated_at = %s")
            params.append(datetime.utcnow())
            params.extend([project_id, slide_num])
            cur.execute(f'''
                UPDATE slide_items SET {', '.join(updates)}
                WHERE project_id = %s AND slide_number = %s
            ''', params)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating slide: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/slides', methods=['POST'])
def add_slide(project_id):
    """Add a new slide to a project."""
    try:
        data = request.json
        title = data.get('title', 'New Slide')
        after = data.get('after')  # insert after this slide number
        content = data.get('content', f'[Edit content for: {title}]')
        slide_type = data.get('type', 'content')
        hints = data.get('illustration_hints', [])
        no_header = data.get('no_header', False)
        with get_db(commit=True) as (conn, cur):
            if after is not None:
                new_num = after + 1
                # Shift existing slides up
                cur.execute('''
                    UPDATE slide_items SET slide_number = slide_number + 1, updated_at = %s
                    WHERE project_id = %s AND slide_number >= %s
                ''', (datetime.utcnow(), project_id, new_num))
            else:
                cur.execute('SELECT COALESCE(MAX(slide_number), 0) + 1 as next_num FROM slide_items WHERE project_id = %s', (project_id,))
                new_num = cur.fetchone()['next_num']
            content_hash = _compute_content_hash({
                'title': title, 'type': slide_type, 'content': content,
                'illustration_hints': hints, 'no_header': no_header,
            })
            cur.execute('''
                INSERT INTO slide_items (project_id, slide_number, title, type, content, illustration_hints, no_header, content_hash, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'new')
                RETURNING id
            ''', (project_id, new_num, title, slide_type, content, json.dumps(hints), no_header, content_hash))
            slide_id = cur.fetchone()['id']
            # Update project total
            cur.execute('SELECT COUNT(*) as cnt FROM slide_items WHERE project_id = %s', (project_id,))
            total = cur.fetchone()['cnt']
            cur.execute('UPDATE slide_projects SET total_slides = %s, updated_at = %s WHERE id = %s',
                        (total, datetime.utcnow(), project_id))
        return jsonify({'success': True, 'id': slide_id, 'slide_number': new_num})
    except Exception as e:
        print(f"Error adding slide: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/slides/<int:slide_num>', methods=['DELETE'])
def delete_slide_item(project_id, slide_num):
    """Delete a slide and renumber the rest."""
    try:
        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM slide_items WHERE project_id = %s AND slide_number = %s', (project_id, slide_num))
            # Renumber slides after the deleted one
            cur.execute('''
                UPDATE slide_items SET slide_number = slide_number - 1, updated_at = %s
                WHERE project_id = %s AND slide_number > %s
            ''', (datetime.utcnow(), project_id, slide_num))
            # Update project total
            cur.execute('SELECT COUNT(*) as cnt FROM slide_items WHERE project_id = %s', (project_id,))
            total = cur.fetchone()['cnt']
            cur.execute('UPDATE slide_projects SET total_slides = %s, updated_at = %s WHERE id = %s',
                        (total, datetime.utcnow(), project_id))
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting slide: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/generate/<int:slide_num>', methods=['POST'])
def generate_one_slide(project_id, slide_num):
    """Generate image for a single slide."""
    try:
        data = request.json or {}
        gemini_key = data.get('geminiApiKey', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
        if not gemini_key:
            return jsonify({'error': 'Gemini API key not configured. Add it in Settings.'}), 400
        with get_db() as (conn, cur):
            cur.execute('SELECT * FROM slide_projects WHERE id = %s', (project_id,))
            project = cur.fetchone()
            if not project:
                return jsonify({'error': 'Project not found'}), 404
            cur.execute('SELECT * FROM slide_items WHERE project_id = %s AND slide_number = %s', (project_id, slide_num))
            slide = cur.fetchone()
            if not slide:
                return jsonify({'error': 'Slide not found'}), 404
            total = project['total_slides']
        slide_data = {
            'slide_number': slide['slide_number'],
            'title': slide['title'],
            'type': slide['type'],
            'content': slide['content'] or '',
            'illustration_hints': slide['illustration_hints'] or [],
            'no_header': slide['no_header'],
        }
        prompt = _build_slide_prompt(slide_data, project['theme'], project['title'], total)
        image_b64 = _generate_slide_image(prompt, api_key=gemini_key)
        if not image_b64:
            return jsonify({'error': 'Failed to generate image'}), 500
        content_hash = _compute_content_hash(slide_data)
        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                UPDATE slide_items SET image_data = %s, content_hash = %s, status = 'generated', updated_at = %s
                WHERE project_id = %s AND slide_number = %s
            ''', (image_b64, content_hash, datetime.utcnow(), project_id, slide_num))
        return jsonify({'success': True, 'has_image': True})
    except Exception as e:
        print(f"Error generating slide: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/generate', methods=['POST'])
def generate_slides(project_id):
    """Generate images for slides (all, changed-only, or specific slide_numbers)."""
    try:
        data = request.json or {}
        changed_only = data.get('changed_only', True)
        slide_numbers = data.get('slide_numbers', None)  # optional list of specific slide numbers
        gemini_key = data.get('geminiApiKey', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
        if not gemini_key:
            return jsonify({'error': 'Gemini API key not configured. Add it in Settings.'}), 400
        with get_db() as (conn, cur):
            cur.execute('SELECT * FROM slide_projects WHERE id = %s', (project_id,))
            project = cur.fetchone()
            if not project:
                return jsonify({'error': 'Project not found'}), 404
            cur.execute('UPDATE slide_projects SET status = %s, updated_at = %s WHERE id = %s',
                        ('generating', datetime.utcnow(), project_id))
            conn.commit()
        import threading
        def _generate_all(pid, theme, title, only_changed, api_key, target_slides=None):
            import time
            try:
                with get_db() as (conn, cur):
                    if target_slides:
                        placeholders = ','.join(['%s'] * len(target_slides))
                        cur.execute(f'SELECT * FROM slide_items WHERE project_id = %s AND slide_number IN ({placeholders}) ORDER BY slide_number',
                                    [pid] + list(target_slides))
                    else:
                        cur.execute('SELECT * FROM slide_items WHERE project_id = %s ORDER BY slide_number', (pid,))
                    slides = cur.fetchall()
                    # Get total slides in project for slide numbering
                    cur.execute('SELECT COUNT(*) as cnt FROM slide_items WHERE project_id = %s', (pid,))
                    total = cur.fetchone()['cnt']
                # Build list of slides that actually need generation
                to_generate = []
                for slide in slides:
                    slide_data = {
                        'slide_number': slide['slide_number'],
                        'title': slide['title'],
                        'type': slide['type'],
                        'content': slide['content'] or '',
                        'illustration_hints': slide['illustration_hints'] or [],
                        'no_header': slide['no_header'],
                    }
                    current_hash = _compute_content_hash(slide_data)
                    if target_slides:
                        # When specific slides requested, always generate them
                        to_generate.append((slide, slide_data, current_hash))
                    elif only_changed and slide['content_hash'] == current_hash and slide['image_data']:
                        continue
                    else:
                        to_generate.append((slide, slide_data, current_hash))
                generated_count = 0
                for idx, (slide, slide_data, current_hash) in enumerate(to_generate):
                    # Update progress: store in project status as "generating:current:total"
                    with get_db(commit=True) as (conn2, cur2):
                        progress_status = f"generating:{idx+1}:{len(to_generate)}"
                        cur2.execute('UPDATE slide_projects SET status = %s, updated_at = %s WHERE id = %s',
                                    (progress_status, datetime.utcnow(), pid))
                    prompt = _build_slide_prompt(slide_data, theme, title, total)
                    image_b64 = _generate_slide_image(prompt, api_key=api_key)
                    if image_b64:
                        with get_db(commit=True) as (conn2, cur2):
                            cur2.execute('''
                                UPDATE slide_items SET image_data = %s, content_hash = %s, status = 'generated', updated_at = %s
                                WHERE id = %s
                            ''', (image_b64, current_hash, datetime.utcnow(), slide['id']))
                        generated_count += 1
                    time.sleep(2)
                with get_db(commit=True) as (conn, cur):
                    cur.execute('UPDATE slide_projects SET status = %s, updated_at = %s WHERE id = %s',
                                ('ready', datetime.utcnow(), pid))
                print(f"Slide generation complete: {generated_count} slides for project {pid}")
            except Exception as e:
                print(f"Slide generation error: {e}")
                with get_db(commit=True) as (conn, cur):
                    cur.execute('UPDATE slide_projects SET status = %s, updated_at = %s WHERE id = %s',
                                ('error', datetime.utcnow(), pid))
        thread = threading.Thread(
            target=_generate_all,
            args=(project_id, project['theme'], project['title'], changed_only, gemini_key, slide_numbers),
            daemon=True,
        )
        thread.start()
        return jsonify({'success': True, 'status': 'generating'})
    except Exception as e:
        print(f"Error starting slide generation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/slides/<int:slide_num>/image', methods=['GET'])
def get_slide_image(project_id, slide_num):
    """Get the generated image for a slide."""
    try:
        with get_db() as (conn, cur):
            cur.execute('SELECT image_data FROM slide_items WHERE project_id = %s AND slide_number = %s', (project_id, slide_num))
            row = cur.fetchone()
            if not row or not row['image_data']:
                return jsonify({'error': 'No image available'}), 404
        return jsonify({'image_data': row['image_data']})
    except Exception as e:
        print(f"Error getting slide image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/export/pdf', methods=['POST'])
def export_slide_pdf(project_id):
    """Export all slides as a PDF. Returns base64-encoded PDF."""
    try:
        import img2pdf
        with get_db() as (conn, cur):
            cur.execute('SELECT title FROM slide_projects WHERE id = %s', (project_id,))
            project = cur.fetchone()
            if not project:
                return jsonify({'error': 'Project not found'}), 404
            cur.execute('''
                SELECT image_data FROM slide_items
                WHERE project_id = %s AND image_data IS NOT NULL
                ORDER BY slide_number
            ''', (project_id,))
            slides = cur.fetchall()
        if not slides:
            return jsonify({'error': 'No generated slides to export'}), 400
        img_bytes_list = []
        for s in slides:
            img_bytes_list.append(base64.b64decode(s['image_data']))
        pdf_bytes = img2pdf.convert(img_bytes_list)
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        return jsonify({
            'success': True,
            'pdf_data': pdf_b64,
            'filename': f"{project['title'].replace(' ', '_')}.pdf",
            'slide_count': len(slides),
        })
    except Exception as e:
        print(f"Error exporting PDF: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/generate-outline', methods=['POST'])
def generate_slide_outline():
    """Generate a slide outline from various sources using LLM."""
    try:
        data = request.json
        source_type = data.get('source_type', 'ticker')  # ticker | research | summary | meetingprep | custom
        source_ids = data.get('source_ids', [])
        custom_text = data.get('custom_text', '')
        ticker = data.get('ticker', '').upper()

        context_parts = []
        company_name = ''
        theme = 'sketchnote'
        presentation_subject = ''

        if source_type == 'ticker':
            # Original behavior: pull from stock_overviews + portfolio_analyses
            if not ticker:
                return jsonify({'error': 'Ticker is required for ticker-based generation'}), 400
            overview_data = None
            analysis_data = None
            with get_db() as (conn, cur):
                cur.execute('SELECT * FROM stock_overviews WHERE ticker = %s', (ticker,))
                overview_row = cur.fetchone()
                if overview_row:
                    overview_data = {
                        'company_name': overview_row['company_name'],
                        'company_overview': overview_row['company_overview'],
                        'business_model': overview_row['business_model'],
                        'business_mix': overview_row.get('business_mix', ''),
                        'opportunities': overview_row['opportunities'],
                        'risks': overview_row['risks'],
                        'conclusion': overview_row['conclusion'],
                    }
                cur.execute('SELECT * FROM portfolio_analyses WHERE ticker = %s', (ticker,))
                analysis_row = cur.fetchone()
                if analysis_row:
                    analysis_data = {
                        'company': analysis_row['company'],
                        'analysis': analysis_row['analysis'],
                    }
            if not overview_data and not analysis_data:
                return jsonify({'error': f'No existing data found for {ticker}. Create an overview or analysis first.'}), 404
            company_name = (overview_data or {}).get('company_name', '') or (analysis_data or {}).get('company', ticker)
            presentation_subject = f"an equity research presentation about {ticker} ({company_name})"
            # Auto-detect theme from sector
            sector = None
            if analysis_data and analysis_data.get('analysis'):
                a = analysis_data['analysis']
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {}
                sector = a.get('sector', '')
            theme = SECTOR_THEME_MAP.get(sector, 'sketchnote')
            context_parts = [f"Ticker: {ticker}", f"Company: {company_name}"]
            if overview_data:
                for field in ['company_overview', 'business_model', 'business_mix', 'opportunities', 'risks', 'conclusion']:
                    if overview_data.get(field):
                        label = field.replace('_', ' ').title()
                        context_parts.append(f"{label}:\n{overview_data[field]}")
            if analysis_data and analysis_data.get('analysis'):
                a = analysis_data['analysis']
                if isinstance(a, dict):
                    context_parts.append(f"Investment Analysis:\n{json.dumps(a, indent=2)[:3000]}")

        elif source_type == 'research':
            if not source_ids:
                return jsonify({'error': 'Select at least one research document'}), 400
            with get_db() as (conn, cur):
                placeholders = ','.join(['%s'] * len(source_ids))
                cur.execute(f'SELECT id, name, smart_name, content FROM research_documents WHERE id IN ({placeholders})', source_ids)
                rows = cur.fetchall()
            if not rows:
                return jsonify({'error': 'No research documents found for the selected IDs'}), 404
            presentation_subject = "a presentation based on the following research documents"
            for row in rows:
                doc_name = row['smart_name'] or row['name'] or f"Document {row['id']}"
                content = (row['content'] or '')[:5000]
                context_parts.append(f"Document: {doc_name}\n{content}")

        elif source_type == 'summary':
            if not source_ids:
                return jsonify({'error': 'Select at least one meeting summary'}), 400
            with get_db() as (conn, cur):
                placeholders = ','.join(['%s'] * len(source_ids))
                cur.execute(f'SELECT id, title, topic, raw_notes, summary FROM meeting_summaries WHERE id IN ({placeholders})', source_ids)
                rows = cur.fetchall()
            if not rows:
                return jsonify({'error': 'No meeting summaries found for the selected IDs'}), 404
            presentation_subject = "a presentation based on the following meeting notes and summaries"
            for row in rows:
                title = row['title'] or f"Summary {row['id']}"
                context_parts.append(f"Meeting: {title}")
                if row.get('topic'):
                    context_parts.append(f"Topic: {row['topic']}")
                if row.get('raw_notes'):
                    context_parts.append(f"Notes:\n{row['raw_notes'][:3000]}")
                if row.get('summary'):
                    context_parts.append(f"Summary:\n{row['summary'][:3000]}")

        elif source_type == 'meetingprep':
            if not source_ids:
                return jsonify({'error': 'Select at least one meeting prep session'}), 400
            with get_db() as (conn, cur):
                placeholders = ','.join(['%s'] * len(source_ids))
                cur.execute(f'''SELECT m.id, m.notes, c.ticker, c.name as company_name, c.sector
                               FROM mp_meetings m JOIN mp_companies c ON m.company_id = c.id
                               WHERE m.id IN ({placeholders})''', source_ids)
                meetings = cur.fetchall()
                if not meetings:
                    return jsonify({'error': 'No meeting prep sessions found for the selected IDs'}), 404
                # Fetch documents for these meetings
                cur.execute(f'SELECT meeting_id, filename, extracted_text FROM mp_documents WHERE meeting_id IN ({placeholders}) ORDER BY upload_order', source_ids)
                docs = cur.fetchall()
            presentation_subject = "a presentation based on meeting prep materials"
            for meeting in meetings:
                context_parts.append(f"Meeting Prep: {meeting['company_name']} ({meeting['ticker']})")
                if meeting.get('notes'):
                    context_parts.append(f"Meeting Notes:\n{meeting['notes'][:2000]}")
                if meeting.get('sector'):
                    theme = SECTOR_THEME_MAP.get(meeting['sector'], 'sketchnote')
            for doc in docs:
                if doc.get('extracted_text'):
                    context_parts.append(f"Document ({doc['filename']}):\n{doc['extracted_text'][:3000]}")

        elif source_type == 'custom':
            if not custom_text.strip():
                return jsonify({'error': 'Please provide text content'}), 400
            presentation_subject = "a presentation based on the following content"
            context_parts.append(custom_text[:10000])

        else:
            return jsonify({'error': f'Unknown source_type: {source_type}'}), 400

        llm_prompt = f"""You are creating slide content for {presentation_subject}.

Based on this source material:

{chr(10).join(context_parts)}

Generate a JSON array of slides for a comprehensive presentation. Each slide should have:
- slide_number (integer starting at 1)
- title (string)
- type (one of: title, toc, section_divider, content, closing)
- content (detailed text describing what should appear on the slide - include specific data, numbers, quotes from the source material)
- illustration_hints (array of strings like "company", "growth", "money", "risk", "data", "leader", "opportunity")
- no_header (boolean - true only for title, section_divider, and closing slides)

Structure: title slide, table of contents, then 5-7 sections covering the key themes from the source material. Each section should have a section divider slide followed by 2-4 content slides. End with a closing slide. Target 30-40 slides total.

IMPORTANT: Fill in actual data, numbers, and specific details from the source material. Do NOT use placeholder brackets like [Company Name] - use real names and real data from the content provided.

Return ONLY valid JSON array, no markdown fencing."""

        # Call Claude for outline generation
        api_key = os.environ.get('ANTHROPIC_API_KEY', '') or data.get('apiKey', '') or data.get('api_key', '')
        if not api_key:
            return jsonify({'error': 'Anthropic API key not configured. Add it in Settings.'}), 500
        client_ai = anthropic.Anthropic(api_key=api_key)
        response = client_ai.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": llm_prompt}],
        )
        response_text = response.content[0].text.strip()
        # Parse JSON from response
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        slides_data = json.loads(response_text)
        return jsonify({
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'theme': theme,
            'slides': slides_data,
        })
    except json.JSONDecodeError as e:
        print(f"Error parsing slide outline JSON: {e}")
        return jsonify({'error': 'Failed to parse LLM response as JSON'}), 500
    except Exception as e:
        print(f"Error generating slide outline: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slides/projects/<int:project_id>/populate', methods=['POST'])
def populate_slides(project_id):
    """Populate a project with slides from generated outline data."""
    try:
        data = request.json
        slides_data = data.get('slides', [])
        if not slides_data:
            return jsonify({'error': 'No slides data provided'}), 400
        with get_db(commit=True) as (conn, cur):
            # Clear existing slides
            cur.execute('DELETE FROM slide_items WHERE project_id = %s', (project_id,))
            for s in slides_data:
                content_hash = _compute_content_hash(s)
                cur.execute('''
                    INSERT INTO slide_items (project_id, slide_number, title, type, content, illustration_hints, no_header, content_hash, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'new')
                ''', (
                    project_id, s['slide_number'], s['title'], s.get('type', 'content'),
                    s.get('content', ''), json.dumps(s.get('illustration_hints', [])),
                    s.get('no_header', False), content_hash,
                ))
            total = len(slides_data)
            cur.execute('UPDATE slide_projects SET total_slides = %s, updated_at = %s WHERE id = %s',
                        (total, datetime.utcnow(), project_id))
        return jsonify({'success': True, 'total_slides': total})
    except Exception as e:
        print(f"Error populating slides: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# SLIDE SOURCE LISTING ENDPOINTS
# ============================================

@app.route('/api/slides/sources/research', methods=['GET'])
def slide_sources_research():
    """List research documents for slide outline source selection."""
    try:
        with get_db() as (_, cur):
            cur.execute('SELECT id, name, smart_name, category_id, doc_type, published_date, LEFT(content, 200) as snippet, created_at FROM research_documents ORDER BY created_at DESC')
            rows = cur.fetchall()
        return jsonify([{
            'id': row['id'],
            'name': row['smart_name'] or row['name'] or f"Document {row['id']}",
            'category': row['category_id'],
            'doc_type': row.get('doc_type') or 'other',
            'date': row['published_date'] or (row['created_at'].isoformat() if row['created_at'] else None),
            'snippet': (row['snippet'] or '')[:150] + ('...' if row.get('snippet') and len(row['snippet']) > 150 else ''),
        } for row in rows])
    except Exception as e:
        print(f"Error listing research sources: {e}")
        return jsonify([])


@app.route('/api/slides/sources/summaries', methods=['GET'])
def slide_sources_summaries():
    """List meeting summaries for slide outline source selection."""
    try:
        with get_db() as (_, cur):
            cur.execute('SELECT id, title, topic, topic_type, LEFT(summary, 200) as snippet, created_at FROM meeting_summaries ORDER BY created_at DESC')
            rows = cur.fetchall()
        return jsonify([{
            'id': row['id'],
            'name': row['title'] or f"Summary {row['id']}",
            'topic': row.get('topic') or 'General',
            'topic_type': row.get('topic_type') or 'other',
            'date': row['created_at'].isoformat() if row['created_at'] else None,
            'snippet': (row['snippet'] or '')[:150] + ('...' if row.get('snippet') and len(row['snippet']) > 150 else ''),
        } for row in rows])
    except Exception as e:
        print(f"Error listing summary sources: {e}")
        return jsonify([])


@app.route('/api/slides/sources/meetingprep', methods=['GET'])
def slide_sources_meetingprep():
    """List meeting prep sessions for slide outline source selection."""
    try:
        with get_db() as (_, cur):
            cur.execute('''
                SELECT m.id, m.meeting_date, m.meeting_type, m.status, m.notes,
                       c.ticker, c.name as company_name, c.sector,
                       (SELECT COUNT(*) FROM mp_documents WHERE meeting_id = m.id) as doc_count
                FROM mp_meetings m
                JOIN mp_companies c ON m.company_id = c.id
                ORDER BY m.created_at DESC
            ''')
            rows = cur.fetchall()
        return jsonify([{
            'id': row['id'],
            'name': f"{row['company_name']} ({row['ticker']})" + (f" - {row['meeting_type']}" if row.get('meeting_type') else ''),
            'ticker': row['ticker'],
            'company_name': row['company_name'],
            'date': str(row['meeting_date']) if row['meeting_date'] else None,
            'doc_count': row['doc_count'],
            'snippet': (row['notes'] or '')[:150] + ('...' if row.get('notes') and len(row['notes']) > 150 else ''),
        } for row in rows])
    except Exception as e:
        print(f"Error listing meeting prep sources: {e}")
        return jsonify([])


# ============================================
# STUDIO ENDPOINTS
# ============================================

def _gather_source_content(source_config):
    """Gather source content for studio generation. Returns (context_text, metadata)."""
    source_type = source_config.get('source_type', 'custom')
    source_ids = source_config.get('source_ids', [])
    custom_text = source_config.get('custom_text', '')
    context_parts = []
    metadata = {'source_type': source_type}

    if source_type == 'research':
        if source_ids:
            with get_db() as (conn, cur):
                placeholders = ','.join(['%s'] * len(source_ids))
                cur.execute(f'SELECT id, name, smart_name, content FROM research_documents WHERE id IN ({placeholders})', source_ids)
                rows = cur.fetchall()
            for row in rows:
                doc_name = row['smart_name'] or row['name'] or f"Document {row['id']}"
                content = (row['content'] or '')[:5000]
                context_parts.append(f"Document: {doc_name}\n{content}")

    elif source_type == 'summary':
        if source_ids:
            with get_db() as (conn, cur):
                placeholders = ','.join(['%s'] * len(source_ids))
                cur.execute(f'SELECT id, title, topic, raw_notes, summary FROM meeting_summaries WHERE id IN ({placeholders})', source_ids)
                rows = cur.fetchall()
            for row in rows:
                title = row['title'] or f"Summary {row['id']}"
                context_parts.append(f"Meeting: {title}")
                if row.get('topic'):
                    context_parts.append(f"Topic: {row['topic']}")
                if row.get('raw_notes'):
                    context_parts.append(f"Notes:\n{row['raw_notes'][:3000]}")
                if row.get('summary'):
                    context_parts.append(f"Summary:\n{row['summary'][:3000]}")

    elif source_type == 'meetingprep':
        if source_ids:
            with get_db() as (conn, cur):
                placeholders = ','.join(['%s'] * len(source_ids))
                cur.execute(f'''SELECT m.id, m.notes, c.ticker, c.name as company_name, c.sector
                               FROM mp_meetings m JOIN mp_companies c ON m.company_id = c.id
                               WHERE m.id IN ({placeholders})''', source_ids)
                meetings = cur.fetchall()
                cur.execute(f'SELECT meeting_id, filename, extracted_text FROM mp_documents WHERE meeting_id IN ({placeholders}) ORDER BY upload_order', source_ids)
                docs = cur.fetchall()
            for meeting in meetings:
                context_parts.append(f"Meeting Prep: {meeting['company_name']} ({meeting['ticker']})")
                if meeting.get('notes'):
                    context_parts.append(f"Meeting Notes:\n{meeting['notes'][:2000]}")
            for doc in docs:
                if doc.get('extracted_text'):
                    context_parts.append(f"Document ({doc['filename']}):\n{doc['extracted_text'][:3000]}")

    elif source_type == 'custom':
        if custom_text.strip():
            context_parts.append(custom_text[:10000])

    elif source_type == 'ticker':
        ticker = source_config.get('ticker', '').upper()
        if ticker:
            with get_db() as (conn, cur):
                cur.execute('SELECT * FROM stock_overviews WHERE ticker = %s', (ticker,))
                overview_row = cur.fetchone()
                cur.execute('SELECT * FROM portfolio_analyses WHERE ticker = %s', (ticker,))
                analysis_row = cur.fetchone()
            if overview_row:
                for field in ['company_overview', 'business_model', 'business_mix', 'opportunities', 'risks', 'conclusion']:
                    if overview_row.get(field):
                        context_parts.append(f"{field.replace('_', ' ').title()}:\n{overview_row[field]}")
            if analysis_row and analysis_row.get('analysis'):
                a = analysis_row['analysis']
                if isinstance(a, str):
                    try: a = json.loads(a)
                    except: a = {}
                if isinstance(a, dict):
                    context_parts.append(f"Investment Analysis:\n{json.dumps(a, indent=2)[:3000]}")

    return '\n\n'.join(context_parts), metadata


@app.route('/api/studio/init', methods=['POST'])
def init_studio_tables():
    """Manually create studio tables if they don't exist."""
    try:
        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                CREATE TABLE IF NOT EXISTS studio_design_themes (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    style_prompt TEXT NOT NULL,
                    preview_image TEXT,
                    is_default BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS studio_outputs (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    type VARCHAR(30) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    theme_id INTEGER REFERENCES studio_design_themes(id) ON DELETE SET NULL,
                    source_config JSONB DEFAULT '{}',
                    settings JSONB DEFAULT '{}',
                    content JSONB DEFAULT '{}',
                    image_data TEXT,
                    progress_current INTEGER DEFAULT 0,
                    progress_total INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS studio_slide_images (
                    id SERIAL PRIMARY KEY,
                    output_id INTEGER REFERENCES studio_outputs(id) ON DELETE CASCADE,
                    slide_number INTEGER NOT NULL,
                    image_data TEXT,
                    content_hash VARCHAR(64),
                    status VARCHAR(20) DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_studio_slide_images_output ON studio_slide_images(output_id)')
            # Seed default themes
            cur.execute('SELECT COUNT(*) as cnt FROM studio_design_themes')
            if cur.fetchone()['cnt'] == 0:
                for key, theme in SLIDE_THEMES.items():
                    cur.execute('''
                        INSERT INTO studio_design_themes (name, description, style_prompt, is_default)
                        VALUES (%s, %s, %s, %s)
                    ''', (theme['name'], theme.get('illustration_guidance', ''), theme['style_prefix'], True))
        return jsonify({'success': True, 'message': 'Studio tables created'})
    except Exception as e:
        print(f"Studio init error: {e}")
        return jsonify({'error': str(e)}), 500


STUDIO_DESIGN_THEMES = [
    {'name': 'Sketchnote', 'category': 'classic', 'colors': '#F5E6D3,#FF6B6B,#4ECDC4', 'style_prompt': "Background: Warm beige/cream colored textured paper (like aged notebook paper)\nIllustrations: Cute hand-drawn cartoon style with colorful doodles\nTitle text: Large, colorful, hand-lettered style typography (rounded, playful, multicolor gradients)\nBody text: Clean, readable text in dark gray/brown\nDecorations: Small stars, sparkles, arrows, dots, swirls scattered around\nLayout: Professional yet friendly, like a designer's sketchnote\nAspect ratio: 16:9 widescreen slide\nResolution: High quality, crisp text\nALL text MUST be in English\nDO NOT include any watermarks or AI generation notices"},
    {'name': 'Frosted Glass', 'category': 'corporate', 'colors': '#0F172A,#38BDF8,#2DD4BF', 'style_prompt': "Background: Dark navy (#0F172A) with soft gradient orbs of blue and purple floating behind frosted-glass panels\nLayout: Content inside semi-transparent white glass cards with backdrop blur and subtle 1px white/10% borders, rounded corners 16px\nTitle text: Clean white sans-serif typography (like Inter or SF Pro), bold weight\nBody text: Light gray (#CBD5E1) sans-serif\nAccent colors: Cool blue (#38BDF8) and teal (#2DD4BF) for highlights and decorative elements\nDecorations: Subtle glassmorphism layers, thin divider lines at 20% white opacity\nAspect ratio: 16:9 widescreen\nALL text MUST be in English\nDO NOT include watermarks"},
    {'name': 'Bold Editorial', 'category': 'creative', 'colors': '#FFFFFF,#1A1A1A,#DC2626', 'style_prompt': "Background: Stark white or off-white\nTitle text: Dramatic oversized serif typography (like Playfair Display) at 80-120pt in solid black, occupying 50-60% of slide area\nBody text: Light-weight sans-serif in dark gray\nAccent color: Single bold vermillion red (#DC2626) used sparingly for underlines, pull quotes, thin rules\nLayout: Strong asymmetric grid, text at unexpected positions, generous whitespace\nDecorations: Thin horizontal rules, high-contrast black-and-white elements\nPhotography style: High-contrast with tight crops\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Aurora Gradient', 'category': 'creative', 'colors': '#7C3AED,#EC4899,#14B8A6', 'style_prompt': "Background: Rich flowing mesh gradients blending deep violet (#7C3AED), magenta (#EC4899), teal (#14B8A6), and warm coral -- reminiscent of northern lights, with soft organic blobs of color\nTitle text: Clean white sans-serif bold typography with subtle text shadow for legibility\nBody text: White or very light gray sans-serif\nContent cards: Semi-transparent dark overlays (rgba black 30-40%) with soft rounded corners\nDecorations: Thin white accent lines, minimal iconography, optional subtle grain texture\nAspect ratio: 16:9 widescreen\nALL text MUST be in English\nDO NOT include watermarks"},
    {'name': 'Zen Minimal', 'category': 'minimalist', 'colors': '#F5F0EB,#8B9467,#6B7280', 'style_prompt': "Background: Warm off-white or soft cream (#F5F0EB) with subtle paper or linen texture, 60-70% intentionally blank\nTitle text: Elegant thin-weight serif typography (like Cormorant Garamond) in dark charcoal, never pure black\nBody text: Humanist sans-serif in warm gray\nPalette: Muted earth tones -- warm stone gray, soft clay, sage green (#8B9467), pale sand\nDecorations: Single carefully placed visual element per slide -- delicate ink-wash illustration or minimal line drawing, thin hairline rules\nLayout: Generous margins, radical restraint, Japanese wabi-sabi inspired\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Cyberpunk Neon', 'category': 'dark', 'colors': '#0A0A0A,#00F0FF,#FF006E', 'style_prompt': "Background: True black (#0A0A0A) or very dark blue-black (#0D1117) with subtle grid/dot matrix pattern at 5-8% opacity\nTitle text: Geometric sans-serif or monospace typography (like Space Grotesk) in neon cyan (#00F0FF) or white, with glow effects\nBody text: White or light gray monospace\nAccent colors: Electric cyan (#00F0FF), hot magenta (#FF006E), acid green (#39FF14)\nDecorations: Glowing neon line effects (box shadows with color spread), angular sharp-cornered containers\nData viz: Neon colors against dark canvas\nAspect ratio: 16:9 widescreen\nALL text MUST be in English\nDO NOT include watermarks"},
    {'name': 'Marble Luxe', 'category': 'luxury', 'colors': '#1A1A1A,#C9A96E,#FFFFFF', 'style_prompt': "Background: Dark charcoal or black marble texture with sophisticated gray and gold veining\nTitle text: Elegant high-contrast serif typography (like Didot or Bodoni) in metallic gold (#C9A96E) or white\nBody text: Light serif in white or soft gold\nAccent: Metallic gold for borders, thin decorative lines, icons, and flourishes\nDecorations: Thin gold line dividers and frame borders, generous margins, centered layouts\nContent areas: Subtle drop shadows, premium luxury feel\nAspect ratio: 16:9 widescreen\nALL text MUST be in English\nDO NOT include watermarks"},
    {'name': 'Neobrutalist', 'category': 'playful', 'colors': '#FFE600,#FF5CBE,#3B82F6', 'style_prompt': "Background: Bright saturated colors -- primary yellow (#FFE600), hot pink, electric blue -- changing per slide section\nTitle text: Bold chunky sans-serif (like Space Grotesk Black) in black or white\nBody text: Clean sans-serif in black\nBorders: Thick 3-4px black borders around ALL elements\nShadows: Hard-edge solid drop shadows offset 4-6px in black behind content cards\nDecorations: Geometric shapes (circles, rectangles, zigzag lines), NO gradients, NO blur -- everything flat and intentional\nLayout: Raw, unpolished, high contrast, generous padding\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Botanical', 'category': 'organic', 'colors': '#B7C4A0,#FAF7F2,#C2622D', 'style_prompt': "Background: Warm linen white or very pale sage (#FAF7F2)\nTitle text: Elegant serif (like Lora or DM Serif Display) in dark forest green or charcoal\nBody text: Clean humanist sans-serif in warm gray\nPalette: Soft sage green (#B7C4A0), warm cream, muted forest tones, terracotta accents (#C2622D)\nDecorations: Delicate botanical line illustrations -- leaves, ferns, eucalyptus branches, wildflowers as frames and corner accents. Occasional watercolor-wash texture in pale green or blush\nLayout: Open and airy with organic, slightly asymmetric placement, soft rounded containers\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Data Dashboard', 'category': 'analytical', 'colors': '#F8F9FA,#2563EB,#F59E0B', 'style_prompt': "Background: Clean white or very light gray (#F8F9FA) with structured multi-panel grid layout\nTitle text: Technical sans-serif (like IBM Plex Sans or Inter) bold in dark gray\nBody text: Clean sans-serif, data labels in monospace\nPrimary accent: Confident blue (#2563EB) with secondary teal, amber (#F59E0B), and coral for data differentiation\nLayout: Slide divided into 2-4 content zones separated by thin gray lines or subtle card boundaries\nDecorations: Prominent donut charts, bar graphs, sparklines, key metric callouts with large numbers, subtle gray icons, cards with very light 2px shadows\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Retro Analog', 'category': 'retro', 'colors': '#CC6B2C,#D4A843,#6B7F3E', 'style_prompt': "Background: Slightly textured surface mimicking aged paper or cardboard, warm cream (#FFF3E0)\nTitle text: Retro slab-serif typography (like Rockwell or Cooper) with rounded, friendly letterforms in burnt orange or brown\nBody text: Warm sans-serif in dark brown\nPalette: Burnt orange (#CC6B2C), mustard yellow (#D4A843), avocado green (#6B7F3E), warm brown, cream\nDecorations: Halftone dot patterns, subtle noise overlays for vintage texture, rounded rectangles and circles, hand-drawn or screen-printed illustration style\nPhotography: Warm, slightly faded vintage color grading\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Tech Futurist', 'category': 'tech', 'colors': '#0F172A,#3B82F6,#E2E8F0', 'style_prompt': "Background: Deep space navy (#0F172A) with ultra-thin geometric line art: wireframe grids, concentric circles, angular connector lines, node-network patterns in low-opacity cyan or silver\nTitle text: Clean geometric sans-serif in white, tracking slightly wider than normal\nBody text: Light sans-serif in silver/light gray (#E2E8F0)\nAccent: Electric blue (#3B82F6) and white\nDecorations: Thin line borders, micro-detail decorative elements (small crosses, dots at intersections, coordinate markers), subtle blue glow on key elements\nLayout: Mathematically precise alignment, clean structured blocks\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Pastel Cloud', 'category': 'soft', 'colors': '#E8DEFF,#DBEAFE,#FDE8EF', 'style_prompt': "Background: Soft diffused gradient blending pastel lavender (#E8DEFF), baby blue (#DBEAFE), blush pink (#FDE8EF), mint (#D1FAE5) -- gentle and cloudlike\nTitle text: Clean rounded sans-serif (like Nunito or Quicksand) medium-weight in dark gray or soft navy\nBody text: Regular weight sans-serif in medium gray\nContent cards: White or frosted-white with 16-20px border radius and very soft shadows\nDecorations: Rounded pill-shaped buttons and tags, simple friendly line icons with rounded caps\nLayout: Soft, accessible, modern, generous spacing\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Swiss Mono', 'category': 'minimalist', 'colors': '#000000,#FFFFFF,#6B7280', 'style_prompt': "Background: Pure white\nTitle text: Bold sans-serif (like Helvetica Neue Bold or Archivo Black) in pure black, large and confident\nBody text: Light-weight sans-serif in dark gray\nPalette: STRICTLY black, white, and 3-4 shades of gray -- NO color whatsoever\nLayout: Strong Swiss/International typographic grid with precise 12-column alignment, heavy whitespace\nDecorations: Thick horizontal rules, bold section dividers, large slide numbers as typographic elements\nPhotography: True black and white, high contrast\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Warm Terracotta', 'category': 'warm', 'colors': '#C2622D,#FAF7F2,#8B4513', 'style_prompt': "Background: Soft white (#FAF7F2) or light sand with subtle linen or canvas texture\nTitle text: Rounded approachable serif (like Fraunces) in rich terracotta (#C2622D) or deep clay\nBody text: Modern geometric sans-serif in warm charcoal (#3D3024)\nPalette: Terracotta/burnt sienna, deep clay (#8B4513), warm charcoal, dusty rose, sage green\nDecorations: Organic soft shapes -- irregular blobs, arched frames, rounded containers. Thin terracotta accent lines and dots\nPhotography: Warm sun-kissed color grading\nLayout: Balanced but relaxed, generous breathing room\nAspect ratio: 16:9 widescreen\nALL text MUST be in English"},
    {'name': 'Holographic', 'category': 'futuristic', 'colors': '#E0E7FF,#F0ABFC,#67E8F9', 'style_prompt': "Background: Light silver-white or soft gray base with holographic/iridescent color shifts -- rainbow refractions of pink (#F0ABFC), blue, green, purple shimmering like holographic foil\nTitle text: Clean modern sans-serif in dark charcoal or black for contrast, bold weight\nBody text: Medium sans-serif in dark gray\nAccent: Metallic silver and chrome for borders and decorative lines\nContent cards: Iridescent gradient borders or holographic background fills at low opacity\nDecorations: Subtle light-leak or lens-flare effects, minimalist centered layout with ample whitespace\nAspect ratio: 16:9 widescreen\nALL text MUST be in English\nDO NOT include watermarks"},
]


@app.route('/api/studio/seed-themes', methods=['POST'])
def seed_studio_themes():
    """Seed the expanded studio design themes."""
    try:
        with get_db(commit=True) as (conn, cur):
            # Clear existing themes
            cur.execute('DELETE FROM studio_design_themes')
            for t in STUDIO_DESIGN_THEMES:
                cur.execute('''
                    INSERT INTO studio_design_themes (name, description, style_prompt, is_default)
                    VALUES (%s, %s, %s, TRUE)
                ''', (t['name'], t.get('category', ''), t['style_prompt']))
        return jsonify({'success': True, 'count': len(STUDIO_DESIGN_THEMES)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Studio Theme CRUD ---

_THEME_COLORS = {t['name']: t['colors'] for t in STUDIO_DESIGN_THEMES}

@app.route('/api/studio/themes', methods=['GET'])
def get_studio_themes():
    try:
        with get_db() as (_, cur):
            cur.execute('SELECT id, name, description, style_prompt, is_default, preview_image IS NOT NULL as has_preview, created_at FROM studio_design_themes ORDER BY is_default DESC, name ASC')
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d['colors'] = _THEME_COLORS.get(row['name'], '')
            result.append(d)
        return jsonify(result)
    except Exception as e:
        print(f"Error listing studio themes: {e}")
        return jsonify([])


@app.route('/api/studio/themes', methods=['POST'])
def create_studio_theme():
    try:
        data = request.json
        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                INSERT INTO studio_design_themes (name, description, style_prompt)
                VALUES (%s, %s, %s) RETURNING id
            ''', (data['name'], data.get('description', ''), data['style_prompt']))
            theme_id = cur.fetchone()['id']
        return jsonify({'id': theme_id, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/themes/<int:theme_id>', methods=['PUT'])
def update_studio_theme(theme_id):
    try:
        data = request.json
        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                UPDATE studio_design_themes SET name=%s, description=%s, style_prompt=%s, updated_at=%s
                WHERE id=%s
            ''', (data['name'], data.get('description', ''), data['style_prompt'], datetime.utcnow(), theme_id))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/themes/<int:theme_id>', methods=['DELETE'])
def delete_studio_theme(theme_id):
    try:
        with get_db(commit=True) as (conn, cur):
            cur.execute('SELECT is_default FROM studio_design_themes WHERE id=%s', (theme_id,))
            row = cur.fetchone()
            if row and row['is_default']:
                return jsonify({'error': 'Cannot delete default themes'}), 400
            cur.execute('DELETE FROM studio_design_themes WHERE id=%s AND is_default=FALSE', (theme_id,))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/themes/<int:theme_id>/preview', methods=['POST'])
def generate_studio_theme_preview(theme_id):
    try:
        data = request.json or {}
        gemini_key = data.get('geminiApiKey') or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
        with get_db() as (_, cur):
            cur.execute('SELECT style_prompt, name FROM studio_design_themes WHERE id=%s', (theme_id,))
            theme = cur.fetchone()
        if not theme:
            return jsonify({'error': 'Theme not found'}), 404
        prompt = f"Generate a sample presentation slide preview showcasing this visual style:\n\n{theme['style_prompt']}\n\nCreate a sample slide with the title '{theme['name']} Theme Preview' and some placeholder content demonstrating the visual style."
        image_data = _generate_slide_image(prompt, api_key=gemini_key)
        if image_data:
            with get_db(commit=True) as (conn, cur):
                cur.execute('UPDATE studio_design_themes SET preview_image=%s, updated_at=%s WHERE id=%s', (image_data, datetime.utcnow(), theme_id))
            return jsonify({'success': True})
        return jsonify({'error': 'Failed to generate preview image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Studio Output CRUD ---

@app.route('/api/studio/outputs', methods=['GET'])
def get_studio_outputs():
    try:
        type_filter = request.args.get('type')
        with get_db() as (_, cur):
            query = 'SELECT id, title, type, status, theme_id, source_config, settings, progress_current, progress_total, error_message, created_at, updated_at FROM studio_outputs'
            params = []
            if type_filter:
                query += ' WHERE type = %s'
                params.append(type_filter)
            query += ' ORDER BY created_at DESC'
            cur.execute(query, params)
            rows = cur.fetchall()
        return jsonify([dict(row) for row in rows])
    except Exception as e:
        print(f"Error listing studio outputs: {e}")
        return jsonify([])


@app.route('/api/studio/outputs', methods=['POST'])
def create_studio_output():
    try:
        data = request.json
        with get_db(commit=True) as (conn, cur):
            cur.execute('''
                INSERT INTO studio_outputs (title, type, source_config, settings, theme_id)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            ''', (
                data['title'], data['type'],
                json.dumps(data.get('source_config', {})),
                json.dumps(data.get('settings', {})),
                data.get('theme_id')
            ))
            output_id = cur.fetchone()['id']
        return jsonify({'id': output_id, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/outputs/<int:output_id>', methods=['GET'])
def get_studio_output(output_id):
    try:
        with get_db() as (_, cur):
            cur.execute('SELECT * FROM studio_outputs WHERE id=%s', (output_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({'error': 'Output not found'}), 404
            result = dict(row)
            # Don't send image_data in the main response (it's large)
            result['image_data'] = None
            result['has_image'] = row['image_data'] is not None
            # For slides, get slide images metadata
            if row['type'] == 'slides':
                cur.execute('SELECT id, slide_number, status, content_hash, image_data IS NOT NULL as has_image FROM studio_slide_images WHERE output_id=%s ORDER BY slide_number', (output_id,))
                result['slide_images'] = [dict(si) for si in cur.fetchall()]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/outputs/<int:output_id>/image', methods=['GET'])
def get_studio_output_image(output_id):
    try:
        slide_num = request.args.get('slide', type=int)
        with get_db() as (_, cur):
            if slide_num is not None:
                cur.execute('SELECT image_data FROM studio_slide_images WHERE output_id=%s AND slide_number=%s', (output_id, slide_num))
            else:
                cur.execute('SELECT image_data FROM studio_outputs WHERE id=%s', (output_id,))
            row = cur.fetchone()
        if row and row['image_data']:
            return jsonify({'image_data': row['image_data']})
        return jsonify({'image_data': None}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/outputs/<int:output_id>', methods=['DELETE'])
def delete_studio_output(output_id):
    try:
        with get_db(commit=True) as (conn, cur):
            cur.execute('DELETE FROM studio_outputs WHERE id=%s', (output_id,))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Studio Generation ---

def _generate_studio_slides(output_id, source_content, settings, api_keys):
    """Generate slides (outline + images) in background thread."""
    import time
    try:
        slide_count = settings.get('slide_count', 30)
        theme_name = settings.get('theme_name', 'sketchnote')

        # Look up studio design theme if one is set
        custom_theme_prompt = ''
        with get_db(commit=True) as (conn, cur):
            cur.execute("SELECT theme_id FROM studio_outputs WHERE id=%s", (output_id,))
            out_row = cur.fetchone()
            if out_row and out_row.get('theme_id'):
                cur.execute("SELECT name, style_prompt FROM studio_design_themes WHERE id=%s", (out_row['theme_id'],))
                theme_row = cur.fetchone()
                if theme_row:
                    theme_name = theme_row['name'].lower().replace(' ', '')
                    custom_theme_prompt = theme_row['style_prompt']
            cur.execute("UPDATE studio_outputs SET status='generating', progress_current=0, progress_total=%s, updated_at=%s WHERE id=%s",
                        (slide_count, datetime.utcnow(), output_id))

        # Phase 1: Generate outline
        llm_prompt = f"""You are creating slide content for a presentation.

Based on this source material:

{source_content}

Generate a JSON array of exactly {slide_count} slides for a comprehensive presentation. Each slide should have:
- slide_number (integer starting at 1)
- title (string)
- type (one of: title, toc, section_divider, content, closing)
- content (detailed text describing what should appear on the slide - include specific data, numbers, quotes from the source material)
- illustration_hints (array of strings like "company", "growth", "money", "risk", "data", "leader", "opportunity")
- no_header (boolean - true only for title, section_divider, and closing slides)

Structure: title slide, table of contents, then 5-7 sections covering the key themes. Each section should have a section divider slide followed by 2-4 content slides. End with a closing slide.

IMPORTANT: Fill in actual data and specific details from the source material. Do NOT use placeholder brackets.

Return ONLY valid JSON array, no markdown fencing."""

        result = call_llm(
            messages=[{"role": "user", "content": llm_prompt}],
            tier="standard", max_tokens=8192,
            anthropic_api_key=api_keys.get('anthropic', ''),
            gemini_api_key=api_keys.get('gemini', ''),
        )
        response_text = result['text'].strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        slides_data = json.loads(response_text)

        # Store outline in content
        style_instructions = settings.get('style_instructions', '')
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET content=%s, progress_total=%s, updated_at=%s WHERE id=%s",
                        (json.dumps({'slides': slides_data, 'theme_name': theme_name, 'style_instructions': style_instructions, 'custom_theme_prompt': custom_theme_prompt}), len(slides_data), datetime.utcnow(), output_id))
            # Create slide image records
            for s in slides_data:
                content_hash = _compute_content_hash(s)
                cur.execute('''
                    INSERT INTO studio_slide_images (output_id, slide_number, content_hash, status)
                    VALUES (%s, %s, %s, 'new')
                ''', (output_id, s['slide_number'], content_hash))

        # Phase 2: Generate images
        gemini_key = api_keys.get('gemini', '')
        for i, slide in enumerate(slides_data):
            if custom_theme_prompt:
                # Use the studio design theme directly instead of built-in SLIDE_THEMES
                prompt = f"You are generating a presentation slide image.\n\nVISUAL STYLE (MUST follow exactly):\n{custom_theme_prompt}\n\nSLIDE CONTENT:\n{slide.get('content', '').strip()}"
                hints = slide.get('illustration_hints', [])
                if hints:
                    prompt += "\n\nILLUSTRATIONS TO INCLUDE:\n" + "\n".join(f"- {h}" for h in hints)
            else:
                prompt = _build_slide_prompt(slide, theme_name, '', len(slides_data))
            if style_instructions:
                prompt += f"\n\nADDITIONAL STYLE INSTRUCTIONS (follow these closely):\n{style_instructions}"
            image_data = _generate_slide_image(prompt, api_key=gemini_key)
            with get_db(commit=True) as (conn, cur):
                if image_data:
                    cur.execute("UPDATE studio_slide_images SET image_data=%s, status='done' WHERE output_id=%s AND slide_number=%s",
                                (image_data, output_id, slide['slide_number']))
                else:
                    cur.execute("UPDATE studio_slide_images SET status='error' WHERE output_id=%s AND slide_number=%s",
                                (output_id, slide['slide_number']))
                cur.execute("UPDATE studio_outputs SET progress_current=%s, status=%s, updated_at=%s WHERE id=%s",
                            (i + 1, f"generating:{i+1}:{len(slides_data)}", datetime.utcnow(), output_id))
            time.sleep(2)

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='ready', updated_at=%s WHERE id=%s", (datetime.utcnow(), output_id))

    except Exception as e:
        print(f"Studio slides generation error: {e}")
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='error', error_message=%s, updated_at=%s WHERE id=%s",
                        (str(e), datetime.utcnow(), output_id))


def _generate_studio_infographic(output_id, source_content, settings, api_keys):
    """Generate an infographic image."""
    try:
        orientation = settings.get('orientation', 'landscape')
        aspect_map = {'landscape': '16:9', 'portrait': '9:16', 'square': '1:1'}
        aspect = aspect_map.get(orientation, '16:9')

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='generating', updated_at=%s WHERE id=%s", (datetime.utcnow(), output_id))

        # Get summary points from LLM
        result = call_llm(
            messages=[{"role": "user", "content": f"Summarize the following content into key data points, statistics, and insights suitable for an infographic. Format as a clear, visual-friendly summary:\n\n{source_content[:6000]}"}],
            tier="standard", max_tokens=2048,
            anthropic_api_key=api_keys.get('anthropic', ''),
            gemini_api_key=api_keys.get('gemini', ''),
        )
        summary = result['text'].strip()

        # Get theme style
        theme_style = ''
        with get_db() as (_, cur):
            cur.execute('SELECT so.theme_id, sdt.style_prompt FROM studio_outputs so LEFT JOIN studio_design_themes sdt ON so.theme_id = sdt.id WHERE so.id=%s', (output_id,))
            row = cur.fetchone()
            if row and row.get('style_prompt'):
                theme_style = row['style_prompt']

        prompt = f"""Create a beautiful, professional infographic image.

{f'VISUAL STYLE: {theme_style}' if theme_style else 'VISUAL STYLE: Clean, modern infographic with vibrant colors, clear hierarchy, and professional typography.'}

Aspect ratio: {aspect}

CONTENT TO VISUALIZE:
{summary}

Create a visually stunning infographic that presents this information clearly with:
- A compelling title at the top
- Key statistics highlighted in large, bold numbers
- Clear sections with icons and illustrations
- A logical flow from top to bottom
- Professional color scheme
ALL text MUST be in English."""

        style_instructions = settings.get('style_instructions', '')
        if style_instructions:
            prompt += f"\n\nADDITIONAL STYLE INSTRUCTIONS (follow these closely):\n{style_instructions}"

        gemini_key = api_keys.get('gemini', '')
        key = gemini_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
        client = genai.Client(api_key=key)
        image_data = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=genai_types.ImageConfig(aspect_ratio=aspect),
                    ),
                )
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data is not None:
                        image_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                        break
                if image_data:
                    break
            except Exception as e:
                print(f"Infographic attempt {attempt+1} failed: {e}")
                import time; time.sleep((attempt + 1) * 5)

        with get_db(commit=True) as (conn, cur):
            if image_data:
                cur.execute("UPDATE studio_outputs SET status='ready', image_data=%s, content=%s, updated_at=%s WHERE id=%s",
                            (image_data, json.dumps({'orientation': orientation, 'prompt_used': prompt[:500], 'description': summary[:1000]}), datetime.utcnow(), output_id))
            else:
                cur.execute("UPDATE studio_outputs SET status='error', error_message='Failed to generate infographic image', updated_at=%s WHERE id=%s",
                            (datetime.utcnow(), output_id))

    except Exception as e:
        print(f"Studio infographic error: {e}")
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='error', error_message=%s, updated_at=%s WHERE id=%s",
                        (str(e), datetime.utcnow(), output_id))


def _generate_studio_mindmap(output_id, source_content, settings, api_keys):
    """Generate a mind map JSON structure."""
    try:
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='generating', updated_at=%s WHERE id=%s", (datetime.utcnow(), output_id))

        result = call_llm(
            messages=[{"role": "user", "content": f"""Analyze the following content and create a hierarchical mind map structure.

Return a JSON object with this exact structure:
{{
  "root": {{
    "label": "Main Topic",
    "children": [
      {{
        "label": "Branch 1",
        "children": [
          {{ "label": "Sub-topic 1.1", "children": [] }},
          {{ "label": "Sub-topic 1.2", "children": [] }}
        ]
      }},
      {{
        "label": "Branch 2",
        "children": [...]
      }}
    ]
  }}
}}

Create 4-7 main branches, each with 2-5 sub-topics. Sub-topics can have their own children (up to 3 levels deep).
Use concise, descriptive labels (3-8 words each).

SOURCE CONTENT:
{source_content[:6000]}

Return ONLY valid JSON, no markdown fencing."""}],
            tier="standard", max_tokens=4096,
            anthropic_api_key=api_keys.get('anthropic', ''),
            gemini_api_key=api_keys.get('gemini', ''),
        )
        response_text = result['text'].strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        mindmap_data = json.loads(response_text)

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='ready', content=%s, updated_at=%s WHERE id=%s",
                        (json.dumps(mindmap_data), datetime.utcnow(), output_id))

    except Exception as e:
        print(f"Studio mindmap error: {e}")
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='error', error_message=%s, updated_at=%s WHERE id=%s",
                        (str(e), datetime.utcnow(), output_id))


def _generate_studio_report(output_id, source_content, settings, api_keys):
    """Generate a markdown report."""
    try:
        length = settings.get('length', 'standard')
        length_guidance = {
            'brief': 'Keep it concise, around 500-800 words. Focus on key takeaways.',
            'standard': 'Write a thorough report, around 1500-2500 words with detailed analysis.',
            'detailed': 'Write a comprehensive, in-depth report, around 3000-5000 words with exhaustive coverage.',
        }

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='generating', updated_at=%s WHERE id=%s", (datetime.utcnow(), output_id))

        result = call_llm(
            messages=[{"role": "user", "content": f"""Write a professional report in Markdown format based on the following source material.

{length_guidance.get(length, length_guidance['standard'])}

Structure the report with:
- A clear title (# heading)
- Executive summary
- Main sections with ## headings
- Sub-sections with ### headings where needed
- Key findings, data points, and analysis
- Conclusion with actionable insights

Use bullet points, bold text, and other markdown formatting for readability.
Include specific numbers, quotes, and data from the source material.

SOURCE MATERIAL:
{source_content[:8000]}"""}],
            tier="standard", max_tokens=8192,
            anthropic_api_key=api_keys.get('anthropic', ''),
            gemini_api_key=api_keys.get('gemini', ''),
        )

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='ready', content=%s, updated_at=%s WHERE id=%s",
                        (json.dumps({'markdown': result['text'].strip()}), datetime.utcnow(), output_id))

    except Exception as e:
        print(f"Studio report error: {e}")
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='error', error_message=%s, updated_at=%s WHERE id=%s",
                        (str(e), datetime.utcnow(), output_id))


def _generate_studio_quiz(output_id, source_content, settings, api_keys):
    """Generate quiz questions."""
    try:
        question_count = settings.get('question_count', 10)
        difficulty = settings.get('difficulty', 'medium')

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='generating', updated_at=%s WHERE id=%s", (datetime.utcnow(), output_id))

        result = call_llm(
            messages=[{"role": "user", "content": f"""Create {question_count} multiple-choice quiz questions based on the following content.
Difficulty level: {difficulty}

Return a JSON object with this structure:
{{
  "questions": [
    {{
      "id": 1,
      "question": "What is...",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct": 0,
      "explanation": "The correct answer is A because..."
    }}
  ]
}}

Rules:
- Each question must have exactly 4 options
- "correct" is the 0-based index of the correct answer
- Include a brief explanation for each answer
- Questions should test understanding, not just recall
- Vary question types: factual, conceptual, analytical
- Make incorrect options plausible but clearly wrong

SOURCE CONTENT:
{source_content[:6000]}

Return ONLY valid JSON, no markdown fencing."""}],
            tier="standard", max_tokens=4096,
            anthropic_api_key=api_keys.get('anthropic', ''),
            gemini_api_key=api_keys.get('gemini', ''),
        )
        response_text = result['text'].strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        quiz_data = json.loads(response_text)

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='ready', content=%s, updated_at=%s WHERE id=%s",
                        (json.dumps(quiz_data), datetime.utcnow(), output_id))

    except Exception as e:
        print(f"Studio quiz error: {e}")
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='error', error_message=%s, updated_at=%s WHERE id=%s",
                        (str(e), datetime.utcnow(), output_id))


def _generate_studio_flashcard(output_id, source_content, settings, api_keys):
    """Generate flashcards."""
    try:
        card_count = settings.get('card_count', 20)

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='generating', updated_at=%s WHERE id=%s", (datetime.utcnow(), output_id))

        result = call_llm(
            messages=[{"role": "user", "content": f"""Create {card_count} flashcards based on the following content.

Return a JSON object with this structure:
{{
  "cards": [
    {{
      "id": 1,
      "front": "Question or concept to remember",
      "back": "Answer or explanation",
      "category": "Topic Category"
    }}
  ]
}}

Rules:
- Front should be a clear question or key term/concept
- Back should be a concise but complete answer
- Category should group related cards together
- Cover the most important concepts from the material
- Mix factual recall, definitions, and conceptual questions
- Keep answers concise but informative

SOURCE CONTENT:
{source_content[:6000]}

Return ONLY valid JSON, no markdown fencing."""}],
            tier="standard", max_tokens=4096,
            anthropic_api_key=api_keys.get('anthropic', ''),
            gemini_api_key=api_keys.get('gemini', ''),
        )
        response_text = result['text'].strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        flashcard_data = json.loads(response_text)

        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='ready', content=%s, updated_at=%s WHERE id=%s",
                        (json.dumps(flashcard_data), datetime.utcnow(), output_id))

    except Exception as e:
        print(f"Studio flashcard error: {e}")
        with get_db(commit=True) as (conn, cur):
            cur.execute("UPDATE studio_outputs SET status='error', error_message=%s, updated_at=%s WHERE id=%s",
                        (str(e), datetime.utcnow(), output_id))


_STUDIO_GENERATORS = {
    'slides': _generate_studio_slides,
    'infographic': _generate_studio_infographic,
    'mindmap': _generate_studio_mindmap,
    'report': _generate_studio_report,
    'quiz': _generate_studio_quiz,
    'flashcard': _generate_studio_flashcard,
}


@app.route('/api/studio/outputs/<int:output_id>/generate', methods=['POST'])
def generate_studio_output(output_id):
    """Universal generation endpoint. Dispatches to type-specific generator in background thread."""
    try:
        data = request.json or {}
        with get_db() as (_, cur):
            cur.execute('SELECT * FROM studio_outputs WHERE id=%s', (output_id,))
            output = cur.fetchone()
        if not output:
            return jsonify({'error': 'Output not found'}), 404

        output_type = output['type']
        generator = _STUDIO_GENERATORS.get(output_type)
        if not generator:
            return jsonify({'error': f'Unknown output type: {output_type}'}), 400

        source_config = output['source_config'] if isinstance(output['source_config'], dict) else json.loads(output['source_config'] or '{}')
        settings = output['settings'] if isinstance(output['settings'], dict) else json.loads(output['settings'] or '{}')
        source_content, _ = _gather_source_content(source_config)
        if not source_content.strip():
            # Auto-generate from title alone when no source content provided
            source_content = f"Generate content about: {output['title']}"

        api_keys = {
            'anthropic': data.get('apiKey', '') or os.environ.get('ANTHROPIC_API_KEY', ''),
            'gemini': data.get('geminiApiKey', '') or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', ''),
        }

        import threading
        thread = threading.Thread(target=generator, args=(output_id, source_content, settings, api_keys))
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'message': f'Generation started for {output_type}'})
    except Exception as e:
        print(f"Error starting studio generation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/outputs/<int:output_id>/update-slide/<int:slide_num>', methods=['PUT'])
def update_studio_slide(output_id, slide_num):
    """Update a single slide's content (title, content, illustration_hints)."""
    try:
        data = request.json or {}
        with get_db() as (_, cur):
            cur.execute('SELECT content FROM studio_outputs WHERE id=%s', (output_id,))
            output = cur.fetchone()
        if not output:
            return jsonify({'error': 'Output not found'}), 404
        content = output['content'] if isinstance(output['content'], dict) else json.loads(output['content'] or '{}')
        slides = content.get('slides', [])
        updated = False
        for s in slides:
            if s['slide_number'] == slide_num:
                if 'title' in data:
                    s['title'] = data['title']
                if 'content' in data:
                    s['content'] = data['content']
                if 'illustration_hints' in data:
                    s['illustration_hints'] = data['illustration_hints']
                updated = True
                break
        if not updated:
            return jsonify({'error': f'Slide {slide_num} not found'}), 404
        with get_db(commit=True) as (conn, cur):
            cur.execute('UPDATE studio_outputs SET content=%s, updated_at=%s WHERE id=%s',
                        (json.dumps(content), datetime.utcnow(), output_id))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/outputs/<int:output_id>/regenerate-slide/<int:slide_num>', methods=['POST'])
def regenerate_studio_slide(output_id, slide_num):
    """Regenerate a single slide image with optional edit prompt."""
    try:
        data = request.json or {}
        gemini_key = data.get('geminiApiKey', '') or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
        edit_prompt = data.get('edit_prompt', '')

        with get_db() as (_, cur):
            cur.execute('SELECT content FROM studio_outputs WHERE id=%s', (output_id,))
            output = cur.fetchone()
        if not output:
            return jsonify({'error': 'Output not found'}), 404

        content = output['content'] if isinstance(output['content'], dict) else json.loads(output['content'] or '{}')
        slides = content.get('slides', [])
        theme_name = content.get('theme_name', 'sketchnote')
        custom_theme_prompt = content.get('custom_theme_prompt', '')
        slide_data = next((s for s in slides if s['slide_number'] == slide_num), None)
        if not slide_data:
            return jsonify({'error': f'Slide {slide_num} not found'}), 404

        if custom_theme_prompt:
            prompt = f"You are generating a presentation slide image.\n\nVISUAL STYLE (MUST follow exactly):\n{custom_theme_prompt}\n\nSLIDE CONTENT:\n{slide_data.get('content', '').strip()}"
            hints = slide_data.get('illustration_hints', [])
            if hints:
                prompt += "\n\nILLUSTRATIONS TO INCLUDE:\n" + "\n".join(f"- {h}" for h in hints)
        else:
            prompt = _build_slide_prompt(slide_data, theme_name, '', len(slides))
        style_instructions = content.get('style_instructions', '')
        if style_instructions:
            prompt += f"\n\nADDITIONAL STYLE INSTRUCTIONS (follow these closely):\n{style_instructions}"
        if edit_prompt:
            prompt += f"\n\nADDITIONAL INSTRUCTIONS: {edit_prompt}"

        image_data = _generate_slide_image(prompt, api_key=gemini_key)
        if image_data:
            with get_db(commit=True) as (conn, cur):
                cur.execute("UPDATE studio_slide_images SET image_data=%s, status='done', content_hash=%s WHERE output_id=%s AND slide_number=%s",
                            (_compute_content_hash(slide_data), image_data, output_id, slide_num))
            return jsonify({'success': True, 'image_data': image_data})
        return jsonify({'error': 'Failed to generate image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/studio/outputs/<int:output_id>/export', methods=['POST'])
def export_studio_output(output_id):
    """Export studio output as PDF or downloadable file."""
    try:
        data = request.json or {}
        fmt = data.get('format', 'pdf')

        with get_db() as (_, cur):
            cur.execute('SELECT * FROM studio_outputs WHERE id=%s', (output_id,))
            output = cur.fetchone()
        if not output:
            return jsonify({'error': 'Output not found'}), 404

        content = output['content'] if isinstance(output['content'], dict) else json.loads(output['content'] or '{}')

        if output['type'] == 'slides' and fmt == 'pdf':
            with get_db() as (_, cur):
                cur.execute('SELECT image_data FROM studio_slide_images WHERE output_id=%s AND image_data IS NOT NULL ORDER BY slide_number', (output_id,))
                images = cur.fetchall()
            if not images:
                return jsonify({'error': 'No slide images available for export'}), 400
            import img2pdf
            from io import BytesIO
            pdf_images = []
            for img_row in images:
                pdf_images.append(base64.b64decode(img_row['image_data']))
            pdf_bytes = img2pdf.convert(pdf_images)
            return Response(
                pdf_bytes,
                mimetype='application/pdf',
                headers={'Content-Disposition': f'attachment; filename="{output["title"]}.pdf"'}
            )

        elif output['type'] == 'report':
            markdown_text = content.get('markdown', '')
            return Response(
                markdown_text,
                mimetype='text/markdown',
                headers={'Content-Disposition': f'attachment; filename="{output["title"]}.md"'}
            )

        elif output['type'] == 'infographic':
            if output['image_data']:
                img_bytes = base64.b64decode(output['image_data'])
                return Response(
                    img_bytes,
                    mimetype='image/png',
                    headers={'Content-Disposition': f'attachment; filename="{output["title"]}.png"'}
                )
            return jsonify({'error': 'No image data available'}), 400

        else:
            # For quiz, flashcard, mindmap — export as JSON
            return Response(
                json.dumps(content, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename="{output["title"]}.json"'}
            )

    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'database': 'postgresql'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("TDL Equity Analyzer - Backend Server")
    print("With PostgreSQL Database Support")
    print("=" * 50)
    print(f"Starting server on http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
