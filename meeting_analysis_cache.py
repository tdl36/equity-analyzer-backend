"""Reuse unchanged source analysis; never cache meeting-specific synthesis."""
import hashlib,json

def key(ticker,prompt,content,models):
    raw=json.dumps([ticker,prompt,content,models,'meeting-analysis-v1'],sort_keys=True,default=str)
    return 'meeting_source_cache:'+hashlib.sha256(raw.encode()).hexdigest()

def get(get_db,cache_key):
    try:
        with get_db() as (_,cur):
            cur.execute("SELECT value FROM app_settings WHERE key=%s AND updated_at>NOW()-INTERVAL '30 days'",(cache_key,));row=cur.fetchone()
        value=json.loads(row['value']) if row and isinstance(row['value'],str) else row['value'] if row else None
        return value if isinstance(value,dict) else None
    except Exception:return None

def put(get_db,cache_key,analysis):
    if not isinstance(analysis,dict):return
    try:
        with get_db(commit=True) as (_,cur):
            cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(cache_key,json.dumps(analysis)))
    except Exception:pass  # A cache outage must not discard a completed paid analysis.
