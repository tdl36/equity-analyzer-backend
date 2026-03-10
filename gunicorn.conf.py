# Gunicorn configuration for Render deployment
# Extended timeout to support long-running Claude API calls with heartbeat streaming
timeout = 300
workers = 1
threads = 16  # Single process so in-memory state (transcription jobs) is shared across all threads.
              # 16 threads handles plenty of concurrent I/O-bound requests (Claude/Gemini API calls).
