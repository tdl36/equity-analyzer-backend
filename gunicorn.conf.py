# Gunicorn configuration for Render deployment
# Extended timeout to support long-running Claude API calls with heartbeat streaming
timeout = 300
workers = 4
threads = 4  # Each worker handles 4 concurrent requests (I/O-bound API calls)
