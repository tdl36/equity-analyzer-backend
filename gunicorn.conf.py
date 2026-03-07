# Gunicorn configuration for Render deployment
# Extended timeout to support long-running Claude API calls with heartbeat streaming
timeout = 300
workers = 2
