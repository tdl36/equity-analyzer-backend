#!/bin/bash
# Run the Charlie backend LOCALLY against the local Postgres (charlie_test),
# in Flask debug mode (auto-reloads on save). For local dev only — not deploy.
#
#   ./run-local.sh          →  http://127.0.0.1:5000
#
# Pairs with `npm run dev` (frontend on :3000). Flip the frontend to the local
# backend by running this in the browser console, then reload:
#   localStorage.setItem('charlie_local_backend','1')   // '0' or removeItem = back to prod
set -euo pipefail
cd "$(dirname "$0")"

# Local database (prod data copied down)
export DATABASE_URL="postgresql://tonydlee@127.0.0.1:5432/charlie_test"

# Secrets from Keychain (service 'charlie-agent') — same ones the local agent uses.
kc() { security find-generic-password -s charlie-agent -a "$1" -w 2>/dev/null || true; }
export ANTHROPIC_API_KEY="$(kc ANTHROPIC_API_KEY)"
export GEMINI_API_KEY="$(kc GEMINI_API_KEY)"
export GOOGLE_API_KEY="${GEMINI_API_KEY}"
export CHARLIE_API_KEY="$(kc CHARLIE_API_KEY)"
export TELEGRAM_BOT_TOKEN="$(kc TELEGRAM_BOT_TOKEN)"
export TELEGRAM_CHAT_ID="$(kc TELEGRAM_CHAT_ID)"

# Local-dev toggles
export APSCHEDULER_DISABLED=1   # don't run the media-tracker scheduler locally
export FLASK_DEBUG=1
export PORT=5000

# flask CLI gives the auto-reloader (app_v3.py hardcodes app.run(debug=False)).
exec .venv/bin/flask --app app_v3 run --host 127.0.0.1 --port 5000 --debug
