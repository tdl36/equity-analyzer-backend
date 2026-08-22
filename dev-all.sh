#!/bin/bash
# Launch BOTH Charlie dev servers together. Ctrl+C stops both.
#   ./dev-all.sh
# (In VS Code, use Terminal > Run Build Task / Cmd+Shift+B → "dev: all" instead.)
cd "$(dirname "$0")"
echo "▸ frontend → http://127.0.0.1:3000/?local=1   |   backend → http://127.0.0.1:5000"
echo "  (Ctrl+C stops both)"
node dev.mjs & FE=$!
./run-local.sh & BE=$!
trap 'echo; echo "stopping…"; kill $FE $BE 2>/dev/null; pkill -f "tailwindcss.*--watch" 2>/dev/null' INT TERM EXIT
wait
