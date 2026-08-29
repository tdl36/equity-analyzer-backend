#!/bin/bash
# Rebuild the golden-render harness with cache-proof asset names.
#
# Browser caching cost several wasted debugging cycles: CSS and JS were being
# served stale while measurements were taken against them, producing readings
# that contradicted the source. Unique filenames per build make that impossible.
set -euo pipefail
cd "$(dirname "$0")/.."
TS=$(date +%s)
npx esbuild tools/deepdive_harness.jsx --bundle --outfile="/tmp/ddh/h_$TS.js" \
  --loader:.jsx=jsx --loader:.json=json --format=iife \
  --define:process.env.NODE_ENV='"production"' --log-level=error
npx tailwindcss -i src/tailwind-input.css -o "/tmp/ddh/tw_$TS.css" --minify 2>/dev/null | tail -1
cat > /tmp/ddh/index.html <<HTML
<!doctype html><html><head><meta charset="utf-8">
<title>Deep Dive golden render</title>
<link rel="stylesheet" href="./tw_$TS.css">
<style>body{margin:0;background:#555;} #root{display:inline-block;}</style>
</head><body><div id="root"></div><script src="./h_$TS.js"></script></body></html>
HTML
# keep the last few builds only
ls -t /tmp/ddh/h_*.js 2>/dev/null | tail -n +4 | xargs -r rm -f
ls -t /tmp/ddh/tw_*.css 2>/dev/null | tail -n +4 | xargs -r rm -f
echo "harness refreshed: h_$TS.js + tw_$TS.css"
