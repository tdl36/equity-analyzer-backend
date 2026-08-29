#!/bin/bash
# End-to-end print check: does each Deep Dive view actually PAINT?
#
# Exists because the failure it guards against is invisible to every other
# check: a blanket `visibility: hidden` in an unrelated print stylesheet left
# the artifact laid out perfectly -- right page size, getBBox() correct for
# every element -- while painting nothing. Only rendering a real PDF and
# looking for extractable text catches that.
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || { echo "SKIP: Chrome not installed"; exit 0; }
./tools/refresh_harness.sh >/dev/null
PORT=8899
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/index.html"; then
  (cd /tmp/ddh && python3 -m http.server $PORT >/dev/null 2>&1 &)
  sleep 2
fi
FAIL=0
for SPEC in "onepager:1" "twopager:2" "memo:3"; do
  V="${SPEC%%:*}"; WANT="${SPEC##*:}"
  OUT="/tmp/ddpdf/smoke_$V.pdf"; mkdir -p /tmp/ddpdf
  "$CHROME" --headless --disable-gpu --no-sandbox --virtual-time-budget=12000 \
    --print-to-pdf="$OUT" --no-pdf-header-footer \
    "http://127.0.0.1:$PORT/index.html?view=$V&fixture=de&chrome=1&print=1" >/dev/null 2>&1
  PAGES=$(pdfinfo "$OUT" 2>/dev/null | awk '/^Pages/{print $2}')
  CHARS=$(pdftotext "$OUT" - 2>/dev/null | wc -c | tr -d ' ')
  if [ "$PAGES" != "$WANT" ] || [ "$CHARS" -lt 1000 ]; then
    echo "FAIL $V: pages=$PAGES (want $WANT) textchars=$CHARS (want >=1000)"; FAIL=1
  else
    echo "ok   $V: pages=$PAGES textchars=$CHARS"
  fi
done
exit $FAIL
