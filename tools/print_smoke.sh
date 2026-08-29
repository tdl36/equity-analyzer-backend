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
FIXTURE="${FIXTURE:-de}"
for SPEC in "onepager:1" "twopager:2" "memo:3"; do
  V="${SPEC%%:*}"; WANT="${SPEC##*:}"
  OUT="/tmp/ddpdf/smoke_$V.pdf"; mkdir -p /tmp/ddpdf
  "$CHROME" --headless --disable-gpu --no-sandbox --virtual-time-budget=12000 \
    --print-to-pdf="$OUT" --no-pdf-header-footer \
    "http://127.0.0.1:$PORT/index.html?view=$V&fixture=$FIXTURE&chrome=1&print=1" >/dev/null 2>&1
  PAGES=$(pdfinfo "$OUT" 2>/dev/null | awk '/^Pages/{print $2}')
  CHARS=$(pdftotext "$OUT" - 2>/dev/null | wc -c | tr -d ' ')
  if [ "$PAGES" != "$WANT" ] || [ "$CHARS" -lt 1000 ]; then
    echo "FAIL $V: pages=$PAGES (want $WANT) textchars=$CHARS (want >=1000)"; FAIL=1
  else
    echo "ok   $V: pages=$PAGES textchars=$CHARS"
  fi
  # The memo lost signpost rows off the bottom of page 2 without any other
  # signal, so count them explicitly rather than trusting the page count.
  # Text printed on top of other text is invisible to any DOM check: the
  # preflight measures screen media, and even a print-media preview does not
  # paginate the way printing does. Measure the artifact itself.
  # pdf_overlap exits non-zero when it finds collisions; under set -e that
  # would abort the sweep instead of reporting the rest.
  OV=$(./.venv/bin/python tools/pdf_overlap.py "$OUT" 2>&1 | tail -1) || true
  case "$OV" in
    *"0 colliding"*) : ;;
    *) echo "FAIL $V: $OV"; (./.venv/bin/python tools/pdf_overlap.py "$OUT" 2>&1 | head -5) || true; FAIL=1 ;;
  esac
  if [ "$V" = "memo" ]; then
    ROWS=$(pdftotext -f 2 -l 2 "$OUT" - 2>/dev/null \
      | tr -d " \t" | grep -ci "sustainedthrough\|>84M\|breakeven\|signpost" || true)
    SIGN=$(python3 tools/count_signposts.py "$OUT" "$FIXTURE" 2>/dev/null || echo "?")
    echo "     signpost rows rendered: $SIGN"
    [ "$SIGN" = "6" ] || { echo "FAIL memo: expected 6 signposts, got $SIGN"; FAIL=1; }
  fi
done
exit $FAIL
