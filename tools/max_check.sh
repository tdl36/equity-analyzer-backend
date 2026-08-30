#!/bin/bash
# Regenerate the worst-case fixture and report layout failures for each view.
set -uo pipefail
cd "$(dirname "$0")/.."
./.venv/bin/python tools/make_max_fixture.py >/dev/null
./tools/refresh_harness.sh >/dev/null 2>&1
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TOTAL=0
for V in onepager twopager memo; do
  "$CHROME" --headless --disable-gpu --no-sandbox --virtual-time-budget=15000 \
    --window-size=1400,1900 --dump-dom \
    "http://127.0.0.1:8899/index.html?view=$V&fixture=max&chrome=1&print=1&qa=1" \
    2>/dev/null > /tmp/ddpdf/maxchk_$V.html
  N=$(python3 - "$V" <<'PY'
import re,html,json,sys
v=sys.argv[1]
d=open(f'/tmp/ddpdf/maxchk_{v}.html').read()
q=re.search(r'<pre id="__qa">(.*?)</pre>', d, re.S)
qa=json.loads(html.unescape(q.group(1))) if q else []
iss=[i for pg in qa for i in pg.get('issues',[])]
print(len(iss))
for i in iss[:4]: print('       ', i[:88], file=sys.stderr)
PY
)
  echo "  max/$V: $N issue(s)"
  TOTAL=$((TOTAL+N))
done
echo "TOTAL=$TOTAL"
