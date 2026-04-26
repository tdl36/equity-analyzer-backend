#!/bin/bash
# Install Charlie Local Agent as a proper macOS LaunchAgent.
#
# Replaces the legacy CharlieAgent.app + Terminal indirection with a
# direct launchd → python3 setup. Requires Full Disk Access granted to
# python3.9 binary (one-time, via System Settings).
#
# Usage:  bash scripts/install-launchagent.sh

set -e

PLIST="$HOME/Library/LaunchAgents/com.charlie.local-agent.plist"
APP_DIR="$HOME/Projects/equity-analyzer-backend"
RUN_SCRIPT="$HOME/Applications/CharlieAgent.app/Contents/MacOS/run"
LOG="$HOME/Library/Logs/charlie-agent.log"

# Pull API keys out of the existing CharlieAgent.app run script so we
# don't ask the user to paste them again. Falls back to current env.
extract_var() {
  local name="$1"
  local val=""
  if [[ -f "$RUN_SCRIPT" ]]; then
    val=$(grep -oE "${name}=[^[:space:]]+" "$RUN_SCRIPT" | head -1 | sed -E "s/^${name}=//")
  fi
  if [[ -z "$val" ]]; then
    val="${!name}"
  fi
  printf '%s' "$val"
}

CHARLIE_API_KEY=$(extract_var CHARLIE_API_KEY)
ANTHROPIC_API_KEY=$(extract_var ANTHROPIC_API_KEY)
TELEGRAM_BOT_TOKEN=$(extract_var TELEGRAM_BOT_TOKEN)
TELEGRAM_CHAT_ID=$(extract_var TELEGRAM_CHAT_ID)
PYTHONPATH_VAL="$HOME/Library/Python/3.9/lib/python/site-packages"

if [[ -z "$CHARLIE_API_KEY" ]] || [[ -z "$ANTHROPIC_API_KEY" ]]; then
  echo "ERROR: Could not find CHARLIE_API_KEY or ANTHROPIC_API_KEY in $RUN_SCRIPT or environment." >&2
  echo "Either keep the existing CharlieAgent.app or export the keys before running this script." >&2
  exit 1
fi

# Stop any current agent (Terminal-launched python and/or launchd-managed)
pkill -9 -f charlie_local_agent.py 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.charlie.local-agent" 2>/dev/null || true
sleep 2

mkdir -p "$(dirname "$PLIST")"
mkdir -p "$(dirname "$LOG")"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.charlie.local-agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Library/Developer/CommandLineTools/usr/bin/python3</string>
    <string>${APP_DIR}/charlie_local_agent.py</string>
  </array>
  <key>WorkingDirectory</key><string>${APP_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CHARLIE_API_KEY</key><string>${CHARLIE_API_KEY}</string>
    <key>ANTHROPIC_API_KEY</key><string>${ANTHROPIC_API_KEY}</string>
    <key>TELEGRAM_BOT_TOKEN</key><string>${TELEGRAM_BOT_TOKEN}</string>
    <key>TELEGRAM_CHAT_ID</key><string>${TELEGRAM_CHAT_ID}</string>
    <key>PYTHONPATH</key><string>${PYTHONPATH_VAL}</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>LimitLoadToSessionType</key><string>Aqua</string>
  <!-- Python's logging FileHandler already writes ${LOG} directly. If we
       point StandardOutPath at the same file, every line gets duplicated
       (StreamHandler -> launchd captures stdout -> writes to file too).
       Discard stdout/stderr; rely on the Python file handler. -->
  <key>StandardOutPath</key><string>/dev/null</string>
  <key>StandardErrorPath</key><string>/tmp/charlie-agent.stderr.log</string>
</dict>
</plist>
EOF

chmod 600 "$PLIST"

# Load it
launchctl bootstrap "gui/$(id -u)" "$PLIST"

sleep 5

if pgrep -f charlie_local_agent.py >/dev/null 2>&1; then
  echo "OK: agent running."
  pgrep -lf charlie_local_agent.py
  echo "--- last 3 log lines ---"
  tail -3 "$LOG" 2>/dev/null || echo "(log empty)"
else
  echo "WARN: agent not running. Check $LOG"
  tail -20 "$LOG" 2>/dev/null
fi

echo ""
echo "=== One-time FDA grant required ==="
echo "If you haven't done this already:"
echo "  1. System Settings → Privacy & Security → Full Disk Access"
echo "  2. Click +, press Cmd+Shift+G, paste:"
echo "       /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9"
echo "  3. Toggle the entry ON"
echo "  4. Restart agent: launchctl kickstart -k gui/\$(id -u)/com.charlie.local-agent"
echo ""
echo "To uninstall later:"
echo "  launchctl bootout gui/\$(id -u)/com.charlie.local-agent && rm $PLIST"
