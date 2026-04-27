#!/usr/bin/env bash
# Migrate Charlie agent secrets from the launchd plist EnvironmentVariables
# block into the macOS user Keychain. After this runs, the plist no longer
# carries plaintext secrets — the agent reads them at startup via
# `security find-generic-password`.
#
# Idempotent: re-running updates Keychain entries to whatever is currently in
# the plist. Safe to run multiple times.
#
# Usage:
#   bash scripts/migrate-secrets-to-keychain.sh
#
# After it succeeds, restart the agent:
#   launchctl kickstart -k gui/$(id -u)/com.charlie.local-agent

set -euo pipefail

SERVICE="charlie-agent"
PLIST="${HOME}/Library/LaunchAgents/com.charlie.local-agent.plist"
PLIST_BAK="${PLIST}.pre-keychain-bak"

if [ ! -f "$PLIST" ]; then
  echo "ERROR: plist not found at $PLIST"
  echo "Run scripts/install-launchagent.sh first to create the agent."
  exit 1
fi

# Read each secret from the plist using PlistBuddy; missing keys are silently skipped.
read_plist() {
  /usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:$1" "$PLIST" 2>/dev/null || true
}

write_keychain() {
  local account="$1"
  local secret="$2"
  if [ -z "$secret" ]; then
    echo "  (skip $account — empty)"
    return
  fi
  # -U updates if entry exists; otherwise adds.
  security add-generic-password -U -s "$SERVICE" -a "$account" -w "$secret" >/dev/null 2>&1 || \
    security add-generic-password    -s "$SERVICE" -a "$account" -w "$secret" >/dev/null 2>&1
  echo "  ✓ $account stored in Keychain"
}

echo "→ Backing up plist to $PLIST_BAK"
cp "$PLIST" "$PLIST_BAK"

echo "→ Migrating secrets to Keychain (service=$SERVICE)"
for KEY in CHARLIE_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID; do
  VAL="$(read_plist "$KEY")"
  write_keychain "$KEY" "$VAL"
done

echo "→ Stripping secrets from plist"
for KEY in CHARLIE_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID; do
  /usr/libexec/PlistBuddy -c "Delete :EnvironmentVariables:$KEY" "$PLIST" 2>/dev/null || true
done

echo "→ Done."
echo ""
echo "Verify with:"
echo "  security find-generic-password -s $SERVICE -a CHARLIE_API_KEY -w"
echo ""
echo "Restart the agent so it picks up the new secret source:"
echo "  launchctl kickstart -k gui/\$(id -u)/com.charlie.local-agent"
echo ""
echo "Backup of original plist (with plaintext secrets) is at:"
echo "  $PLIST_BAK"
echo "  → after you confirm the agent works, shred it: rm \"$PLIST_BAK\""
