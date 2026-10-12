#!/usr/bin/env bash
# Reach the avatar from another PC or phone over the internet (the microphone needs HTTPS).
# The Linux counterpart of scripts/tunnel.ps1: starts the backend with random owner/customer tokens and
# opens a free Cloudflare "quick tunnel" (no account) to it. The links change every time; traffic passes
# through Cloudflare. They are printed and saved (owner only readable) in logs/tunnel_url.txt and
# logs/customer_url.txt. Stop with Ctrl+C, or with --detach it keeps running until scripts/run.sh --stop.
#
#   scripts/tunnel.sh --profile avatar
#   scripts/tunnel.sh --profile avatar --detach
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROFILE=avatar PORT=8000 DETACH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

# cloudflared (Apache-2.0): a pinned release, checked against its published SHA-256
CF_VERSION=2026.10.0
CF_SHA256=d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db
CF="$ROOT/tools/cloudflared/cloudflared-$CF_VERSION"
if [ ! -x "$CF" ]; then
  mkdir -p "$(dirname "$CF")"
  echo "downloading cloudflared $CF_VERSION..."
  curl -sfL -o "$CF.part" "https://github.com/cloudflare/cloudflared/releases/download/$CF_VERSION/cloudflared-linux-amd64"
  echo "$CF_SHA256  $CF.part" | sha256sum -c --quiet - || { rm -f "$CF.part"; echo "cloudflared checksum mismatch" >&2; exit 1; }
  chmod +x "$CF.part" && mv "$CF.part" "$CF"
fi

export AVATAR_ACCESS_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')"
export AVATAR_CUSTOMER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')"
scripts/run.sh --profile "$PROFILE" --port "$PORT"

LOG=logs/cloudflared.log
URL=""
for attempt in 1 2 3; do
  : > "$LOG"
  setsid "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 < /dev/null &
  CF_PID=$!
  for _ in $(seq 1 60); do
    sleep 1
    kill -0 "$CF_PID" 2>/dev/null || break
    # not api.trycloudflare.com: that one appears in the error when Cloudflare does not answer
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | grep -v '//api\.' | head -1 || true)"
    [ -n "$URL" ] && break
  done
  [ -n "$URL" ] && break
  kill "$CF_PID" 2>/dev/null || true
  echo "quick tunnel attempt $attempt failed (Cloudflare did not answer), see $LOG"
  sleep 5
done
[ -n "$URL" ] || { echo "tunnel did not start after 3 attempts; see $LOG" >&2; exit 1; }
echo "$CF_PID" > logs/tunnel.pid
umask 077
echo "$URL/?token=$AVATAR_ACCESS_TOKEN" > logs/tunnel_url.txt
echo "$URL/?token=$AVATAR_CUSTOMER_TOKEN" > logs/customer_url.txt
echo
echo "Owner link (settings, leads):       $URL/?token=$AVATAR_ACCESS_TOKEN"
echo "Customer link (conversation only):  $URL/?token=$AVATAR_CUSTOMER_TOKEN"
echo "(use headphones; allow the microphone when the browser asks)"
if [ "$DETACH" = 1 ]; then echo "tunnel pid $CF_PID runs in the background; stop everything with scripts/run.sh --stop"; exit 0; fi
echo "tunnel pid $CF_PID - press Ctrl+C to stop"
trap 'kill "$CF_PID" 2>/dev/null; rm -f logs/tunnel.pid' EXIT
wait "$CF_PID"
