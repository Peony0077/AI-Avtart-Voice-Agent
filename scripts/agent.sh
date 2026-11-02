#!/usr/bin/env bash
# HPN's AI avatar at https://agent.h-townparty.com: starts the backend (profile hpn) with FIXED owner and
# customer links, and keeps this PC's tunnel to HPN's droplet open (its Caddy serves the name and passes
# everything here; the droplet has no GPU). One-time droplet setup: deploy/hpn-edge/install.sh --apply.
#
#   scripts/agent.sh                 # start (in the background); prints the two links
#   scripts/agent.sh --profile avatar
#   scripts/run.sh --stop            # stop everything (backend, tunnel)
#
# The tokens are made once and kept in data/agent_tokens.env (owner-only file): the links stay the same
# across restarts. Delete that file to make new links (the old ones stop working).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROFILE=hpn PORT=8000
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
SITE=https://agent.h-townparty.com
TOKENS=data/agent_tokens.env
mkdir -p data logs
if [ ! -s "$TOKENS" ]; then
  ( umask 077
    printf 'AVATAR_ACCESS_TOKEN=%s\nAVATAR_CUSTOMER_TOKEN=%s\n' \
      "$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')" \
      "$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')" > "$TOKENS" )
fi
set -a; . "./$TOKENS"; set +a
[ -f "$HOME/.ssh/hpn_agent_tunnel" ] || ssh-keygen -q -t ed25519 -N "" -C "agent.h-townparty.com tunnel" \
  -f "$HOME/.ssh/hpn_agent_tunnel"

scripts/run.sh --profile "$PROFILE" --port "$PORT"      # (stops a running backend and quick tunnel first)

setsid scripts/agent_tunnel_loop.sh "$PORT" >>logs/agent_tunnel.log 2>&1 < /dev/null &
echo $! > logs/agent_tunnel.pid
umask 077
rm -f logs/agent_url.txt logs/agent_customer_url.txt
echo "$SITE/?token=$AVATAR_ACCESS_TOKEN" > logs/agent_url.txt
echo "$SITE/?token=$AVATAR_CUSTOMER_TOKEN" > logs/agent_customer_url.txt
# the tunnel is up once the droplet's Caddy reaches the backend through it
for _ in $(seq 1 30); do
  if curl -s -m 5 -o /dev/null -w '%{http_code}' "$SITE/health" 2>/dev/null | grep -q 200; then UP=1; break; fi
  sleep 2
done
echo
echo "Owner link (settings):              $SITE/?token=$AVATAR_ACCESS_TOKEN"
echo "Customer link (conversation only):  $SITE/?token=$AVATAR_CUSTOMER_TOKEN"
if [ "${UP:-0}" = 1 ]; then echo "$SITE is up."; else
  echo "$SITE does not answer yet: see logs/agent_tunnel.log (droplet set up? DNS record agent -> 162.243.127.176?)"
fi
echo "stop everything with scripts/run.sh --stop"
