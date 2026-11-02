#!/usr/bin/env bash
# Keeps the reverse tunnel to HPN's droplet open (agent.h-townparty.com -> this PC's backend), forever:
# reconnects 5 s after it drops (a router restart, a new IP, the droplet rebooting). Started and stopped by
# scripts/agent.sh / scripts/run.sh --stop; deploy/hpn-edge/README.md explains the droplet side.
#   scripts/agent_tunnel_loop.sh <local port>
set -u
PORT="${1:-8000}"
TARGET="${AGENT_TUNNEL_TARGET:-agent-tunnel@162.243.127.176}"
KEY="${AGENT_TUNNEL_KEY:-$HOME/.ssh/hpn_agent_tunnel}"
while true; do
  echo "$(date '+%F %T') connecting $TARGET (172.31.0.1:18080 -> 127.0.0.1:$PORT)"
  ssh -N -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=15 \
      -R "172.31.0.1:18080:127.0.0.1:$PORT" "$TARGET"
  echo "$(date '+%F %T') tunnel closed (exit $?); again in 5 s"
  sleep 5
done
