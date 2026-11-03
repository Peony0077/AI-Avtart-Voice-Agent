#!/usr/bin/env bash
# What the "hpn-avatar" service starts when this Linux (WSL) starts (deploy/systemd/hpn-avatar.service):
#   * agent.h-townparty.com set up (data/agent_domain_ready exists): scripts/agent.sh - the permanent domain,
#     fixed links (data/agent_tokens.env), the tunnel to HPN's droplet;
#   * until then: a Cloudflare quick tunnel (scripts/tunnel.sh: new links in logs/tunnel_url.txt and
#     logs/customer_url.txt every start).
# Waits for the internet first (a cold Windows start brings the network up after WSL).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
for _ in $(seq 1 60); do
  curl -s -m 5 -o /dev/null https://h-townparty.com/robots.txt && break
  sleep 3
done
if [ -f data/agent_domain_ready ]; then
  exec scripts/agent.sh --profile "${AVATAR_PROFILE_AUTOSTART:-hpn}"
fi
exec scripts/tunnel.sh --profile "${AVATAR_PROFILE_AUTOSTART:-hpn}" --detach
