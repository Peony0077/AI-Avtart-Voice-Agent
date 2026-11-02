#!/usr/bin/env bash
# Set up agent.h-townparty.com on HPN's droplet from this PC (or from Git Bash on Windows): copies
# deploy/hpn-edge/ and the tunnel's public key to the droplet and runs setup-server.sh there as root.
#
#   deploy/hpn-edge/install.sh --check            # what would change (nothing is changed)
#   deploy/hpn-edge/install.sh --apply
#   deploy/hpn-edge/install.sh --off              # remove it again
#   HPN_SSH=htown-vps deploy/hpn-edge/install.sh --apply     # another ssh name for root@droplet
#
# The tunnel key (~/.ssh/hpn_agent_tunnel) is made on the PC that runs the avatar (scripts/agent.sh makes
# it too); when this runs elsewhere, pass its public key: TUNNEL_PUB=path/to/hpn_agent_tunnel.pub
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---check}"
SSH_TARGET="${HPN_SSH:-root@162.243.127.176}"
PUB="${TUNNEL_PUB:-$HOME/.ssh/hpn_agent_tunnel.pub}"
[ -f "$PUB" ] || ssh-keygen -q -t ed25519 -N "" -C "agent.h-townparty.com tunnel" -f "${PUB%.pub}"
REMOTE=/root/agent-h-townparty-setup
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$SSH_TARGET" "rm -rf $REMOTE && mkdir -p $REMOTE"
scp -q -o BatchMode=yes "$HERE/setup-server.sh" "$HERE/30-agent.caddy" "$SSH_TARGET:$REMOTE/"
scp -q -o BatchMode=yes "$PUB" "$SSH_TARGET:$REMOTE/agent-tunnel.pub"
ssh -o BatchMode=yes "$SSH_TARGET" "bash $REMOTE/setup-server.sh $MODE"
echo "(the droplet's host key, as this PC knows it - compare with the one your Windows PC knows:)"
ssh-keygen -lF "${SSH_TARGET#*@}" 2>/dev/null | grep -v '^#' | head -2 || true
