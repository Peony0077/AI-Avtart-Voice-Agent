#!/usr/bin/env bash
# agent.h-townparty.com on HPN's droplet: run as root ON THE DROPLET (deploy/hpn-edge/install.sh copies this
# folder there and runs it). It adds, and changes nothing else:
#
#   1. user agent-tunnel: key-only, no shell, no password; its key may open ONE port, 172.31.0.1:18080 (the
#      edge_hpn bridge's gateway: reachable from the Caddy container, not from the internet). The owner's PC
#      holds the tunnel: ssh -N -R 172.31.0.1:18080:127.0.0.1:8000 agent-tunnel@<droplet> (scripts/agent.sh).
#   2. /etc/ssh/sshd_config.d/60-agent-tunnel.conf: a Match block for that user only. Checked: root's (and
#      every other user's) effective sshd settings must be identical before and after, or it is taken out.
#   3. ufw: that port, from the edge_hpn subnet only.
#   4. /opt/edge/conf/sites-enabled/30-agent.caddy, only once agent.h-townparty.com points at this droplet
#      (else Caddy would ask Let's Encrypt for a certificate it cannot get). The whole Caddyfile is validated
#      before a graceful reload; h-townparty.com and bingo must answer the same before and after, or the
#      file is removed and Caddy reloaded back.
#
#   bash setup-server.sh --check     # (default) what would be done; changes nothing
#   bash setup-server.sh --apply
#   bash setup-server.sh --off       # remove all four
#   bash setup-server.sh --apply hpn-agent.lyly.earth   # another name (default agent.h-townparty.com)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---check}"
NAME="${2:-agent.h-townparty.com}"
[[ "$NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$ ]] || { echo "STOP: not a domain name: $NAME" >&2; exit 1; }
USER_NAME=agent-tunnel
NET=edge_hpn
GW=172.31.0.1
SUBNET=172.31.0.0/24
PORT=18080
CADDY=edge-caddy-1
SITE_DIR=/opt/edge/conf/sites-enabled
SITE=$SITE_DIR/30-agent.caddy
SSHD_DROPIN=/etc/ssh/sshd_config.d/60-agent-tunnel.conf
KEY_FILE="$HERE/agent-tunnel.pub"

say()  { printf '%s\n' "$*"; }
ok()   { printf '  ok    %s\n' "$*"; }
todo() { printf '  todo  %s\n' "$*"; }
die()  { printf 'STOP: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root"
case "$MODE" in --check|--apply|--off) ;; *) die "unknown option $MODE (--check, --apply or --off)";; esac

code() {  # code <host> <path>: the HTTPS status this droplet's Caddy gives for that name
  curl -sk -o /dev/null -m 15 -w '%{http_code}' --resolve "$1:443:127.0.0.1" "https://$1$2" || echo 000
}
sites_state() { printf 'apex=%s bingo=%s' "$(code h-townparty.com /)" "$(code bingo.h-townparty.com /)"; }
caddy_validate() { timeout 60 docker exec "$CADDY" caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1; }
caddy_reload()   { timeout 60 docker exec "$CADDY" caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile; }
# every user's effective sshd settings except agent-tunnel's (root, and an ordinary user name)
sshd_others() { for u in root nobody; do sshd -T -C "user=$u,host=example.com,addr=203.0.113.9" 2>/dev/null | sort; done; }
reload_sshd() { systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true; }
bridge_if() { ip -o -4 addr show | awk -v gw="$GW" '$4 ~ "^"gw"/" {print $2; exit}'; }
dns_ok() {
  local mine theirs
  mine=$(curl -4 -s -m 10 https://api.ipify.org || true)
  theirs=$(getent ahostsv4 "$NAME" | awk 'NR==1{print $1}')
  [ -n "$mine" ] && [ "$mine" = "$theirs" ]
}

# --- facts -----------------------------------------------------------------------------------------------
say "== $NAME on $(hostname) ($MODE)"
docker inspect -f '{{.State.Running}}' "$CADDY" 2>/dev/null | grep -qx true || die "Caddy container $CADDY is not running"
[ -d "$SITE_DIR" ] || die "$SITE_DIR missing"
gw=$(docker network inspect "$NET" -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)
[ "$gw" = "$GW" ] || die "network $NET's gateway is '$gw', expected $GW"
BR=$(bridge_if); [ -n "$BR" ] || die "no host interface has $GW"
ok "Caddy $CADDY running; $NET gateway $GW on $BR"
BEFORE=$(sites_state); ok "now: $BEFORE"
if sshd -T 2>/dev/null | grep -qiE '^(allowusers|allowgroups) '; then
  die "sshd has AllowUsers/AllowGroups: add $USER_NAME there by hand first (not changed by this script)"
fi

if [ "$MODE" = --off ]; then
  if [ -f "$SITE" ]; then rm -f "$SITE"; caddy_validate && caddy_reload; say "  removed $SITE (Caddy reloaded)"; fi
  if [ -f "$SSHD_DROPIN" ]; then rm -f "$SSHD_DROPIN"; sshd -t && reload_sshd; say "  removed $SSHD_DROPIN"; fi
  if command -v ufw >/dev/null && ufw status | grep -q "$PORT"; then
    ufw delete allow in on "$BR" from "$SUBNET" to "$GW" port "$PORT" proto tcp >/dev/null || true; say "  ufw rule removed"
  fi
  if id "$USER_NAME" >/dev/null 2>&1; then pkill -u "$USER_NAME" || true; usermod -L "$USER_NAME"; rm -f "/home/$USER_NAME/.ssh/authorized_keys"; say "  $USER_NAME locked, key removed"; fi
  say "after: $(sites_state)"
  exit 0
fi

[ -s "$KEY_FILE" ] || die "$KEY_FILE missing (the PC's tunnel key)"
grep -qE '^ssh-ed25519 [A-Za-z0-9+/=]+' "$KEY_FILE" || die "$KEY_FILE is not an ed25519 public key"
KEY=$(head -1 "$KEY_FILE" | awk '{print $1" "$2}')
AUTH="restrict,port-forwarding,permitlisten=\"$GW:$PORT\",command=\"/bin/false\" $KEY agent.h-townparty.com tunnel"
DROPIN="# agent.h-townparty.com: the owner's PC holds a reverse tunnel for HPN's AI avatar (deploy/hpn-edge in
# -AI-Avtart-Voice-Agent). Only user $USER_NAME, only remote forwarding to $GW:$PORT. Remove with setup-server.sh --off.
Match User $USER_NAME
	AuthenticationMethods publickey
	PasswordAuthentication no
	KbdInteractiveAuthentication no
	AllowTcpForwarding remote
	GatewayPorts clientspecified
	PermitListen $GW:$PORT
	PermitOpen none
	AllowAgentForwarding no
	AllowStreamLocalForwarding no
	X11Forwarding no
	PermitTTY no
	PermitTunnel no
"

if [ "$MODE" = --check ]; then
  id "$USER_NAME" >/dev/null 2>&1 && ok "user $USER_NAME exists" || todo "create user $USER_NAME (no shell, no password)"
  [ -f "$SSHD_DROPIN" ] && ok "$SSHD_DROPIN exists" || todo "write $SSHD_DROPIN (Match User $USER_NAME only)"
  command -v ufw >/dev/null && ufw status | grep -q "Status: active" && todo "ufw: allow $SUBNET -> $GW:$PORT on $BR" || ok "ufw not active"
  if dns_ok; then todo "install $SITE, validate, reload Caddy"; else todo "$NAME does not point at this droplet yet: the Caddy site waits (add DNS: A agent -> this droplet's IP)"; fi
  say "nothing changed (--check)"
  exit 0
fi

# --- 1. the tunnel user ------------------------------------------------------------------------------------
if ! id "$USER_NAME" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$USER_NAME"
fi
passwd -l "$USER_NAME" >/dev/null 2>&1 || true
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/.ssh"
printf '%s\n' "$AUTH" > "/home/$USER_NAME/.ssh/authorized_keys"
chown "$USER_NAME:$USER_NAME" "/home/$USER_NAME/.ssh/authorized_keys"; chmod 600 "/home/$USER_NAME/.ssh/authorized_keys"
ok "user $USER_NAME: key installed (one port only)"

# --- 2. sshd: this user only -------------------------------------------------------------------------------
others_before=$(sshd_others)
printf '%s' "$DROPIN" > "$SSHD_DROPIN.new"
mv "$SSHD_DROPIN.new" "$SSHD_DROPIN"; chmod 644 "$SSHD_DROPIN"
if ! sshd -t; then rm -f "$SSHD_DROPIN"; die "sshd rejected the drop-in: removed, nothing reloaded"; fi
if [ "$(sshd_others)" != "$others_before" ]; then
  rm -f "$SSHD_DROPIN"; die "the drop-in changed other users' sshd settings: removed, nothing reloaded"
fi
sshd -T -C "user=$USER_NAME,host=example.com,addr=203.0.113.9" | grep -qi "^gatewayports clientspecified" \
  || { rm -f "$SSHD_DROPIN"; die "the Match block did not apply to $USER_NAME: removed"; }
reload_sshd
ok "sshd: Match User $USER_NAME (root and everyone else unchanged)"

# --- 3. firewall --------------------------------------------------------------------------------------------
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow in on "$BR" from "$SUBNET" to "$GW" port "$PORT" proto tcp comment "$NAME tunnel" >/dev/null
  ok "ufw: $SUBNET -> $GW:$PORT on $BR"
fi

# --- 4. Caddy -------------------------------------------------------------------------------------------------
if ! dns_ok; then
  say "WAIT: $NAME does not point at this droplet yet. Add the DNS record (A agent -> $(curl -4 -s -m 10 https://api.ipify.org))"
  say "      and run --apply again: the tunnel part is done, the Caddy site is not installed yet."
  exit 0
fi
[ -f "$SITE" ] && cp -p "$SITE" "$SITE.bak.$(date +%Y%m%d%H%M%S)"
sed "s/agent\.h-townparty\.com/$NAME/g" "$HERE/30-agent.caddy" > "$SITE.new" && chmod 644 "$SITE.new" && mv "$SITE.new" "$SITE"
if ! caddy_validate; then
  rm -f "$SITE"; die "Caddy rejected 30-agent.caddy: removed, Caddy not reloaded (still running the old config)"
fi
caddy_reload
sleep 3
AFTER=$(sites_state)
if [ "$AFTER" != "$BEFORE" ]; then
  rm -f "$SITE"; caddy_validate && caddy_reload
  die "after the reload: $AFTER (before: $BEFORE): 30-agent.caddy removed and Caddy reloaded back"
fi
ok "Caddy reloaded; still $AFTER"
# the certificate: Caddy asks Let's Encrypt in the background (seconds to a minute)
for _ in $(seq 1 30); do
  c=$(code "$NAME" /); case "$c" in 200|503) break;; esac; sleep 2
done
ok "$NAME answers $c (503 = the PC's tunnel is not connected yet: the 'resting' page)"
say "done"
