#!/usr/bin/env bash
# Start / stop the backend on Linux (it starts llama-server and, if enabled, the avatar worker itself).
# The Linux counterpart of scripts/run.ps1.
#
#   scripts/run.sh                     # profile from .env (AVATAR_PROFILE), else "default"
#   scripts/run.sh --profile avatar    # config/profiles/avatar.yaml
#   scripts/run.sh --build             # rebuild the frontend first
#   scripts/run.sh --host 0.0.0.0      # LAN: owner and customer tokens are generated (never open without one)
#   scripts/run.sh --stop              # stop everything (backend, its helpers, our quick tunnel)
#   scripts/run.sh --keep-tunnel       # restart only the backend: the tunnel and its links (tokens) stay, open
#                                      # pages reconnect by themselves
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROFILE="" PORT=8000 HOST=127.0.0.1 BUILD=0 STOP=0 FOREGROUND=0 KEEP_TUNNEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --build) BUILD=1; shift ;;
    --stop) STOP=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --keep-tunnel) KEEP_TUNNEL=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
mkdir -p logs
PIDFILE=logs/backend.pid

stop_all() {
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE")"
    # the backend runs in its own process group (setsid): stop the whole group
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  if [ "$KEEP_TUNNEL" = 0 ] && [ -f logs/tunnel.pid ]; then
    kill "$(cat logs/tunnel.pid)" 2>/dev/null || true
    rm -f logs/tunnel.pid
  fi
  # helpers of ours left behind by a killed backend (matched by this repo's paths only)
  pkill -f "$ROOT/tools/llama.cpp/" 2>/dev/null || true
  pkill -f "$ROOT/avatar/musetalk/worker.py" 2>/dev/null || true
  pkill -f "$ROOT/voice/xtts/worker.py" 2>/dev/null || true
}

if [ "$STOP" = 1 ]; then stop_all; echo stopped; exit 0; fi
stop_all

# .env -> environment, so that the workers and helper scripts see the same settings as the backend
if [ -f .env ]; then set -a; . ./.env; set +a; fi
[ -n "$PROFILE" ] && export AVATAR_PROFILE="$PROFILE"
export AVATAR_MODEL_HOME="${AVATAR_MODEL_HOME:-$HOME/AI-models}"
export HF_HOME="${HF_HOME:-$AVATAR_MODEL_HOME/hf}"
export HF_HUB_OFFLINE=1 CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1

if [ "$BUILD" = 1 ] || [ ! -f frontend/dist/index.html ]; then
  (cd frontend && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && npm run build)
fi

if [ "$KEEP_TUNNEL" = 1 ]; then
  # the running tunnel's links stay valid: the same tokens (from the links tunnel.sh saved)
  token_of() { sed -n 's/.*[?&]token=\([^&]*\).*/\1/p' "$1" 2>/dev/null; }
  export AVATAR_ACCESS_TOKEN="${AVATAR_ACCESS_TOKEN:-$(token_of logs/tunnel_url.txt)}"
  export AVATAR_CUSTOMER_TOKEN="${AVATAR_CUSTOMER_TOKEN:-$(token_of logs/customer_url.txt)}"
  [ -n "$AVATAR_ACCESS_TOKEN" ] || { echo "--keep-tunnel: no tunnel links in logs/ (start with scripts/tunnel.sh)" >&2; exit 1; }
fi
if [ "$HOST" != 127.0.0.1 ] && [ "$HOST" != localhost ]; then
  # reachable from other machines: never without an access token (without one, everyone is the owner)
  export AVATAR_ACCESS_TOKEN="${AVATAR_ACCESS_TOKEN:-$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(18))')}"
  export AVATAR_CUSTOMER_TOKEN="${AVATAR_CUSTOMER_TOKEN:-$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(18))')}"
fi

UV=(.venv/bin/python -m uvicorn app.main:app --app-dir backend --host "$HOST" --port "$PORT" --ws-max-size 16777216
    --no-access-log)   # the access log would write every ?token= into logs/uvicorn.log
if [ "$FOREGROUND" = 1 ]; then exec "${UV[@]}"; fi

setsid "${UV[@]}" >logs/uvicorn.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "backend pid $(cat "$PIDFILE") (profile ${AVATAR_PROFILE:-from .env/default}); waiting for health..."
for _ in $(seq 1 300); do
  sleep 1
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then tail -40 logs/uvicorn.log; echo "backend exited during startup" >&2; exit 1; fi
  if HEALTH="$(curl -sf "http://127.0.0.1:$PORT/health")"; then break; fi
done
[ -n "${HEALTH:-}" ] || { echo "backend did not become healthy in 300 s (see logs/uvicorn.log)" >&2; exit 1; }
echo "$HEALTH"
if [ "$KEEP_TUNNEL" = 1 ]; then
  echo "Owner link (unchanged):     $(cat logs/tunnel_url.txt)"
  echo "Customer link (unchanged):  $(cat logs/customer_url.txt)"
elif [ -n "${AVATAR_ACCESS_TOKEN:-}" ]; then
  # 0.0.0.0 is not an address another device can open: list this PC's own addresses
  if [ "$HOST" = 0.0.0.0 ]; then ADDRS="$(hostname -I 2>/dev/null)"; else ADDRS="$HOST"; fi
  for a in ${ADDRS:-127.0.0.1}; do
    case "$a" in *:*) continue ;; esac          # (IPv6: skipped)
    echo "owner:    http://$a:$PORT/?token=$AVATAR_ACCESS_TOKEN"
    echo "customer: http://$a:$PORT/?token=$AVATAR_CUSTOMER_TOKEN"
  done
  echo "(plain http: no microphone and JPEG video on other devices; use scripts/tunnel.sh for https)"
else
  echo "open http://127.0.0.1:$PORT/"
fi
