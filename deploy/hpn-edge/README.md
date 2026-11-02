# agent.h-townparty.com

HPN's AI avatar on its own subdomain. HPN's droplet (162.243.127.176, no GPU) only passes the traffic through;
the avatar runs on the owner's PC with the GPU.

```
visitor --https--> droplet: Caddy (edge-caddy-1), 30-agent.caddy --> 172.31.0.1:18080
                   ^ the PC's ssh reverse tunnel (user agent-tunnel, one port only) --> PC 127.0.0.1:8000 (backend)
```

## Once

1. DNS at GoDaddy (whoever manages h-townparty.com's records): **A** record, name `agent`, value
   `162.243.127.176`, TTL 1 hour. The website, bingo and email records are not touched.
2. The droplet (root over SSH, from this PC or Git Bash on Windows with `HPN_SSH=htown-vps`):
   ```
   deploy/hpn-edge/install.sh --check     # shows what it would do; changes nothing
   deploy/hpn-edge/install.sh --apply
   ```
   It adds the user `agent-tunnel` (key only, can open one port, no shell), an sshd `Match User agent-tunnel`
   block (checked: every other user's sshd settings stay identical), a ufw rule (from the Caddy network only)
   and `/opt/edge/conf/sites-enabled/30-agent.caddy` - the last only once the DNS record points at the droplet.
   Caddy's whole config is validated before a graceful reload, and h-townparty.com and bingo must answer the
   same before and after, or the site file is taken out again. HPN's own files are not changed.
   Run `--apply` again after adding DNS if it said WAIT.

## Every time

```
scripts/agent.sh          # backend (profile hpn) + the tunnel; prints the owner and customer links
scripts/run.sh --stop     # stop both
```

The links stay the same (tokens in `data/agent_tokens.env`; delete it for new links). While the PC or the tunnel
is off, the subdomain shows a short "The AI avatar is resting right now" page with HPN's phone and email.
Tunnel log: `logs/agent_tunnel.log`.

## Remove

`deploy/hpn-edge/install.sh --off` (site file, sshd block, ufw rule; the user is locked and its key removed),
then delete the DNS record.
