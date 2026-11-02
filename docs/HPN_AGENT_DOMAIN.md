# Connect agent.h-townparty.com

The UI is already available through the existing temporary tunnel. The hostname
`agent.h-townparty.com` is a proposed permanent address; it is not connected yet.
No DNS or Cloudflare account settings have been changed.

Observed on November 3, 2026:

- `h-townparty.com` nameservers: `ns17.domaincontrol.com`, `ns18.domaincontrol.com`
  (GoDaddy DNS).
- `agent.h-townparty.com`: no CNAME or IPv4 record resolved.
- The AI Agent backend listens on `http://127.0.0.1:8000` on the GPU computer.

## 1. Put the domain's DNS on Cloudflare

Cloudflare's Free/Pro setup uses Cloudflare authoritative nameservers. A GoDaddy
CNAME pointing at the existing random `trycloudflare.com` hostname is not a
permanent custom-domain tunnel setup.

1. Sign in to Cloudflare, add/onboard the domain **h-townparty.com**, and choose
   the Free plan. The domain registration can remain at GoDaddy.
2. Export the current DNS zone from GoDaddy and compare it with Cloudflare's
   imported records. Preserve the root website record, `www`, other subdomains,
   MX, SPF, DKIM, DMARC, verification TXT records, and any other records in use.
   The automatic scan can miss records. Keep existing website/mail host records
   DNS-only during the migration unless those services are already configured
   for Cloudflare's proxy.
3. If DNSSEC is enabled, follow Cloudflare's DNSSEC migration instructions before
   replacing nameservers; an old registrar DS record can invalidate DNS answers.
4. Cloudflare displays two assigned nameservers. In GoDaddy's domain settings,
   open **Nameservers → Change → I'll use my own nameservers** and enter those
   exact two Cloudflare nameservers. Do not guess them or add them as ordinary
   DNS records.
5. Wait for Cloudflare to show the domain as **Active**. Confirm the existing HPN
   website and email still work. Re-enable DNSSEC using Cloudflare's instructions
   after activation if it was enabled previously.

References:
[Cloudflare full DNS setup](https://developers.cloudflare.com/dns/zone-setups/full-setup/setup/)
and [GoDaddy nameserver settings](https://www.godaddy.com/en/help/change-my-domain-nameservers-664).

## 2. Create a permanent tunnel

1. In the Cloudflare dashboard, go to **Networking → Tunnels → Create Tunnel**.
2. Name it **houston-party-network-agent**.
3. Select the GPU computer's operating system (Linux, x86-64 for this project).
   Run the dashboard's **Install and Run** command on that computer. The command
   contains the new tunnel's private connector token; enter it on that computer,
   not in a chat message or source file.
4. Keep the backend running on port 8000 and wait for the new tunnel to show
   **Healthy**. The temporary and permanent connectors can coexist while testing.

## 3. Publish the AI Agent route

Open the permanent tunnel, then **Routes → Add route → Published application**:

| Setting | Value |
| --- | --- |
| Subdomain | `agent` |
| Domain | `h-townparty.com` |
| Path | Leave blank to route all paths, including `/ws` and static assets |
| Service type | `HTTP` |
| Service address | `127.0.0.1:8000` |
| Combined service URL, if shown as one field | `http://127.0.0.1:8000` |

The connector must run on the same computer as the backend for this loopback
address to work. HTTPS is provided publicly by Cloudflare; the local service is
HTTP. Save the route.

The published route normally creates this DNS record automatically. Verify it
under **h-townparty.com → DNS → Records**; add it manually only if missing:

| DNS field | Value |
| --- | --- |
| Type | `CNAME` |
| Name | `agent` |
| Target | `<THE-NEW-TUNNEL-UUID>.cfargotunnel.com` |
| Proxy status | Proxied / orange cloud |
| TTL | Auto |

Use the actual UUID of the **new permanent tunnel**. The tunnel name and its
connector token are not the DNS target. Do not point this record at the temporary
`operational-providence-pleasant-frozen.trycloudflare.com` address.

Reference: [Cloudflare tunnel and published application setup](https://developers.cloudflare.com/tunnel/get-started/).

## 4. Test

Once the tunnel is Healthy and DNS/TLS are ready, open:

```text
https://agent.h-townparty.com/houston-party-network/?token=<existing-customer-token>
```

The backend's existing owner/customer access tokens remain required. The new
hostname has separate browser storage, so the first visit must include the
appropriate token; use the customer token for customer testing.

Check that the AI Agent loads, `/ws` connects, and microphone permission works.
The GPU computer, backend, and permanent connector must stay running. The random
temporary tunnel can be retired after the new hostname has been verified.
