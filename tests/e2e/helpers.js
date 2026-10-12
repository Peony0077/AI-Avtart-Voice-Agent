import { expect } from '@playwright/test'
import fs from 'node:fs'

const LOGS = new URL('../../logs/', import.meta.url)

// The token of a link that scripts/tunnel.ps1 / tunnel.sh saved (logs/tunnel_url.txt: owner,
// logs/customer_url.txt: customer) - but only while it belongs to the running backend. The files stay
// after the tunnel and the backend are stopped, and a backend started later (scripts/run.*) has other
// tokens or none: run.* writes logs/backend.pid at every start and deletes it on stop, and the tunnel
// writes its links after that, so a link older than backend.pid (or without one) is stale.
export function savedLinkToken(name) {
  const file = new URL(name, LOGS)
  const pid = new URL('backend.pid', LOGS)
  if (!fs.existsSync(file) || !fs.existsSync(pid)) return null
  if (fs.statSync(file).mtimeMs < fs.statSync(pid).mtimeMs) return null
  try {
    const link = new URL(fs.readFileSync(file, 'utf8').trim())
    const target = process.env.AVATAR_URL && new URL(process.env.AVATAR_URL)
    // a server elsewhere: this PC's link is not for it (unless AVATAR_URL is that very link's address)
    if (target && !/^(127\.0\.0\.1|localhost)$/.test(target.hostname) && target.origin !== link.origin) return null
    return link.searchParams.get('token')
  } catch {
    return null
  }
}

// The owner's access token: AVATAR_TOKEN or AVATAR_ACCESS_TOKEN (the name the server and the scripts use),
// else the owner link of a running tunnel. None: the server must run without a token (scripts/run.* on
// 127.0.0.1).
export const ownerToken = () => process.env.AVATAR_TOKEN || process.env.AVATAR_ACCESS_TOKEN || savedLinkToken('tunnel_url.txt')
// The customer link's token: AVATAR_CUSTOMER_TOKEN, else the customer link of a running tunnel.
export const customerToken = () => process.env.AVATAR_CUSTOMER_TOKEN || savedLinkToken('customer_url.txt')

// An HTTP endpoint with the owner's token: /health details, /health/gpu and /api/metrics need it when the
// server has one (without it /health says only {"status": "ok"})
export const withOwnerToken = (path) => {
  const t = ownerToken()
  return t ? `${path}${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(t)}` : path
}

// Page path incl. the access token when the server requires one (see ownerToken)
// NOSMOOTH=1: no in-between frames (compares the GPU load of the test browser, which shares this GPU)
export const appPath = (token = ownerToken()) => {
  const params = [token ? `token=${encodeURIComponent(token)}` : '', process.env.NOSMOOTH ? 'nosmooth' : '',
    process.env.NOGRAIN ? 'nograin' : ''].filter(Boolean)
  return params.length ? `/?${params.join('&')}` : '/'
}

export const manifest = JSON.parse(fs.readFileSync(new URL('./fixtures/manifest.json', import.meta.url)))

// Collect every JSON message the server sends (via Playwright's WebSocket inspection).
export function captureServerMessages(page) {
  const messages = []
  page.on('websocket', (ws) => {
    ws.on('framereceived', ({ payload }) => {
      if (typeof payload === 'string') {
        try {
          messages.push({ ...JSON.parse(payload), _t: Date.now() })
        } catch { /* ignore */ }
      }
    })
  })
  return messages
}

export async function openAndStartVoice(page) {
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  await expect(page.getByTestId('voice-start')).toBeEnabled({ timeout: 30_000 })
  await page.getByTestId('voice-start').click()
  await expect(page.getByTestId('mic')).toBeVisible({ timeout: 15_000 })
  return Date.now() // fake-mic file starts playing ~now
}

export async function waitFor(messages, predicate, timeoutMs = 60_000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const hit = messages.find(predicate)
    if (hit) return hit
    await new Promise((r) => setTimeout(r, 50))
  }
  throw new Error(`timed out waiting for server message (${timeoutMs} ms)`)
}

export function turnMetrics(messages, turnId) {
  return messages.find((m) => m.type === 'metrics' && m.turn_id === turnId)
}
