import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { appPath, customerToken, withOwnerToken } from './helpers.js'

// Several customers at once (backend profile hpn, conversations: 3): N customers open the customer link in
// their own browsers at the same moment and ask a question by voice (fake microphone: mic_hpn_wedding.wav).
// Each must get the answer, its voice and its lip-synced video; reported per customer: time to the first
// audio, the avatar's first frame, lip-synced frames received and late frames.
//   LOAD_CUSTOMERS=3 npx playwright test --project hpn load_hpn
//   LOAD_CUSTOMERS=3 LOAD_DELAYS=0,10000,15000 npx playwright test --project hpn load_hpn   (B 10 s, C 15 s later)
const N = Number(process.env.LOAD_CUSTOMERS ?? 3)
// when each customer starts, in ms from the first (default: all at once); e.g. LOAD_DELAYS=0,10000,15000
const DELAYS = (process.env.LOAD_DELAYS ?? '').split(',').filter(Boolean).map(Number)

test(`${N} customers talk to the avatar at the same time`, async ({ browser, request }) => {
  test.setTimeout(240_000)
  const token = customerToken()
  test.skip(!token, 'needs the customer link (scripts/tunnel.sh or AVATAR_CUSTOMER_TOKEN)')
  const runs = await Promise.all(Array.from({ length: N }, async (_, i) => {
    if (DELAYS[i]) await new Promise((r) => setTimeout(r, DELAYS[i]))
    const context = await browser.newContext()          // its own browser storage: its own visitor id
    const page = await context.newPage()
    const json = []
    const frames = { speech: 0, idle: 0 }
    page.on('websocket', (ws) => ws.on('framereceived', ({ payload }) => {
      if (typeof payload === 'string') {
        try { json.push({ ...JSON.parse(payload), _t: Date.now() }) } catch { /* ignore */ }
      } else {
        const kind = payload[0]
        if (kind === 2 || kind === 4) frames.speech += 1
        else if (kind === 3 || kind === 5) frames.idle += 1
      }
    }))
    await page.goto(appPath(token))
    await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
    await expect(page.getByTestId('voice-start')).toBeEnabled({ timeout: 30_000 })
    const t0 = Date.now()
    await page.getByTestId('voice-start').click()
    const waitFor = async (pred, ms) => {
      const end = Date.now() + ms
      while (Date.now() < end) {
        const hit = json.find(pred)
        if (hit) return hit
        await page.waitForTimeout(50)
      }
      return null
    }
    const said = await waitFor((m) => m.type === 'user_message' && m.source === 'voice', 60_000)
    const helper = said && await waitFor((m) => m.type === 'helper_reply' && m.turn_id === said.turn_id, 20_000)
    const done = said && await waitFor((m) => m.type === 'assistant_done' && m.turn_id === said.turn_id, 120_000)
    const metrics = said && await waitFor((m) => m.type === 'metrics' && m.turn_id === said.turn_id, 15_000)
    const waited = json.filter((m) => m.type === 'waiting').length > 0
    await context.close()
    return { customer: i + 1, waited, heard: said?.text, intent: helper?.reply?.intent, done: !!done,
      interrupted: done?.interrupted, speech_frames: frames.speech, idle_frames: frames.idle,
      ms_from_speech_end: metrics?.ms_from_t0, avatar_first_ms: metrics?.avatar_first_ms,
      avatar_forward: metrics?.avatar_forward, voice_started_after_ms: t0 && said ? said._t - t0 : null }
  }))
  const health = await (await request.get(withOwnerToken('/health'))).json()
  const report = { customers: N, runs, avatar_streams: health.avatar_streams, conversations: health.conversations }
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync(`results/load_hpn_${N}.json`, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 1))
  for (const r of runs) {
    expect(r.intent, `customer ${r.customer}`).toBe('prices.wedding')
    expect(r.done, `customer ${r.customer}`).toBe(true)
    expect(r.speech_frames, `customer ${r.customer}: lip-synced frames`).toBeGreaterThan(50)
  }
})
