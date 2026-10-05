import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { appPath, captureServerMessages, waitFor } from './helpers.js'

// A light, steady stream (owner, 2026-10-05: "sometimes the avatar is stuck while she is talking ... make
// it light and smooth"): the voice as Opus (~56 instead of 384 kbit/s), video only while the page keeps up
// (it acknowledges what it received), the video bitrate following the connection, no video for a hidden
// page. Measured before on a slow link: 800 kbit/s - 8.9 s of gaps in one reply and 99 % of the lips late.

async function speak(page, messages, text) {
  await page.getByTestId('input').fill(text)
  await page.getByTestId('send').click()
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === text, 10_000)
  return waitFor(messages, (m) => m.type === 'metrics' && m.turn_id === q.turn_id, 90_000)
}

test('the voice comes as Opus, the page acknowledges, and the whole reply is heard', async ({ page }) => {
  const messages = captureServerMessages(page)
  let opusMs = 0
  let opusBytes = 0
  let acks = 0
  let hello = null
  let endedMs = null
  page.on('websocket', (ws) => {
    ws.on('framereceived', ({ payload }) => {
      if (typeof payload !== 'string' && payload[0] === 6) {
        opusMs += payload.readUInt32LE(16) / 24
        opusBytes += payload.length
      }
    })
    ws.on('framesent', ({ payload }) => {
      if (typeof payload !== 'string') return
      const m = JSON.parse(payload)
      if (m.type === 'ack') acks++
      if (m.type === 'hello') hello = m
      if (m.type === 'playback' && m.event === 'ended') endedMs = m.played_ms
    })
  })
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(session.voice_available === false, 'voice models not loaded')
  const met = await speak(page, messages, 'Tell me one short fact about the moon.')
  await expect.poll(() => endedMs, { timeout: 30_000 }).not.toBeNull()
  const summary = { test: 'light_voice', hello_audio: hello?.audio, opus_ms: Math.round(opusMs), played_ms: endedMs,
    opus_kbit_s: Math.round(opusBytes * 8 / opusMs), acks, gaps_ms: met.playback_gap_ms }
  console.log(JSON.stringify(summary))
  expect(hello.audio).toEqual(['opus', 'pcm'])
  expect(hello.ack).toBe(true)
  expect(opusMs).toBeGreaterThan(500)                          // the reply came as Opus ...
  expect(Math.abs(endedMs - opusMs)).toBeLessThan(100)         // ... all of it was played (exact length)
  expect(summary.opus_kbit_s).toBeLessThan(100)                 // PCM was 384 kbit/s
  expect(acks).toBeGreaterThan(5)
  expect(met.playback_gap_ms ?? 0).toBeLessThan(300)
})

test('on a slow connection (800 kbit/s) the voice starts soon and has no gaps', async ({ page, context }) => {
  const messages = captureServerMessages(page)
  const cdp = await context.newCDPSession(page)
  await cdp.send('Network.enable')
  await cdp.send('Network.emulateNetworkConditions', { offline: false, latency: 60, downloadThroughput: 800_000 / 8,
    uploadThroughput: 1_000_000 / 8 })
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 60_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 30_000)
  test.skip(!session.avatar_available || session.voice_available === false, 'needs the avatar and the voice')
  await page.waitForTimeout(3000)
  const results = []
  for (const text of ['Tell me two short facts about the sun.', 'Tell me two short facts about the ocean.']) {
    const before = await page.evaluate(() => ({ late: window.__avatarStats.late, drawn: window.__avatarStats.drawnSpeech }))
    const met = await speak(page, messages, text)
    await page.waitForTimeout(2500)
    const after = await page.evaluate(() => ({ late: window.__avatarStats.late, drawn: window.__avatarStats.drawnSpeech }))
    results.push({ heard_ms: Math.round(met.ms_from_t0.client_first_audio), gaps_ms: met.playback_gap_ms,
      lips_late: `${after.late - before.late}/${after.drawn - before.drawn}` })
  }
  console.log(JSON.stringify({ test: 'light_slow_link', results }))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/light_slow_link.json', JSON.stringify(results, null, 2))
  for (const r of results) {
    expect(r.gaps_ms ?? 0).toBeLessThan(300)                    // was 8.9 s
    expect(r.heard_ms).toBeLessThan(3000)
  }
  const [late, drawn] = results.at(-1).lips_late.split('/').map(Number)
  expect(late / drawn).toBeLessThan(0.15)                       // the video has adapted (was 99 % late)
})

test('a hidden page gets no video and has a picture again when shown', async ({ page }) => {
  const messages = captureServerMessages(page)
  let videoBytes = 0
  page.on('websocket', (ws) => ws.on('framereceived', ({ payload }) => {
    if (typeof payload !== 'string' && payload[0] >= 2 && payload[0] <= 5) videoBytes += payload.length
  }))
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(!session.avatar_available, 'backend not started with an avatar profile')
  await page.waitForTimeout(2000)
  const setHidden = (hidden) => page.evaluate((h) => {
    Object.defineProperty(document, 'hidden', { value: h, configurable: true })
    Object.defineProperty(document, 'visibilityState', { value: h ? 'hidden' : 'visible', configurable: true })
    document.dispatchEvent(new Event('visibilitychange'))
  }, hidden)
  await setHidden(true)
  await page.waitForTimeout(1000)                                // what was already on its way
  const hiddenStart = videoBytes
  await page.waitForTimeout(4000)
  const hiddenBytes = videoBytes - hiddenStart
  const idleBefore = await page.evaluate(() => window.__avatarStats.idle)
  await setHidden(false)
  await page.waitForTimeout(4000)
  const st = await page.evaluate(() => ({ idle: window.__avatarStats.idle, errors: window.__avatarStats.h264Errors ?? 0 }))
  console.log(JSON.stringify({ test: 'light_hidden', hidden_video_bytes: hiddenBytes, idle_frames_after: st.idle - idleBefore,
    decoder_errors: st.errors }))
  expect(hiddenBytes).toBeLessThan(20_000)                      // ~0.35 Mbit/s = 175 KB in 4 s before
  expect(st.idle - idleBefore).toBeGreaterThan(30)              // the idle video is back (from a keyframe)
  expect(st.errors).toBe(0)
})
