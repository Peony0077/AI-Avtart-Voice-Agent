import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { appPath, captureServerMessages, waitFor } from './helpers.js'

// The page's playout delay at the start of replies (AvatarStage nextStartHold, useVoice): over a jittery
// network a reply's first audio waits, so its lip-synced frames are there in time. On this PC it is 0, so
// it is forced here with ?hold=300: the whole reply must still be heard, start later by the hold and stay
// lip-synced, and speech must take over from the idle video where the worker was told it would.
test('a held reply start plays all of the reply, later, in sync', async ({ page }) => {
  const messages = captureServerMessages(page)
  let audioMs = 0
  let firstAudioAt = null
  let startedAt = null
  let endedMs = null
  page.on('websocket', (ws) => {
    ws.on('framereceived', ({ payload }) => {
      if (typeof payload === 'string' || payload[0] !== 1) return
      firstAudioAt ??= Date.now()
      audioMs += (payload.length - 16) / 48                     // PCM16 at 24 kHz after a 16-byte header
    })
    ws.on('framesent', ({ payload }) => {
      if (typeof payload !== 'string' || !payload.includes('"playback"')) return
      const m = JSON.parse(payload)
      if (m.event === 'started') startedAt ??= Date.now()
      if (m.event === 'ended') endedMs = m.played_ms
    })
  })
  const path = appPath()
  await page.goto(`${path}${path.includes('?') ? '&' : '?'}hold=300`)
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(!session.avatar_available || session.voice_available === false, 'needs the avatar and the voice')

  const text = 'What is the capital of France? Answer in two sentences.'
  await page.getByTestId('input').fill(text)
  await page.getByTestId('send').click()
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === text, 10_000)
  await waitFor(messages, (m) => m.type === 'tts_done' && m.turn_id === q.turn_id, 60_000)
  await expect.poll(() => endedMs, { timeout: 30_000 }).not.toBeNull()
  await page.waitForTimeout(1500)                               // the closing tail, back to the idle video
  const st = await page.evaluate(() => ({ holds: window.__voiceStats.holds, late: window.__avatarStats.late,
    drawn: window.__avatarStats.drawnSpeech, startGaps: window.__avatarStats.startGaps }))
  const summary = { test: 'start_hold', held_ms: startedAt - firstAudioAt, audio_ms: Math.round(audioMs),
    played_ms: endedMs, ...st }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/start_hold.json', JSON.stringify(summary, null, 2))
  expect(st.holds).toEqual([300])                               // the reply's first audio was held ...
  expect(summary.held_ms).toBeGreaterThanOrEqual(280)           // ... that long
  expect(summary.held_ms).toBeLessThan(800)
  expect(Math.abs(endedMs - audioMs)).toBeLessThan(100)         // and all of it was heard
  expect(st.drawn).toBeGreaterThan(20)
  expect(st.late / st.drawn).toBeLessThan(0.05)                 // lips in sync
  expect(st.startGaps.length).toBeGreaterThan(0)
  expect(Math.abs(st.startGaps.at(-1))).toBeLessThanOrEqual(4)  // no jump where speech takes over
})
