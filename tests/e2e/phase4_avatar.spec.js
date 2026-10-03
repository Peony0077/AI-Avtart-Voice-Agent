import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { captureServerMessages, openAndStartVoice, waitFor } from './helpers.js'

// Phase 4: realistic avatar. Backend must run with profile "avatar" (scripts\run.ps1 -Profile avatar).
// FAKE_MIC=mic_question.wav
test('avatar idles, lip-syncs the spoken answer in sync with audio, and returns to idle', async ({ page }) => {
  const messages = captureServerMessages(page)
  await openAndStartVoice(page)
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(!session.avatar_available, 'backend not started with the avatar profile')

  const canvas = page.getByTestId('avatar-canvas')
  await expect(canvas).toBeVisible()
  await page.waitForTimeout(1500)
  const idleBefore = await page.evaluate(() => window.__avatarStats.idle)
  expect(idleBefore).toBeGreaterThan(10)                               // idle frames stream (~fps)
  fs.mkdirSync('results', { recursive: true })
  await canvas.screenshot({ path: 'results/avatar_idle.png' })

  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice', 30_000)
  await waitFor(messages, (m) => m.type === 'state' && m.state === 'SPEAKING' && m.turn_id === q.turn_id, 30_000)
  await page.waitForTimeout(700)
  await canvas.screenshot({ path: 'results/avatar_speaking.png' })
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 60_000)
  expect(done.text.toLowerCase()).toContain('paris')

  const stats = await page.evaluate(() => {
    const s = window.__avatarStats
    const off = [...s.avOffsets].sort((a, b) => a - b)
    const arr = s.arrivals.filter((a) => a !== null).sort((a, b) => a - b)
    return { idle: s.idle, speech: s.speech, drawnSpeech: s.drawnSpeech, late: s.late,
      avOffsetP50: off[Math.floor(off.length / 2)], avOffsetP90: off[Math.floor(off.length * 0.9)],
      arrivalLatenessP50: arr[Math.floor(arr.length / 2)], arrivalLatenessMax: arr.at(-1),
      framesBeforeAudioStarted: s.arrivals.filter((a) => a === null).length, clock: s.clock }
  })
  const metrics = await waitFor(messages, (m) => m.type === 'metrics' && m.turn_id === q.turn_id, 10_000)
  console.log(JSON.stringify({ diag: stats }))
  expect(stats.speech).toBeGreaterThan(10)                                // lip-sync frames were received
  expect(stats.drawnSpeech).toBeGreaterThan(10)                           // and drawn by the audio clock
  expect(stats.avOffsetP50).toBeLessThan(80)                              // frame shown ≤80 ms after its audio
  expect(metrics.ms_from_t0.avatar_first_frame).toBeGreaterThan(0)
  await page.waitForTimeout(1000)
  const idleAfter = await page.evaluate(() => window.__avatarStats.idle)
  expect(idleAfter).toBeGreaterThan(idleBefore)                           // back to idle stream

  // Phase 6 head gestures: e.g. a small nod whenever the avatar starts listening, the LLM's gesture per reply
  let gestures = null
  if ((session.avatar?.gestures ?? []).length) {
    const health = await (await page.request.get('/health')).json()
    gestures = health.avatar.stats.gestures ?? {}
    expect(Object.values(gestures).reduce((a, b) => a + b, 0)).toBeGreaterThan(0)
  }

  const summary = { test: 'phase4_avatar', reply: done.text, ...stats, first_audio_ms: metrics.ms_from_t0.client_first_audio,
    avatar_first_frame_ms: metrics.ms_from_t0.avatar_first_frame, avatar_render_ms: metrics.avatar_first_ms, gestures }
  console.log(JSON.stringify(summary))
  fs.writeFileSync('results/phase4_avatar.json', JSON.stringify(summary, null, 2))
})
