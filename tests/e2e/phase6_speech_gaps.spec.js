import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { appPath, captureServerMessages, waitFor } from './helpers.js'

// Speech gaps (found in user testing): every sentence after the first waited up to 1.5 s for its first
// avatar frame, and Kokoro wrapped every clause in ~0.3 s + ~0.5 s of silence (~0.8 s between sentences).
// Measures what the listener hears: silences inside the received audio, and time the player ran dry.
test('a multi-sentence reply plays without long gaps', async ({ page }) => {
  const messages = captureServerMessages(page)
  const audio = []                                       // [turnId, seq, Int16Array]
  page.on('websocket', (ws) => ws.on('framereceived', ({ payload }) => {
    if (typeof payload === 'string' || payload.length <= 16 || payload[0] !== 1) return
    const turnId = payload.readUInt32LE(4)
    const pcm = new Int16Array(payload.buffer.slice(payload.byteOffset + 16, payload.byteOffset + payload.length))
    audio.push([turnId, payload.readUInt32LE(8), pcm])
  }))
  const path = appPath()
  await page.goto(`${path}${path.includes('?') ? '&' : '?'}pcm`)   // PCM: the samples are analysed below
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(session.voice_available === false, 'voice models not loaded')

  const text = 'Tell me four short facts about octopuses, one sentence each.'
  await page.getByTestId('input').fill(text)
  await page.getByTestId('send').click()
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === text, 10_000)
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 90_000)
  const metrics = await waitFor(messages, (m) => m.type === 'metrics' && m.turn_id === q.turn_id, 10_000)

  // silences inside the reply's audio (between the first and the last voiced sample)
  const parts = audio.filter(([t]) => t === q.turn_id).sort((a, b) => a[1] - b[1]).map(([, , p]) => p)
  const pcm = Int16Array.from(parts.flatMap((p) => Array.from(p)))
  const win = 240                                         // 10 ms at 24 kHz
  const voiced = []
  for (let i = 0; i + win <= pcm.length; i += win) {
    let peak = 0
    for (let k = i; k < i + win; k++) peak = Math.max(peak, Math.abs(pcm[k]))
    voiced.push(peak > 330)                               // ~ -40 dBFS
  }
  const first = voiced.indexOf(true)
  const last = voiced.lastIndexOf(true)
  let longest = 0
  let run = 0
  for (let i = first; i <= last; i++) {
    run = voiced[i] ? 0 : run + 10
    longest = Math.max(longest, run)
  }
  const leadMs = first * 10
  const gaps = await page.evaluate((t) => window.__voiceStats?.gaps?.[t] ?? null, q.turn_id)
  const segments = messages.filter((m) => m.type === 'tts_segment' && m.turn_id === q.turn_id).length
  const summary = { test: 'speech_gaps', reply: done.text, segments, audio_ms: Math.round(pcm.length / 24),
    leading_silence_ms: leadMs, longest_pause_ms: longest, player_ran_dry_ms: gaps, server_gap_ms: metrics.playback_gap_ms,
    avatar: session.avatar_available }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/speech_gaps.json', JSON.stringify(summary, null, 2))
  expect(segments).toBeGreaterThanOrEqual(3)              // really several sentences
  expect(leadMs).toBeLessThan(100)                        // the first word starts right away
  expect(longest).toBeLessThan(500)                       // natural pauses only (was ~800 ms)
  expect(gaps).toBeLessThan(300)                          // the player never waits for audio (was up to 1.5 s)
})
