import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { captureServerMessages, manifest, openAndStartVoice, waitFor } from './helpers.js'

// Phase 3: barge-in. Run with FAKE_MIC=mic_story_bargein.wav
// The fake mic asks for a long story, then says "Wait, stop please. What is two plus two?"
// while the assistant is still talking.
test('user speech interrupts the assistant; stale audio never plays; follow-up is answered', async ({ page }) => {
  test.skip(process.env.FAKE_MIC !== 'mic_story_bargein.wav', 'needs FAKE_MIC=mic_story_bargein.wav')
  const messages = captureServerMessages(page)
  const tStart = await openAndStartVoice(page)

  const first = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice', 30_000)
  expect(first.text.toLowerCase()).toContain('lighthouse')
  await waitFor(messages, (m) => m.type === 'state' && m.state === 'SPEAKING', 30_000)

  const intr = await waitFor(messages, (m) => m.type === 'interrupt' && m.turn_id === first.turn_id, 30_000)
  expect(intr.reason).toBe('barge_in')
  const onsetMs = manifest.mic_story_bargein.speech[1].start_s * 1000
  const detectMs = intr._t - tStart - onsetMs   // approx: mic file starts when the capture starts

  const doneIntr = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === first.turn_id)
  expect(doneIntr.interrupted).toBe(true)
  expect(doneIntr.text.length).toBeGreaterThan(0) // what the user actually heard is recorded

  const follow = await waitFor(messages, (m) => m.type === 'user_message' && m.turn_id > first.turn_id, 30_000)
  expect(follow.text.toLowerCase()).toMatch(/two plus two|2 plus 2|2 \+ 2/)
  const followDone = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === follow.turn_id, 60_000)
  expect(followDone.text.toLowerCase()).toMatch(/four|\b4\b/)

  const stats = await page.evaluate(() => window.__voiceStats)
  const flush = stats.interrupts.find((i) => i.turnId === first.turn_id)
  const flushMs = flush.flushedAt - flush.receivedAt
  expect(flushMs).toBeLessThan(50)
  // Audio packets of the interrupted turn that arrived after the interrupt were blocked, never played.
  const serverStale = messages.filter((m) => m.type === 'tts_segment' && m.turn_id === first.turn_id && m._t > intr._t).length
  expect(serverStale).toBe(0)

  const summary = { test: 'phase3_bargein', detect_ms_after_onset_approx: Math.round(detectMs), flush_ms: Math.round(flushMs * 10) / 10,
    heard: doneIntr.text, follow_up: follow.text, reply: followDone.text, blocked_stale_client: stats.blockedStale }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/phase3_bargein.json', JSON.stringify(summary, null, 2))
})

test('Stop button interrupts a spoken answer', async ({ page }) => {
  test.skip(process.env.FAKE_MIC !== 'mic_question.wav', 'needs FAKE_MIC=mic_question.wav')
  const messages = captureServerMessages(page)
  await openAndStartVoice(page)
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice', 30_000)
  await waitFor(messages, (m) => m.type === 'state' && m.state === 'SPEAKING' && m.turn_id === q.turn_id, 30_000)
  await page.getByTestId('stop').click()
  const intr = await waitFor(messages, (m) => m.type === 'interrupt' && m.turn_id === q.turn_id, 5_000)
  expect(intr.reason).toBe('manual')
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING')
})
