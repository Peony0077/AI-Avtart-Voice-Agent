import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { captureServerMessages, manifest, openAndStartVoice, turnMetrics, waitFor } from './helpers.js'

// Phase 2: spoken question through the fake microphone → VAD → STT → LLM → TTS → browser playback.
// Run with FAKE_MIC=mic_question.wav (default).

test('spoken question is transcribed, answered and played back', async ({ page }) => {
  const messages = captureServerMessages(page)
  await openAndStartVoice(page)

  const expected = manifest.mic_question.speech[0].text
  const userMsg = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice', 30_000)
  expect(userMsg.text.toLowerCase()).toContain('capital of france')

  await waitFor(messages, (m) => m.type === 'tts_segment' && m.turn_id === userMsg.turn_id, 20_000)
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === userMsg.turn_id, 60_000)
  expect(done.interrupted).toBe(false)
  expect(done.text.toLowerCase()).toContain('paris')

  const metrics = await waitFor(messages, (m) => m.type === 'metrics' && m.turn_id === userMsg.turn_id, 10_000)
  const ms = metrics.ms_from_t0
  // Every stage must be present and in order.
  for (const key of ['vad_endpoint', 'stt_done', 'llm_first_token', 'llm_first_text', 'first_clause', 'tts_first_audio', 'audio_sent', 'client_first_audio', 'client_playback_ended']) {
    expect(ms[key], key).toBeGreaterThan(0)
  }
  expect(ms.stt_done).toBeLessThan(ms.llm_first_token)
  expect(ms.tts_first_audio).toBeLessThanOrEqual(ms.client_first_audio)

  // State sequence seen by the browser.
  const states = messages.filter((m) => m.type === 'state').map((m) => m.state)
  expect(states).toEqual(expect.arrayContaining(['END_OF_TURN', 'THINKING', 'SPEAKING', 'LISTENING']))
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING')

  const summary = { test: 'phase2_voice', expected, transcript: userMsg.text, reply: done.text, ms_from_speech_end: ms, speech_ms: metrics.speech_ms }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/phase2_voice.json', JSON.stringify({ summary, messages: messages.filter((m) => m.type !== 'llm_delta') }, null, 2))
})
