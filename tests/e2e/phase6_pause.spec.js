import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { captureServerMessages, openAndStartVoice, waitFor } from './helpers.js'

// A pause in the middle of a spoken question must not split it (found in user testing: one question
// became three turns, each answered by its last fragment). Two fixtures, run separately:
//   FAKE_MIC=mic_pause_unfinished.wav  "I would like to know | (1.0 s) | what the weather will be like tomorrow."
//      -> the first part ends unfinished ("know"): the session waits for the rest
//   FAKE_MIC=mic_pause_question.wav    "Could you explain? | (1.1 s) | how rain forms in the clouds."
//      -> the first part looks finished: its turn starts, the user goes on, the parts are merged
const CASES = {
  'mic_pause_unfinished.wav': { first: 'like to know', second: 'weather' },
  'mic_pause_question.wav': { first: 'explain', second: 'rain' },
}

test('a pause in the middle of a question does not split it', async ({ page }) => {
  const fixture = process.env.FAKE_MIC
  test.skip(!CASES[fixture], 'needs FAKE_MIC=mic_pause_unfinished.wav or mic_pause_question.wav')
  const { first, second } = CASES[fixture]
  const messages = captureServerMessages(page)
  await openAndStartVoice(page)
  await waitFor(messages, (m) => m.type === 'session', 10_000)

  // the turn that contains the second part of the question
  const full = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice'
    && m.text.toLowerCase().includes(second), 40_000)
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === full.turn_id, 60_000)
  const userTurns = messages.filter((m) => m.type === 'user_message' && m.source === 'voice')
  const answered = messages.filter((m) => m.type === 'assistant_done' && !m.continued && !m.interrupted)
  const summary = { test: 'pause', fixture, user_turns: userTurns.map((m) => m.text),
    continued: messages.filter((m) => m.continued).length, answered: answered.length, reply: done.text }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync(`results/pause_${fixture.replace('.wav', '')}.json`, JSON.stringify(summary, null, 2))

  expect(full.text.toLowerCase()).toContain(first)     // both parts in the question that was answered
  expect(answered).toHaveLength(1)                       // answered once, not the first half separately
  expect(done.interrupted).toBe(false)
  // the chat shows one user message for the question
  await expect(page.locator('li.msg.user')).toHaveCount(1)
})
