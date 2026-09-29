import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { captureServerMessages, openAndStartVoice, waitFor } from './helpers.js'

// Phase 5: spoken question answered from the user's documents, with citations shown.
// FAKE_MIC=mic_rag.wav ("What are the opening hours of the Moonlight Bakery on Sundays?")
test('voice question is answered from the documents with a citation', async ({ page }) => {
  test.skip(process.env.FAKE_MIC !== 'mic_rag.wav', 'needs FAKE_MIC=mic_rag.wav')
  const messages = captureServerMessages(page)
  await openAndStartVoice(page)
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice', 30_000)
  expect(q.text.toLowerCase()).toContain('moonlight bakery')
  const cites = await waitFor(messages, (m) => m.type === 'citations' && m.turn_id === q.turn_id, 30_000)
  expect(cites.sources.map((s) => s.source)).toContain('moonlight_bakery.md')
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 60_000)
  expect(done.text).toMatch(/9/)
  expect(done.text).toMatch(/1 ?p\.?m|13|one/i)
  await expect(page.getByTestId('sources').last()).toContainText('Moonlight Bakery')
  const metrics = await waitFor(messages, (m) => m.type === 'metrics' && m.turn_id === q.turn_id, 10_000)
  const summary = { test: 'phase5_rag', transcript: q.text, reply: done.text, sources: cites.sources.map((s) => [s.source, s.score]),
    rag: metrics.rag, first_audio_ms: metrics.ms_from_t0.client_first_audio }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/phase5_rag.json', JSON.stringify(summary, null, 2))
})
