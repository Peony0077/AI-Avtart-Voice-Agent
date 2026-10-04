import { expect, test } from '@playwright/test'
import { appPath, captureServerMessages, waitFor } from './helpers.js'

// Typed messages are spoken (and lip-synced by the avatar) without starting the microphone;
// Mute makes replies text-only. Found in user testing: typed replies were text-only, so the
// avatar never spoke unless "Start voice" had been clicked.
test('a typed message is spoken and lip-synced without the microphone; Mute makes it text-only', async ({ page }) => {
  const messages = captureServerMessages(page)
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(session.voice_available === false, 'voice models not loaded')

  await page.getByTestId('input').fill('What is the capital of France? One short sentence.')
  await page.getByTestId('send').click()
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'text', 10_000)
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 60_000)
  expect(done.text.toLowerCase()).toContain('paris')
  const metrics = await waitFor(messages, (m) => m.type === 'metrics' && m.turn_id === q.turn_id, 10_000)
  expect(metrics.speech_ms).toBeGreaterThan(500)                        // the reply was synthesised ...
  expect(metrics.ms_from_t0.client_first_audio).toBeGreaterThan(0)      // ... and played in the browser
  await expect(page.getByTestId('mic')).toHaveCount(0)                   // microphone was never started
  if (session.avatar_available) {
    const speech = await page.evaluate(() => window.__avatarStats.drawnSpeech)
    expect(speech).toBeGreaterThan(10)                                  // the avatar's lips moved
  }

  // Mute: the next reply is text only
  await page.getByTestId('speaker-toggle').click()
  await expect(page.getByTestId('speaker-toggle')).toHaveText('Unmute')
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 30_000 })
  await page.getByTestId('input').fill('And the capital of Italy? One short sentence.')
  await page.getByTestId('send').click()
  const q2 = await waitFor(messages, (m) => m.type === 'user_message' && m.turn_id > q.turn_id, 10_000)
  const m2 = await waitFor(messages, (m) => m.type === 'metrics' && m.turn_id === q2.turn_id, 60_000)
  expect(m2.speech_ms).toBeNull()
  console.log(JSON.stringify({ test: 'phase6_typed_speech', reply: done.text, first_audio_ms: metrics.ms_from_t0.client_first_audio,
    speech_ms: metrics.speech_ms, muted_reply_speech_ms: m2.speech_ms }))
})

// Found while fixing the above: closing the tab while the avatar spoke left the avatar worker waiting
// for a "playback ended" that never came, so the next visitor got no idle video at all.
test('closing the tab while the avatar speaks does not freeze the avatar for the next visitor', async ({ browser }) => {
  const first = await browser.newPage()
  const messages = captureServerMessages(first)
  await first.goto(appPath())
  await expect(first.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(!session.avatar_available, 'backend not started with an avatar profile')
  await first.getByTestId('input').fill('Please tell me a long, detailed story about a dragon and a lighthouse keeper.')
  await first.getByTestId('send').click()
  await expect(first.getByTestId('state')).toHaveText('Speaking', { timeout: 30_000 })
  await first.waitForTimeout(1500)                                       // audio is playing
  await first.close()                                                    // gone mid-reply

  const second = await browser.newPage()
  await second.goto(appPath())
  await expect(second.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  await second.waitForTimeout(2500)
  const idle = await second.evaluate(() => window.__avatarStats.idle)
  expect(idle).toBeGreaterThan(20)                                       // idle video streams (~20 fps)
  await second.close()
})

// Warm persona: good news gets a smile (while thinking and while speaking); sad news never does.
test('the avatar smiles at good news and not at sad news', async ({ page }) => {
  const messages = captureServerMessages(page)
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(!session.avatar_available, 'backend not started with an avatar profile')

  const ask = async (text) => {
    await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 60_000 })
    await page.getByTestId('input').fill(text)
    await page.getByTestId('send').click()
    const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === text, 10_000)
    return q
  }
  const happy = await ask('I just got the job I applied for!')
  await waitFor(messages, (m) => m.type === 'state' && m.state === 'SPEAKING' && m.turn_id === happy.turn_id, 30_000)
  await page.waitForTimeout(900)
  await page.getByTestId('avatar-canvas').screenshot({ path: 'results/avatar_smiling.png' })
  await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === happy.turn_id, 60_000)
  const faces = (q) => messages.filter((m) => m.type === 'behavior' && m._t >= q._t && m._t <= (messages.find(
    (d) => d.type === 'assistant_done' && d.turn_id === q.turn_id)?._t ?? Infinity)).map((m) => m.expression)
  expect(faces(happy)).toContain('smile')

  const sad = await ask('My grandmother passed away last week.')
  await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === sad.turn_id, 60_000)
  expect(faces(sad)).not.toContain('smile')
  console.log(JSON.stringify({ test: 'warmth', happy_faces: faces(happy), sad_faces: faces(sad) }))
})
