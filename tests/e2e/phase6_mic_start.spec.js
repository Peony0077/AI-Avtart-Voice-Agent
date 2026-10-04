import fs from 'node:fs'
import { expect, test } from '@playwright/test'
import { appPath, captureServerMessages, waitFor } from './helpers.js'

// Owner: "this project didn't hear my words". None of the owner's sessions ever started the
// microphone: the badge said "Listening" while it was off, and a microphone that could not start said
// so only in small text in the header and never told the server. Runs in the "mic" project: a browser
// without the autoplay exemption of the voice tests.
test.setTimeout(120_000)

const LOG = new URL('../../logs/backend.log', import.meta.url)
const LOCAL = !process.env.AVATAR_URL || /\/\/(127\.0\.0\.1|localhost)[:/]/.test(process.env.AVATAR_URL)
const logSince = (size) => (LOCAL ? fs.readFileSync(LOG).subarray(size).toString('utf8') : '')
const logSize = () => (LOCAL ? fs.statSync(LOG).size : 0)

async function open(page) {
  const messages = captureServerMessages(page)
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  return messages
}

test('the badge says Listening only with the microphone on; one tap on the avatar starts it', async ({ page }) => {
  const messages = await open(page)
  await expect(page.getByTestId('state')).toHaveText('Ready · mic off')
  await expect(page.getByTestId('talk')).toBeVisible()
  await page.getByTestId('talk').click()
  await waitFor(messages, (m) => m.type === 'voice' && m.enabled, 15_000)
  await expect(page.getByTestId('listening')).toBeVisible()
  await expect(page.getByTestId('talk')).toHaveCount(0)
  // the fake microphone asks "What is the capital of France?" (after the greeting is interrupted or done)
  const heard = await waitFor(messages, (m) => m.type === 'user_message' && m.source !== 'greeting' && /capital/i.test(m.text), 60_000)
  expect(heard.text).toMatch(/capital of France/i)
  const state = await page.getByTestId('state').getAttribute('data-state')
  console.log(JSON.stringify({ test: 'phase6_mic_start', heard: heard.text, state }))
})

test('a blocked microphone is explained on the avatar and reported to the server', async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'))
  })
  const size = logSize()
  await open(page)
  await page.getByTestId('talk').click()
  await expect(page.getByTestId('stage-error')).toContainText('microphone is blocked')
  await expect(page.getByTestId('talk')).toHaveText('Try again')
  await expect(page.getByTestId('state')).toHaveText('Ready · mic off')
  if (LOCAL) await expect.poll(() => logSince(size), { timeout: 10_000 }).toContain('browser problem (voice_start): NotAllowedError')
})

test('a browser without microphone access (in-app browser) is told what to do', async ({ page }) => {
  await page.addInitScript(() => { Object.defineProperty(navigator, 'mediaDevices', { value: undefined }) })
  await open(page)
  await page.getByTestId('voice-start').click()
  await expect(page.getByTestId('stage-error')).toContainText('cannot use the microphone here')
})

test('a muted microphone (only silence) is pointed out instead of listening to nothing', async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      const ctx = new AudioContext()
      return ctx.createMediaStreamDestination().stream        // a track that only carries zeros
    }
  })
  const size = logSize()
  await open(page)
  await page.getByTestId('talk').click()
  await expect(page.getByTestId('listening')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('stage-error')).toContainText('sends no sound', { timeout: 15_000 })
  if (LOCAL) await expect.poll(() => logSince(size), { timeout: 10_000 }).toContain('SilentMicrophone')
})
