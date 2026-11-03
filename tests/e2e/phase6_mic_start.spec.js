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
  await expect(page.getByTestId('state')).toHaveText('Ready to chat')
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
  await expect(page.getByTestId('state')).toHaveText('Ready to chat')
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

// Microphones a page can be given (3 Nov 2026: a customer's Chrome took "CABLE Output (VB-Audio Virtual
// Cable)", Windows' default input, and sent only silence while the real microphone worked).
function fakeMicrophones(page, devices, loud) {
  return page.addInitScript(({ devices, loud }) => {
    navigator.mediaDevices.enumerateDevices = async () =>
      devices.map((d) => ({ kind: 'audioinput', groupId: 'g', toJSON() { return this }, ...d }))
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      // like the customer's Chrome: only an exact device is honoured ("ideal" gave the default input)
      const wanted = constraints?.audio?.deviceId?.exact
      const device = wanted ? devices.find((d) => d.deviceId === wanted) : devices[0]
      if (!device) throw new DOMException('no such microphone', 'OverconstrainedError')
      const ctx = new AudioContext()
      const out = ctx.createMediaStreamDestination()
      if (loud.includes(device.deviceId)) {                    // a real microphone: a voice-loud tone
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()
        gain.gain.value = 0.2
        osc.connect(gain).connect(out)
        osc.start()
      }
      const track = out.stream.getAudioTracks()[0]
      Object.defineProperty(track, 'label', { value: device.label })
      track.getSettings = () => ({ deviceId: device.deviceId, sampleRate: 48000, channelCount: 1 })
      window.__micUsed = device.label
      return out.stream
    }
  }, { devices, loud })
}

test('a virtual audio cable is not taken for the microphone: the real one is used at once', async ({ page }) => {
  await fakeMicrophones(page, [
    { deviceId: 'default', label: 'Default - CABLE Output (VB-Audio Virtual Cable)' },
    { deviceId: 'cable', label: 'CABLE Output (VB-Audio Virtual Cable)' },
    { deviceId: 'realmic', label: 'Microphone (Realtek(R) Audio)' },
  ], ['realmic'])
  const size = logSize()
  await open(page)
  await page.getByTestId('talk').click()
  await expect(page.getByTestId('listening')).toBeVisible({ timeout: 15_000 })
  await expect.poll(() => page.evaluate(() => window.__micUsed), { timeout: 10_000 }).toBe('Microphone (Realtek(R) Audio)')
  await page.waitForTimeout(6500)                                // past the 5 s silence check
  await expect(page.getByTestId('stage-error')).toHaveCount(0)
  if (LOCAL) await expect.poll(() => logSince(size), { timeout: 10_000 }).toContain('VirtualMicrophone')
})

test('a microphone that sends no sound offers the others, and one tap switches', async ({ page }) => {
  await fakeMicrophones(page, [
    { deviceId: 'default', label: 'Default - Headset Microphone (Muted Device)' },
    { deviceId: 'muted', label: 'Headset Microphone (Muted Device)' },
    { deviceId: 'usb', label: 'Microphone (USB Audio)' },
  ], ['usb'])
  await open(page)
  await page.getByTestId('talk').click()
  await expect(page.getByTestId('stage-error')).toContainText('sends no sound', { timeout: 15_000 })
  const choice = page.getByTestId('mic-choice')
  await expect(choice).toHaveText('Use Microphone (USB Audio)')
  await choice.click()
  await expect.poll(() => page.evaluate(() => window.__micUsed), { timeout: 10_000 }).toBe('Microphone (USB Audio)')
  await expect(page.getByTestId('stage-error')).toHaveCount(0)
  await page.waitForTimeout(6500)                                // the new microphone passes the check
  await expect(page.getByTestId('stage-error')).toHaveCount(0)
})

test('a microphone chosen earlier but gone now: the default one, and a virtual cable is still skipped', async ({ page }) => {
  await page.addInitScript(() => { try { localStorage.setItem('avatarMic', 'unplugged-usb-mic') } catch {} })
  await fakeMicrophones(page, [
    { deviceId: 'default', label: 'Default - CABLE Output (VB-Audio Virtual Cable)' },
    { deviceId: 'cable', label: 'CABLE Output (VB-Audio Virtual Cable)' },
    { deviceId: 'yeti', label: 'Microphone (Yeti Stereo Microphone) (046d:0ab7)' },
    { deviceId: 'buds', label: 'Headset (2- OpenBuds P7) (Bluetooth)' },
  ], ['yeti', 'buds'])
  await open(page)
  await page.getByTestId('talk').click()
  await expect(page.getByTestId('listening')).toBeVisible({ timeout: 15_000 })
  await expect.poll(() => page.evaluate(() => window.__micUsed), { timeout: 10_000 })
    .toBe('Microphone (Yeti Stereo Microphone) (046d:0ab7)')
  await page.waitForTimeout(6500)
  await expect(page.getByTestId('stage-error')).toHaveCount(0)
})
