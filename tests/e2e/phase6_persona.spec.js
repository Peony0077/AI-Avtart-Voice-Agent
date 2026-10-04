import fs from 'node:fs'
import { expect, test } from '@playwright/test'
import { appPath, captureServerMessages, waitFor } from './helpers.js'

// Features of Anam / HeyGen LiveAvatar: persona settings (name, personality, greeting, voice), a greeting
// when the conversation starts, memory of the user across sessions (with "Forget me"), the current date,
// and a transcript download. Restores the persona it found.
test.setTimeout(300_000)

const reply = async (page, messages, text) => {
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 60_000 })
  await page.getByTestId('input').fill(text)
  await page.getByTestId('send').click()
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === text, 10_000)
  return (await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 90_000)).text
}

const open = async (page) => {
  const messages = captureServerMessages(page)
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  return messages
}

const greet = async (page, messages) => {
  // what "Start voice" sends (here without a microphone): speaker on, then greet
  await page.evaluate(() => { window.__avatarConn.sendJson({ type: 'speaker', enabled: true }); window.__avatarConn.sendJson({ type: 'greet' }) })
  const g = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'greeting', 15_000)
  // no audio player runs in this tab without a click: report the greeting as played
  const done = await waitFor(messages, (m) => m.type === 'tts_done' && m.turn_id === g.turn_id, 30_000)
  await page.evaluate(([id, ms]) => window.__avatarConn.sendJson({ type: 'playback', turn_id: id, event: 'ended', played_ms: ms }),
    [g.turn_id, done.total_ms])
  return (await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === g.turn_id, 60_000)).text
}

// What she remembers about the owner (read by the server on every use; erased by this test, which then
// teaches her about "Sam"): put back afterwards when the server runs on this PC.
const MEMORY = new URL('../../data/memory.json', import.meta.url)
const LOCAL = !process.env.AVATAR_URL || /\/\/(127\.0\.0\.1|localhost)[:/]/.test(process.env.AVATAR_URL)

test('persona, greeting, memory across sessions, date and transcript', async ({ browser }) => {
  const savedMemory = LOCAL && fs.existsSync(MEMORY) ? fs.readFileSync(MEMORY) : null
  const page = await browser.newPage()
  let messages = await open(page)

  // settings: read the current persona (restored at the end, also when the test fails), then change it
  await page.getByTestId('settings-open').click()
  const original = await waitFor(messages, (m) => m.type === 'persona', 10_000)
  try {
    await run(browser, page, messages, original)
  } finally {
    const fix = await browser.newPage()
    const m = await open(fix)
    await fix.evaluate((o) => {
      window.__avatarConn.sendJson({ type: 'set_persona', persona: { name: o.name, personality: o.personality,
        greeting: o.greeting, voice: o.voice, memory: o.memory } })
      window.__avatarConn.sendJson({ type: 'forget_memory' })        // what the test told her about "Sam"
    }, original)
    await waitFor(m, (x) => x.type === 'persona' && x.name === original.name && !x.remembered, 10_000)
    await fix.close()
    // the test's sessions began before that "Forget me", so their memory updates are dropped
    if (savedMemory) fs.writeFileSync(MEMORY, savedMemory)
  }
})

async function run(browser, page, messages, original) {
  await page.getByTestId('persona-name').fill('Mia')
  await page.getByTestId('persona-personality').fill('Cheerful and warm, loves cooking.')
  await page.getByTestId('persona-greeting').fill("Hello! I'm Mia. It's lovely to see you.")
  await page.getByTestId('persona-memory').check()
  await page.getByTestId('persona-save').click()
  await waitFor(messages, (m) => m.type === 'persona' && m.name === 'Mia', 10_000)
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('settings')).not.toBeVisible()
  // start from no memory, always: the sessions of the tests before this one are still being merged into
  // it in the background ("The user asked what the capital of France is."); "Forget me" drops those too
  await page.evaluate(() => window.__avatarConn.sendJson({ type: 'forget_memory' }))

  // the greeting: spoken, and no "You" bubble for it
  const hello = await greet(page, messages)
  expect(hello).toContain('Mia')
  await expect(page.locator('li[data-role="user"]')).toHaveCount(0)
  const name = await reply(page, messages, "What's your name?")
  expect(name).toMatch(/Mia/)
  const day = await reply(page, messages, 'What day of the week is it today?')
  const weekday = new Date().toLocaleDateString('en-US', { weekday: 'long' })
  expect(day).toContain(weekday)
  await reply(page, messages, 'By the way, my name is Sam and I love hiking in the mountains.')

  // transcript download: made in the page
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('transcript').click()])
  const transcript = fs.readFileSync(await download.path(), 'utf-8')
  expect(transcript).toContain('You: By the way, my name is Sam')
  expect(transcript).toMatch(/Mia: .+/)
  await page.close()                        // the session ends: she updates her memory of the user

  // a new session: she remembers Sam, and welcomes him back
  const page2 = await browser.newPage()
  messages = await open(page2)
  await page2.getByTestId('settings-open').click()
  let p = await waitFor(messages, (m) => m.type === 'persona', 10_000)
  for (let i = 0; i < 20 && !p.remembered; i++) {        // the memory update runs after the first session
    await page2.waitForTimeout(1000)
    await page2.evaluate(() => window.__avatarConn.sendJson({ type: 'get_persona' }))
    p = await waitFor(messages, (m) => m.type === 'persona' && m._t > Date.now() - 900, 5_000)
  }
  expect(p.remembered).toMatch(/Sam/)
  expect(p.remembered).toMatch(/hik/i)
  // only what the user said: not the avatar's cooking talk, not the date, no guessed gender
  expect(p.remembered).not.toMatch(/(he|she|his|her|him)|cook|recipe|october|sunday/i)
  await expect(page2.getByTestId('memory')).toContainText('Sam')
  await page2.keyboard.press('Escape')
  // the memory is in the new session (it started before the update finished, so open a fresh one)
  await page2.close()
  const page3 = await browser.newPage()
  messages = await open(page3)
  const welcome = await greet(page3, messages)
  expect(welcome).toMatch(/Sam/)

  // Forget me erases it
  await page3.getByTestId('settings-open').click()
  await page3.getByTestId('forget').click()
  await waitFor(messages, (m) => m.type === 'persona' && !m.remembered, 10_000)
  await expect(page3.getByTestId('memory')).toContainText('nothing yet')
  console.log(JSON.stringify({ test: 'phase6_persona', hello, name, day, remembered: p.remembered, welcome }))

  await page3.close()
}
