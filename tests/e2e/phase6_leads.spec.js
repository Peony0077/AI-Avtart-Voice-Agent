import fs from 'node:fs'
import { expect, test } from '@playwright/test'
import { appPath, captureServerMessages, customerToken, waitFor } from './helpers.js'

// Lead generation (owner, 2026-10-05): with a business set, she guides the customer and asks for their
// contact details; after the conversation the lead is written down and the owner sees it under Leads.
// Also: the voice preview in Settings, and the time in the customer's own time zone (US customers).
// Puts the owner's persona, leads and memory files back afterwards.
test.setTimeout(300_000)

const DATA = (name) => new URL(`../../data/${name}`, import.meta.url)
const LOCAL = !process.env.AVATAR_URL || /\/\/(127\.0\.0\.1|localhost)[:/]/.test(process.env.AVATAR_URL)
const keep = (name) => (LOCAL && fs.existsSync(DATA(name)) ? fs.readFileSync(DATA(name)) : null)
const putBack = (name, saved) => {
  if (!LOCAL) return
  if (saved) fs.writeFileSync(DATA(name), saved)
  else if (fs.existsSync(DATA(name))) fs.rmSync(DATA(name))
}

async function open(page) {
  const messages = captureServerMessages(page)
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  await waitFor(messages, (m) => m.type === 'session', 15_000)
  return messages
}

const reply = async (page, messages, text) => {
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 60_000 })
  await page.getByTestId('input').fill(text)
  await page.getByTestId('send').click()
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === text, 10_000)
  return (await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 90_000)).text
}

test('a conversation with a business set becomes a lead the owner sees', async ({ browser }) => {
  const saved = { persona: keep('persona.json'), leads: keep('leads.jsonl'), memory: keep('memory.json') }
  const page = await (await browser.newContext()).newPage()
  let messages = await open(page)
  try {
    await page.getByTestId('settings-open').click()
    await page.getByTestId('business-company').fill('Sunny Roofs Solar')
    await page.getByTestId('business-offer').fill('Solar panels for homes in Texas, with installation and a 25-year warranty.')
    await page.getByTestId('business-next-step').fill('book a free roof check')
    await expect(page.getByTestId('lead-gen')).toBeChecked()
    await page.getByTestId('persona-save').click()
    await waitFor(messages, (m) => m.type === 'persona' && m.company === 'Sunny Roofs Solar', 10_000)
    // voice preview: a WAV sample of the chosen voice
    await page.getByTestId('voice-preview').click()
    const preview = await waitFor(messages, (m) => m.type === 'voice_preview', 30_000)
    expect(preview.wav_b64?.length).toBeGreaterThan(20_000)
    await page.keyboard.press('Escape')

    const a1 = await reply(page, messages, "Hi! I'm thinking about solar panels for my house in Austin.")
    const a2 = await reply(page, messages, "It's a three-bedroom house and I'd like to start this summer.")
    const a3 = await reply(page, messages, "Sure. I'm Sam Rivera, and my email is sam.rivera@example.com.")
    console.log(JSON.stringify({ test: 'phase6_leads', replies: [a1, a2, a3] }))
    expect(`${a1} ${a2}`).toMatch(/\?/)                         // she asks about their needs
    expect(a3).toMatch(/Sam/)                                  // and uses the name once she knows it
    await page.close()                                         // the conversation ends: the lead is written down

    const owner = await (await browser.newContext()).newPage()
    messages = await open(owner)
    let rows = []
    for (let i = 0; i < 30 && !rows.some((r) => r.email === 'sam.rivera@example.com'); i++) {
      await owner.evaluate(() => window.__avatarConn.sendJson({ type: 'get_leads' }))
      rows = (await waitFor(messages, (m) => m.type === 'leads' && m._t > Date.now() - 900, 10_000)).leads
      if (!rows.some((r) => r.email === 'sam.rivera@example.com')) await owner.waitForTimeout(1000)
    }
    const lead = rows.find((r) => r.email === 'sam.rivera@example.com')
    console.log(JSON.stringify({ test: 'phase6_leads', lead }))
    expect(lead).toBeTruthy()
    expect(lead.name).toMatch(/Sam/)
    expect(lead.need).toMatch(/solar/i)
    await owner.getByTestId('leads-open').click()
    await expect(owner.getByTestId('leads-table')).toContainText('sam.rivera@example.com')
    await owner.close()
  } finally {
    // the owner's own settings, leads and memory as they were
    const fix = await (await browser.newContext()).newPage()
    const m = await open(fix)
    if (saved.persona) {
      const p = JSON.parse(saved.persona.toString('utf8'))
      await fix.evaluate((persona) => window.__avatarConn.sendJson({ type: 'set_persona', persona }), p)
      await waitFor(m, (x) => x.type === 'persona' && x.company === (p.company ?? ''), 10_000)
    } else {
      await fix.evaluate(() => window.__avatarConn.sendJson({ type: 'set_persona',
        persona: { company: '', offer: '', next_step: '', lead_gen: true } }))
      await waitFor(m, (x) => x.type === 'persona' && x.company === '', 10_000)
    }
    await fix.close()
    putBack('leads.jsonl', saved.leads)
    putBack('memory.json', saved.memory)
    if (saved.persona) putBack('persona.json', saved.persona)
  }
})

test('the time is the customer\'s own, said the American way', async ({ browser }) => {
  // a customer in New York, on the customer link (their own, empty memory)
  const token = customerToken()
  test.skip(!token, 'no customer link: start the backend with scripts\\tunnel.ps1 or scripts/tunnel.sh, or set AVATAR_CUSTOMER_TOKEN')
  const page = await (await browser.newContext({ timezoneId: 'America/New_York' })).newPage()
  const messages = captureServerMessages(page)
  await page.goto(appPath(token))
  await waitFor(messages, (m) => m.type === 'session', 30_000)
  const answer = await reply(page, messages, 'What time is it right now?')
  const ny = new Date().toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: 'numeric' })  // "9 PM"
  const hour = ny.split(' ')[0]
  console.log(JSON.stringify({ test: 'phase6_time', answer, ny }))
  expect(answer).toMatch(new RegExp(`\\b(${hour}|${(Number(hour) % 12) + 1})(:\\d\\d)?\\s*(AM|PM|a\\.m\\.|p\\.m\\.|o'clock)?`, 'i'))
  await page.close()
})
