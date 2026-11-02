import { expect, test } from '@playwright/test'
import { appPath, captureServerMessages, customerToken, openAndStartVoice, waitFor } from './helpers.js'

// Houston Party Network mode (backend profile hpn): the avatar answers as the website's Event helper.
// Run against a backend started with --profile hpn:
//   npx playwright test --project hpn            (fake microphone: mic_hpn_wedding.wav)
// Every answer must be HPN's own: the prices from h-townparty.com's catalogue, its buttons, its follow-ups.

const SITE = 'https://h-townparty.com'

async function connected(page, path = appPath()) {
  await page.goto(path)
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
}

// first: the fake microphone starts playing as soon as the page opens, so this test must not wait in line
// behind the previous test's conversation (its place is kept 4 s for a reconnect)
test('a spoken question is answered from the HPN knowledge and spoken back', async ({ page }) => {
  const messages = captureServerMessages(page)
  await openAndStartVoice(page)
  const said = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice', 40_000)
  expect(said.text.toLowerCase()).toContain('wedding')
  const helper = await waitFor(messages, (m) => m.type === 'helper_reply' && m.turn_id === said.turn_id, 15_000)
  expect(helper.reply.intent).toBe('prices.wedding')
  await waitFor(messages, (m) => m.type === 'tts_segment' && m.turn_id === said.turn_id, 20_000)
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === said.turn_id, 90_000)
  expect(done.interrupted).toBe(false)
  expect(done.text).toContain('Signature Wedding, most popular: starting at $1,495.')
})

test('a typed question gets the HPN answer: prices, buttons to the website, follow-ups', async ({ page }) => {
  const messages = captureServerMessages(page)
  await connected(page)
  const session = await waitFor(messages, (m) => m.type === 'session')
  expect(session.hpn?.site).toBe(SITE)

  await page.getByTestId('input').fill('How much is a wedding?')
  await page.getByTestId('send').click()
  const helper = await waitFor(messages, (m) => m.type === 'helper_reply' && m.reply.intent === 'prices.wedding', 15_000)
  expect(helper.reply.cards.map((c) => c.title)).toEqual(
    ['Reception Essentials', 'Signature Wedding (Most Popular)', 'Headliner Wedding Experience'])
  expect(helper.reply.cards.map((c) => c.subtitle)).toEqual(['Starting at $799', 'Starting at $1,495', 'Starting at $2,295'])

  const assistant = page.locator('li[data-role="assistant"]').last()
  await expect(assistant).toContainText('Craig plans each wedding personally', { timeout: 15_000 })
  await expect(assistant).toContainText('Starting at $1,495')
  // the buttons open the website (in a new tab)
  const plan = assistant.locator(`a[href="${SITE}/contact-us/?event=wedding"]`)
  await expect(plan).toBeVisible()
  await expect(plan).toHaveAttribute('target', '_blank')
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 90_000 })   // the price list is ~30 s of speech

  // a suggested next question is asked like a typed one
  await assistant.getByRole('button', { name: 'What does Signature Wedding include?' }).click()
  await waitFor(messages, (m) => m.type === 'helper_reply' && m.reply.intent === 'prices.package', 15_000)
  await expect(page.locator('li[data-role="assistant"]').last()).toContainText('Everything in Reception Essentials, plus:',
    { timeout: 15_000 })
})

test('money and terms nobody has written are never made up', async ({ page }) => {
  const messages = captureServerMessages(page)
  await connected(page)
  await page.getByTestId('input').fill('How much is the deposit?')
  await page.getByTestId('send').click()
  const helper = await waitFor(messages, (m) => m.type === 'helper_reply', 15_000)
  expect(helper.reply.intent).toBe('smalltalk.policy_deposit')
  const done = await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === helper.turn_id, 30_000)
  expect(done.text).toContain("That's one for the team")
  expect(done.text).not.toMatch(/\$\d/)
})

test('the customer link shows only the conversation, with the HPN welcome', async ({ page }) => {
  const token = customerToken()
  test.skip(!token, 'needs the customer link (scripts/tunnel.sh or AVATAR_CUSTOMER_TOKEN)')
  await connected(page, appPath(token))
  await expect(page.getByTestId('settings-open')).toHaveCount(0)
  await expect(page.getByTestId('leads-open')).toHaveCount(0)
  await expect(page.getByRole('button', { name: "What's on tonight?" }).first()).toBeVisible()
})

test('contact details are read back and sent only after yes (test mode: nothing reaches HPN)', async ({ page }) => {
  const messages = captureServerMessages(page)
  await connected(page)
  const ask = async (text, intent) => {
    const before = messages.filter((m) => m.type === 'helper_reply').length
    await page.getByTestId('input').fill(text)
    await page.getByTestId('send').click()
    const reply = await waitFor(messages, (m) => m.type === 'helper_reply'
      && messages.filter((x) => x.type === 'helper_reply').indexOf(m) >= before, 15_000)
    expect(reply.reply.intent).toBe(intent)
    return reply.reply
  }
  await ask('Can someone call me back? My email is avatar.test@example.com', 'enquiry.ask_name')
  const confirm = await ask('Avatar Test', 'enquiry.confirm')
  expect(confirm.text).toContain('Email: **avatar.test@example.com**')
  await expect(page.locator('li[data-role="assistant"]').last()).toContainText('Name: Avatar Test', { timeout: 15_000 })
  // safety: only ever say yes in test mode; a live server is never sent a test enquiry
  if (!confirm.text.includes('Test mode')) {
    await ask('cancel', 'enquiry.cancelled')
    test.skip(true, 'the server sends enquiries to HPN (live): not testing the send')
  }
  const sent = await ask('Yes, send it', 'enquiry.sent_test')
  expect(sent.text).toContain('Nothing was sent')
})
