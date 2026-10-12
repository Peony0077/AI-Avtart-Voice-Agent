import { expect, test } from '@playwright/test'
import { appPath, captureServerMessages, customerToken, waitFor } from './helpers.js'

// The customer link (2026-10-05: customers in the US and Canada use the page too): only the
// conversation, no settings or leads; one conversation at a time, and nobody cuts off a customer -
// the next one waits in line and starts automatically.
test.setTimeout(180_000)

const NO_CUSTOMER_LINK = 'no customer link: start the backend with scripts\\tunnel.ps1 or scripts/tunnel.sh, or set AVATAR_CUSTOMER_TOKEN'

async function open(page, token) {
  const messages = captureServerMessages(page)
  await page.goto(appPath(token))
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  return messages
}

test('a customer sees only the conversation and cannot change the settings', async ({ browser }) => {
  const token = customerToken()
  test.skip(!token, NO_CUSTOMER_LINK)
  const page = await (await browser.newContext()).newPage()
  const messages = await open(page, token)
  const session = await waitFor(messages, (m) => m.type === 'session', 15_000)
  expect(session.role).toBe('customer')
  await expect(page.getByTestId('talk')).toBeVisible()
  await expect(page.getByTestId('settings-open')).toHaveCount(0)
  await expect(page.getByTestId('leads-open')).toHaveCount(0)
  await expect(page.locator('.metrics')).toHaveCount(0)
  // a customer's attempt to change her is refused by the server
  await page.evaluate(() => window.__avatarConn.sendJson({ type: 'set_persona', persona: { name: 'Hacked' } }))
  await page.evaluate(() => window.__avatarConn.sendJson({ type: 'get_persona' }))
  const p = await waitFor(messages, (m) => m.type === 'persona', 10_000)
  expect(p.name).not.toBe('Hacked')
  expect(p.voices).toBeUndefined()                         // no settings details for customers
  await page.close()
})

test('a second customer waits in line and starts when the first leaves', async ({ browser }) => {
  const token = customerToken()
  test.skip(!token, NO_CUSTOMER_LINK)
  const first = await (await browser.newContext()).newPage()
  const m1 = await open(first, token)
  await waitFor(m1, (m) => m.type === 'session', 15_000)
  const second = await (await browser.newContext()).newPage()        // another browser: another customer
  const m2 = await open(second, token)
  await waitFor(m2, (m) => m.type === 'waiting' && m.position === 1, 15_000)
  await expect(second.getByTestId('waiting')).toContainText('number 1 in line')
  await expect(second.getByTestId('talk')).toHaveCount(0)
  // the first one was not cut off once it had the conversation (before it, it may have waited a few
  // seconds for the previous test's closed page: its place is kept briefly for a reload, api/ws.py)
  const started = m1.findIndex((m) => m.type === 'session')
  expect(m1.slice(started).some((m) => m.type === 'waiting')).toBe(false)
  await first.close()
  await waitFor(m2, (m) => m.type === 'session', 15_000)
  await expect(second.getByTestId('waiting')).toHaveCount(0)
  await expect(second.getByTestId('talk')).toBeVisible()
  await second.close()
})

test('the same customer reconnecting continues instead of waiting behind themselves', async ({ browser }) => {
  const token = customerToken()
  test.skip(!token, NO_CUSTOMER_LINK)
  const ctx = await browser.newContext()                             // one browser (one visitor id)
  const a = await ctx.newPage()
  const ma = await open(a, token)
  await waitFor(ma, (m) => m.type === 'session', 15_000)
  const b = await ctx.newPage()
  const mb = await open(b, token)
  await waitFor(mb, (m) => m.type === 'session', 15_000)
  expect(mb.some((m) => m.type === 'waiting')).toBe(false)
  await ctx.close()
})
