import { expect, test } from '@playwright/test'
import { appPath } from './helpers.js'

// Phase 1: typed conversation with streamed replies, explicit state machine, cancellation.

async function waitConnected(page) {
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 20_000 })
}

test('streams a reply and returns to LISTENING', async ({ page }) => {
  await waitConnected(page)
  const states = []
  await page.exposeFunction('__recordState', (s) => states.push(s))
  await page.evaluate(() => {
    const el = document.querySelector('[data-testid="state"]')
    new MutationObserver(() => window.__recordState(el.textContent)).observe(el, { childList: true, characterData: true, subtree: true })
  })

  const t0 = Date.now()
  await page.getByTestId('input').fill('Hi! Please answer in one short sentence: what is your name?')
  await page.getByTestId('send').click()

  const assistant = page.locator('li[data-role="assistant"]').last()
  // reply text has arrived (not only the "…" placeholder); not.toHaveText('…') also passed while the
  // bubble did not exist yet
  await expect(assistant).toHaveText(/\w/, { timeout: 15_000 })
  const firstTextMs = Date.now() - t0
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 30_000 })
  const reply = await assistant.innerText()

  expect(reply.length).toBeGreaterThan(3)
  expect(reply).not.toMatch(/\[|<meta>|\|/) // metadata header is never shown
  expect(states).toContain('Thinking…')
  await expect(page.getByTestId('metrics')).toBeVisible()
  await expect(page.getByTestId('behavior')).toBeVisible()
  console.log(JSON.stringify({ test: 'phase1_stream', firstTextMs, reply, states }))
})

test('Stop cancels a long reply and the next turn still works', async ({ page }) => {
  await waitConnected(page)
  await page.getByTestId('input').fill('Please tell me a long, detailed story about a dragon and a lighthouse keeper.')
  await page.getByTestId('send').click()

  const assistant = page.locator('li[data-role="assistant"]').last()
  // wait for reply text: right after a restart the old check passed before the bubble existed, and
  // Stop then cancelled the turn before it had a bubble to tag "interrupted"
  await expect(assistant).toHaveText(/\w/, { timeout: 15_000 })
  await page.getByTestId('stop').click()
  await expect(assistant.locator('.tag')).toHaveText('interrupted', { timeout: 5_000 })
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING')

  // Text of the cancelled turn must not keep growing (stale deltas are dropped).
  const frozen = await assistant.innerText()
  await page.waitForTimeout(1500)
  expect(await assistant.innerText()).toBe(frozen)

  const replies = page.locator('li[data-role="assistant"]')
  const before = await replies.count()
  await page.getByTestId('input').fill('Now just say OK.')
  await page.getByTestId('send').click()
  await expect(replies).toHaveCount(before + 1, { timeout: 15_000 })   // a new reply, not the cancelled one
  const next = replies.last()
  await expect(next).toHaveText(/\w/, { timeout: 15_000 })
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 30_000 })
  expect((await next.innerText()).length).toBeGreaterThan(0)
})

test('reconnects automatically after the socket drops', async ({ page }) => {
  await waitConnected(page)
  const firstSession = await page.evaluate(() => window.__avatarConn.ws.url)
  await page.evaluate(() => window.__avatarConn.ws.close()) // unexpected drop (not user-initiated)
  await expect(page.getByTestId('conn-status')).not.toHaveText('open', { timeout: 5_000 })
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 15_000 })
  expect(firstSession).toContain('/ws')
  await page.getByTestId('input').fill('Say hi.')
  await page.getByTestId('send').click()
  await expect(page.locator('li[data-role="assistant"]').last()).toHaveText(/\w/, { timeout: 15_000 })
})

test('a wrong access token is rejected when a token is configured', async ({ page }) => {
  test.skip(!process.env.AVATAR_TOKEN, 'only meaningful when the server requires a token')
  await page.goto(appPath('wrong-token'))
  await expect(page.getByTestId('conn-status')).toHaveText('access denied (token)', { timeout: 20_000 })
})
