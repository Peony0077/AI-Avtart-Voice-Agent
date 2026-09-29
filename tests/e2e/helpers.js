import { expect } from '@playwright/test'
import fs from 'node:fs'

export const manifest = JSON.parse(fs.readFileSync(new URL('./fixtures/manifest.json', import.meta.url)))

// Collect every JSON message the server sends (via Playwright's WebSocket inspection).
export function captureServerMessages(page) {
  const messages = []
  page.on('websocket', (ws) => {
    ws.on('framereceived', ({ payload }) => {
      if (typeof payload === 'string') {
        try {
          messages.push({ ...JSON.parse(payload), _t: Date.now() })
        } catch { /* ignore */ }
      }
    })
  })
  return messages
}

export async function openAndStartVoice(page) {
  await page.goto('/')
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  await expect(page.getByTestId('voice-start')).toBeEnabled({ timeout: 30_000 })
  await page.getByTestId('voice-start').click()
  await expect(page.getByTestId('mic')).toBeVisible({ timeout: 15_000 })
  return Date.now() // fake-mic file starts playing ~now
}

export async function waitFor(messages, predicate, timeoutMs = 60_000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const hit = messages.find(predicate)
    if (hit) return hit
    await new Promise((r) => setTimeout(r, 50))
  }
  throw new Error(`timed out waiting for server message (${timeoutMs} ms)`)
}

export function turnMetrics(messages, turnId) {
  return messages.find((m) => m.type === 'metrics' && m.turn_id === turnId)
}
