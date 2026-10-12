import { expect, test } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import { captureServerMessages, openAndStartVoice, waitFor, withOwnerToken } from './helpers.js'

// Phase 6: crash recovery. Kills llama-server, then the avatar worker, in the middle of a session and checks
// that the supervisor / watchdog bring them back and the same browser session keeps working.
// Backend: scripts\run.ps1 / scripts/run.sh with the avatar (or avatar_clone) profile on THIS machine (the test
// kills local processes).
// FAKE_MIC=mic_question.wav

// Windows: a PowerShell filter on Win32_Process; Linux/macOS: a pattern for the full command line (pgrep -f)
function killProcesses({ windows, posix }) {
  if (process.platform === 'win32') {
    const out = execFileSync('powershell', ['-NoProfile', '-Command',
      `Get-CimInstance Win32_Process | Where-Object { ${windows} } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $_.ProcessId }`],
    { encoding: 'utf8' })
    return out.split(/\s+/).filter(Boolean)
  }
  let pids = []
  try {
    pids = execFileSync('pgrep', ['-f', posix], { encoding: 'utf8' }).split(/\s+/).filter(Boolean)
  } catch {
    return []                                   // pgrep exits 1 when nothing matches
  }
  for (const pid of pids) {
    try { process.kill(Number(pid), 'SIGKILL') } catch { /* already gone */ }
  }
  return pids
}

async function health(page) {
  return (await page.request.get(withOwnerToken('/health'))).json()
}

async function until(page, predicate, timeoutMs) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const h = await health(page)
    if (predicate(h)) return { h, ms: Date.now() - start }
    await page.waitForTimeout(500)
  }
  throw new Error('backend did not recover in time')
}

async function ask(page, messages, text, timeoutMs = 90_000) {
  await expect(page.getByTestId('state')).toHaveAttribute('data-state', 'LISTENING', { timeout: 60_000 })
  await page.getByTestId('input').fill(text)
  await page.getByTestId('send').click()
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === text, 10_000)
  return waitFor(messages, (m) => (m.type === 'assistant_done' || m.type === 'error') && m.turn_id === q.turn_id, timeoutMs)
}

test('recovers from llama-server and avatar-worker crashes in the middle of a session', async ({ page }) => {
  test.setTimeout(420_000)
  const messages = captureServerMessages(page)
  await openAndStartVoice(page)
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(!session.avatar_available, 'backend not started with an avatar profile')
  const q = await waitFor(messages, (m) => m.type === 'user_message' && m.source === 'voice', 30_000)
  await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 60_000)
  const h0 = await health(page)

  // 1) LLM server crash
  const llmPids = killProcesses({ windows: "$_.Name -eq 'llama-server.exe'", posix: 'tools/llama.cpp/current/llama-server' })
  expect(llmPids.length).toBeGreaterThan(0)
  const tKill = Date.now()
  // asked while llama-server is down: the backend waits for the watchdog's restart instead of failing
  const afterLlm = await ask(page, messages, 'What is two plus three? Answer in one short sentence.', 150_000)
  const llmTurnMs = Date.now() - tKill
  expect(afterLlm.type).toBe('assistant_done')
  expect(afterLlm.text).toMatch(/five|5/i)
  const h1 = await health(page)
  expect(h1.llm_restarts).toBeGreaterThan(h0.llm_restarts)
  expect(h1.llm_ready).toBe(true)

  // 2) avatar worker crash: the conversation continues with audio only, then the avatar comes back
  const speechBefore = await page.evaluate(() => window.__avatarStats.speech)
  const avPids = killProcesses({ windows: "$_.Name -eq 'python.exe' -and $_.CommandLine -like '*avatar*musetalk*worker.py*'",
    posix: 'avatar/musetalk/worker.py' })
  expect(avPids.length).toBeGreaterThan(0)
  await until(page, (h) => !h.avatar.connected || h.avatar.restarts > h0.avatar.restarts, 20_000)
  const duringOutage = await ask(page, messages, 'Name one colour of the rainbow, in one short sentence.')
  expect(duringOutage.type).toBe('assistant_done')                       // audio-only while the avatar restarts
  const avatar = await until(page, (h) => h.avatar.connected && h.avatar.restarts > h0.avatar.restarts, 240_000)
  const afterAvatar = await ask(page, messages, 'What is the capital of Italy? Answer in one short sentence.')
  expect(afterAvatar.type).toBe('assistant_done')
  expect(afterAvatar.text.toLowerCase()).toContain('rome')
  await page.waitForTimeout(1500)
  const speechAfter = await page.evaluate(() => window.__avatarStats.speech)
  expect(speechAfter).toBeGreaterThan(speechBefore + 10)                  // lip-sync frames flow again

  const summary = { test: 'phase6_recovery', llm_killed: llmPids, llm_turn_after_kill_ms: llmTurnMs, llm_reply: afterLlm.text,
    avatar_killed: avPids, avatar_recovered_ms: avatar.ms, reply_during_avatar_outage: duringOutage.text,
    reply_after_avatar: afterAvatar.text, speech_frames_after: speechAfter - speechBefore,
    llm_restarts: avatar.h.llm_restarts, avatar_restarts: avatar.h.avatar.restarts }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/phase6_recovery.json', JSON.stringify(summary, null, 2))
})
