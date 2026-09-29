import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import { captureServerMessages, openAndStartVoice } from './helpers.js'

// Long-running stability test. Run with:
//   $env:FAKE_MIC='mic_soak.wav'; $env:FAKE_MIC_LOOP='1'; $env:SOAK_MIN='30'; npx playwright test --project=voice soak
// The looping fake mic asks a question every ~19 s (longer answers get barged into).
// With an avatar profile (-Profile avatar / avatar_clone) it also checks the avatar worker and lip-sync timing.
const minutes = Number(process.env.SOAK_MIN ?? 30)
test.setTimeout((minutes + 15) * 60_000)
// A trace of a 30-minute session with a live avatar video stream is huge: stopping it took longer than
// the whole timeout budget (first avatar soak). The JSON summary below is the useful record.
test.use({ trace: 'off', screenshot: 'off' })

test(`voice soak for ${minutes} minutes`, async ({ page, request }) => {
  test.skip(process.env.FAKE_MIC !== 'mic_soak.wav', 'needs FAKE_MIC=mic_soak.wav FAKE_MIC_LOOP=1')
  const messages = captureServerMessages(page)
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e)))
  await openAndStartVoice(page)

  const samples = []
  const end = Date.now() + minutes * 60_000
  while (Date.now() < end) {
    const gpu = await (await request.get('/health/gpu')).json()
    const health = await (await request.get('/health')).json()
    samples.push({ t: Math.round((Date.now() - (end - minutes * 60_000)) / 1000), vram_used_mib: gpu.vram_used_mib,
      ram_available_mib: gpu.ram_available_mib, backend_rss_mib: gpu.backend_rss_mib, llm_restarts: health.llm_restarts,
      avatar_restarts: health.avatar?.restarts ?? null, avatar_speech_frames: health.avatar?.stats?.frames_in ?? null,
      conn: await page.getByTestId('conn-status').innerText() })
    await page.waitForTimeout(60_000)
  }

  const metrics = messages.filter((m) => m.type === 'metrics')
  const complete = metrics.filter((m) => !m.interrupted)
  const lat = complete.map((m) => m.ms_from_t0.client_first_audio).filter(Boolean).sort((a, b) => a - b)
  const pct = (p) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))]
  const serverErrors = messages.filter((m) => m.type === 'error')
  const summary = {
    minutes, turns: metrics.length, completed: complete.length, interrupted: metrics.length - complete.length,
    e2e_first_audio_ms: { p50: pct(0.5), p90: pct(0.9), max: lat.at(-1) },
    server_errors: serverErrors.length, page_errors: errors.length,
    vram_used_mib: { first: samples[0].vram_used_mib, last: samples.at(-1).vram_used_mib, max: Math.max(...samples.map((s) => s.vram_used_mib)) },
    backend_rss_mib: { first: samples[0].backend_rss_mib, last: samples.at(-1).backend_rss_mib, max: Math.max(...samples.map((s) => s.backend_rss_mib)) },
    llm_restarts: samples.at(-1).llm_restarts,
    ram_available_mib: { first: samples[0].ram_available_mib, min: Math.min(...samples.map((s) => s.ram_available_mib)) },
    avatar: samples[0].avatar_restarts === null ? null : {
      restarts: samples.at(-1).avatar_restarts,
      ...(await page.evaluate(() => {
        const s = window.__avatarStats
        const off = [...s.avOffsets].sort((a, b) => a - b)
        return { speech_frames: s.speech, drawn_speech: s.drawnSpeech, late: s.late,
          av_offset_p50: off[Math.floor(off.length / 2)], av_offset_p90: off[Math.floor(off.length * 0.9)] }
      })),
    },
  }
  console.log(JSON.stringify(summary))
  fs.mkdirSync('results', { recursive: true })
  fs.writeFileSync('results/soak.json', JSON.stringify({ summary, samples, metrics }, null, 2))

  expect(serverErrors).toEqual([])
  expect(errors).toEqual([])
  expect(complete.length).toBeGreaterThan(minutes * 2)          // at least ~2 completed turns per minute
  expect(samples.every((s) => s.conn === 'open')).toBe(true)
  expect(summary.llm_restarts).toBe(0)
  expect(summary.vram_used_mib.max - summary.vram_used_mib.first).toBeLessThan(600)   // no VRAM leak
  expect(summary.backend_rss_mib.last - summary.backend_rss_mib.first).toBeLessThan(800) // no big RAM leak
  if (summary.avatar) {
    expect(summary.avatar.restarts).toBe(0)
    expect(summary.avatar.drawn_speech).toBeGreaterThan(minutes * 100)               // lip-sync all along
    expect(summary.avatar.late / summary.avatar.drawn_speech).toBeLessThan(0.05)       // frames ready in time
    expect(summary.avatar.av_offset_p90).toBeLessThan(80)
  }
})
