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

// Our processes' GPU memory (PyTorch caching allocators of the backend and the avatar worker). The
// allocator reserves a block the first time an input is larger than before (a one-off step, measured
// +346 MiB once); a leak keeps growing. So the leak check is the growth over the second half of the run.
function ownTorch(samples) {
  const own = (s) => (s.backend_torch_mib ?? 0) + (s.avatar_torch_mib ?? 0)
  const at = (sec) => samples.find((s) => s.t >= sec) ?? samples.at(-1)
  const warm = at(180)
  const mid = at(samples.at(-1).t / 2)
  const last = samples.at(-1)
  return { at_3min: own(warm), at_half: own(mid), last: own(last), growth_after_3min: own(last) - own(warm),
    growth_second_half: own(last) - own(mid) }
}

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
      backend_torch_mib: gpu.backend_torch_reserved_mib ?? null, avatar_torch_mib: gpu.avatar_torch_reserved_mib ?? null,
      avatar_restarts: health.avatar?.restarts ?? null, avatar_speech_frames: health.avatar?.stats?.frames_in ?? null,
      avatar_gestures: health.avatar?.stats?.gestures ?? null, avatar_skipped: health.avatar?.stats?.skipped_frames ?? null,
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
    llm_restarts_during: samples.at(-1).llm_restarts - samples[0].llm_restarts,
    // our processes' GPU memory (torch caching allocators), from the 3-minute sample (warmed up) to the end
    own_torch_mib: ownTorch(samples),
    ram_available_mib: { first: samples[0].ram_available_mib, min: Math.min(...samples.map((s) => s.ram_available_mib)) },
    avatar: samples[0].avatar_restarts === null ? null : {
      restarts_during: samples.at(-1).avatar_restarts - samples[0].avatar_restarts,
      gestures: samples.at(-1).avatar_gestures, skipped_frames: samples.at(-1).avatar_skipped,
      ...(await page.evaluate(() => {
        const s = window.__avatarStats
        const off = [...s.avOffsets].sort((a, b) => a - b)
        const arr = [...s.recentArrivals].sort((a, b) => a - b)
        const dec = [...s.decodeMs].sort((a, b) => a - b)
        const q = (a, p) => (a.length ? Math.round(a[Math.min(a.length - 1, Math.floor(p * a.length))]) : null)
        return { speech_frames: s.speech, drawn_speech: s.drawnSpeech, late: s.late,
          av_offset_p50: off[Math.floor(off.length / 2)], av_offset_p90: off[Math.floor(off.length * 0.9)],
          arrived_late_100ms: s.arrivedLate, arrived_while_playing: s.arrivedTimed,
          arrival_lateness_p50: q(arr, 0.5), arrival_lateness_p90: q(arr, 0.9), decode_ms_p50: q(dec, 0.5), decode_ms_p90: q(dec, 0.9) }
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
  expect(summary.llm_restarts_during).toBe(0)                  // (counters may be > 0 from earlier tests)
  // No VRAM leak in OUR processes. The whole-GPU number also moves with every other desktop app (browsers,
  // screen recorders, ... measured: +/- 1 GiB between runs), so it is only checked against the spill limit.
  expect(summary.own_torch_mib.growth_second_half).toBeLessThan(150)
  expect(summary.vram_used_mib.max).toBeLessThan(12288 - 300)
  expect(summary.backend_rss_mib.last - summary.backend_rss_mib.first).toBeLessThan(800) // no big RAM leak
  if (summary.avatar) {
    expect(summary.avatar.restarts_during).toBe(0)
    expect(summary.avatar.drawn_speech).toBeGreaterThan(minutes * 100)               // lip-sync all along
    expect(summary.avatar.late / summary.avatar.drawn_speech).toBeLessThan(0.05)       // frames ready in time
    expect(summary.avatar.av_offset_p90).toBeLessThan(80)
  }
})
