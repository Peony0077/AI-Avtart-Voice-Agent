import fs from 'node:fs'
import { expect, test } from '@playwright/test'
import { appPath, captureServerMessages, waitFor } from './helpers.js'

// The end of a spoken reply must flow into the idle video: no jump back to an old frame, no mouth
// snapping from the lip-synced face to the photo's. Found in user testing ("whenever she finishes
// her words, her face moves back to the original face"). Measured on the drawn canvas: share of
// pixels that change visibly from one drawn frame to the next (AvatarStage traceFrame).
const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b)
  return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : 0
}
const r4 = (x) => Math.round(x * 10000) / 10000

test('the end of a spoken reply flows into the idle video without a jump', async ({ page }) => {
  await page.addInitScript((shots) => {
    window.__avatarTrace = []
    if (shots) { window.__avatarShots = []; window.__avatarFilm = [] }
  }, !!process.env.SHOTS)
  const messages = captureServerMessages(page)
  await page.goto(appPath())
  await expect(page.getByTestId('conn-status')).toHaveText('open', { timeout: 30_000 })
  const session = await waitFor(messages, (m) => m.type === 'session', 10_000)
  test.skip(!session.avatar_available, 'backend not started with an avatar profile')
  await page.waitForTimeout(3000)                                   // idle video before the reply

  const rounds = []
  for (const question of ['Tell me one short fact about the moon.', 'Name two colours of the rainbow.']) {
    await expect(page.getByTestId('state')).toHaveText('Listening', { timeout: 60_000 })
    await page.getByTestId('input').fill(question)
    await page.getByTestId('send').click()
    const q = await waitFor(messages, (m) => m.type === 'user_message' && m.text === question, 10_000)
    await waitFor(messages, (m) => m.type === 'assistant_done' && m.turn_id === q.turn_id, 90_000)
    if (process.env.SHOTS && !rounds.length) {                       // filmstrip of the first reply's end
      await page.waitForTimeout(1200)
      const film = await page.evaluate(() => { window.__avatarFilmStop = true; return window.__avatarFilm })
      fs.mkdirSync('results/speech_end/film', { recursive: true })
      film.forEach((f, i) => fs.writeFileSync(`results/speech_end/film/${String(i).padStart(2, '0')}_${f.src}${f.ended ? '_ended' : ''}_${Math.round(f.changed * 1000)}.jpg`,
        Buffer.from(f.jpg.split(',')[1], 'base64')))
    }
    await page.waitForTimeout(3000)                                 // idle video after the reply
    const trace = await page.evaluate(() => window.__avatarTrace.map(({ t, src, turn, ended, changed }) => ({ t, src, turn, ended, changed })))
    const mine = trace.filter((e) => e.turn === q.turn_id)
    const speech = mine.filter((e) => e.src === 'speech' && !e.ended)
    const end = mine.find((e) => e.ended)
    expect(speech.length).toBeGreaterThan(20)                       // the avatar spoke
    expect(end).toBeTruthy()
    const tEnd = end.t
    const before = trace.filter((e) => e.src === 'idle' && e.t < speech[0].t - 500 && e.t > speech[0].t - 6000)
    const after = trace.filter((e) => e.t >= tEnd - 200 && e.t <= tEnd + 2500)
    // the hand-over: from the end of the audio, through the closing tail, to the first idle frames
    const handoffAt = (after.find((e) => e.src === 'idle' && e.t > tEnd) ?? { t: tEnd }).t
    const handoff = after.filter((e) => e.t <= handoffAt + 300)
    // the start: from the last idle frame into speech (cross-faded; the first speech frame follows the
    // idle frame on screen)
    const firstSpeech = trace.indexOf(speech[0])
    const start = trace.slice(firstSpeech, firstSpeech + 4)
    const jumps = after.filter((e) => e.changed > 0.02).map((e) => ({ ms: Math.round(e.t - tEnd), src: e.src, changed: r4(e.changed) }))
    rounds.push({
      turn: q.turn_id,
      idle_changed_p95: r4(pct(before.map((e) => e.changed), 0.95)),
      speech_changed_p95: r4(pct(speech.map((e) => e.changed), 0.95)),
      idle_changed_max: r4(Math.max(...before.map((e) => e.changed))),
      start_changed_max: r4(Math.max(...start.map((e) => e.changed))),
      handoff_ms: Math.round(handoffAt - tEnd),
      handoff_changed_max: r4(Math.max(...handoff.map((e) => e.changed))),
      end_changed_max: r4(Math.max(...after.map((e) => e.changed))),
      end_jumps: jumps,
      // longest time without a new frame after the audio ended (a hold that long looks frozen)
      end_longest_hold_ms: Math.round(Math.max(...after.slice(1).map((e, i) => e.t - after[i].t))),
    })
  }
  if (process.env.SHOTS) {
    const shots = await page.evaluate(() => window.__avatarShots)
    fs.mkdirSync('results/speech_end', { recursive: true })
    shots.forEach((s, i) => fs.writeFileSync(`results/speech_end/${String(i).padStart(2, '0')}_${s.src}_${s.before ? 'before' : 'jump'}_${Math.round(s.changed * 1000)}.jpg`,
      Buffer.from(s.jpg.split(',')[1], 'base64')))
  }
  console.log(JSON.stringify({ test: 'phase6_speech_end', rounds }))
  for (const r of rounds) {
    // the hand-over is no bigger than ordinary frame-to-frame motion (it was 7x bigger: 9.8 % of the
    // pixels changed at once); later idle frames move like idle frames (blinks are the biggest)
    expect(r.handoff_changed_max).toBeLessThanOrEqual(Math.max(0.02, 1.5 * r.speech_changed_p95, r.idle_changed_p95))
    expect(r.end_changed_max).toBeLessThanOrEqual(Math.max(0.035, 1.2 * r.idle_changed_max))
    expect(r.end_longest_hold_ms).toBeLessThan(400)
    expect(r.start_changed_max).toBeLessThanOrEqual(Math.max(0.03, 2 * r.speech_changed_p95))
  }
})
