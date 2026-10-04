import { useEffect, useRef, useState } from 'react'
import { FLAG_TAIL, KIND_IDLE_VIDEO, KIND_VIDEO, parsePacket } from '../lib/protocol.js'

// Renders the avatar video.
// - Speech frames are shown by the audio clock: the frame whose pts (ms on the turn's audio timeline)
//   matches what is audible right now, so the lips stay in sync regardless of network jitter. After the
//   last word the clock keeps running and the reply's closing tail is drawn (the mouth closes and the
//   lip-synced face fades into the avatar's own face, see the worker's _render_tail).
// - Idle frames are buffered and played at the avatar's frame rate by their frame number (the worker
//   numbers idle and speech frames alike), ~2-4 frames behind the newest: arriving over a network they
//   come unevenly, and showing each on arrival made the idle motion jerky.
// - After a reply the idle video continues from the frame after the reply's last one. It used to
//   restart from the last idle frame received *before* the reply - an old head pose and the photo's
//   mouth, seen in user testing as "her face moves back to the original face".
// - Switching between speech and idle frames cross-fades briefly (snapshot of the canvas).
// - In-between frames: the video has 20 frames per second, the screen ~60; instead of holding each
//   frame for three screen refreshes, the next frame is blended in as its time approaches, so motion
//   moves smoothly instead of in steps (owner: "needs more smooth"; off with ?nosmooth).
const SMOOTH = typeof location === 'undefined' || !new URLSearchParams(location.search).has('nosmooth')
const FADE_TO_SPEECH_MS = 120
const FADE_TO_IDLE_MS = 200
const TAIL_GRACE_MS = 800        // after the audio ends, wait (holding the last frame) this long for the closing tail

export default function AvatarStage({ conv, voice }) {
  const canvasRef = useRef(null)
  const idleRef = useRef({ buf: [], pos: null, at: 0, staleBelow: -1, shown: null })
  const speechRef = useRef(new Map())        // turnId -> [{pts, idx, bmp}] sorted by pts
  const doneTurnsRef = useRef(new Set())     // turns whose frames (and closing tail) have been shown
  const tailShiftRef = useRef(new Map())     // turnId -> ms the closing tail is played later than its pts
  const fpsRef = useRef(20)
  const rejectUpToRef = useRef(0)
  const statsRef = useRef({ idle: 0, speech: 0, drawnSpeech: 0, late: 0, avOffsets: [], arrivals: [], clock: null,
    arrivedLate: 0, arrivedTimed: 0, recentArrivals: [], decodeMs: [], tails: 0, idleHolds: 0 })
  const [size, setSize] = useState(null)

  useEffect(() => {
    const offJson = conv.onJson((msg) => {
      if (msg.type === 'session' && msg.avatar?.size) setSize(msg.avatar.size)
      if (msg.type === 'session' && msg.avatar?.fps) fpsRef.current = msg.avatar.fps
      if (msg.type === 'interrupt') {
        rejectUpToRef.current = Math.max(rejectUpToRef.current, msg.turn_id)
        dropTurnsUpTo(speechRef.current, msg.turn_id)
      }
    })
    const offBinary = conv.onBinary(async (buffer) => {
      const p = parsePacket(buffer)
      if (p.kind !== KIND_IDLE_VIDEO && p.kind !== KIND_VIDEO) return
      if (p.kind === KIND_VIDEO && (p.turnId <= rejectUpToRef.current || doneTurnsRef.current.has(p.turnId))) return
      const tDecode = performance.now()
      const bmp = await createImageBitmap(new Blob([p.payload], { type: 'image/jpeg' }))
      const decodeMs = performance.now() - tDecode
      if (p.kind === KIND_IDLE_VIDEO) {
        statsRef.current.idle++
        const idle = idleRef.current
        if (p.seq < idle.staleBelow || idle.buf.some((f) => f.idx === p.seq)) { bmp.close(); return }
        idle.buf.push({ idx: p.seq, bmp })
        idle.buf.sort((a, b) => a.idx - b.idx)
        while (idle.buf.length > 40) releaseIdle(idle, idle.buf.shift())
      } else {
        statsRef.current.speech++
        const eng = voice.engineRef.current
        const audibleNow = eng?.playingTurn === p.turnId ? eng.audibleMs(p.turnId) : null
        const st = statsRef.current
        if (st.arrivals.length < 600) st.arrivals.push(audibleNow === null ? null : Math.round(audibleNow - p.ptsMs))
        // whole-session diagnostics (test hooks): arrival lateness after decoding, and decode time
        if (audibleNow !== null) {
          st.arrivedTimed++
          if (audibleNow - p.ptsMs > 100) st.arrivedLate++
          st.recentArrivals.push(Math.round(audibleNow - p.ptsMs))
          if (st.recentArrivals.length > 2000) st.recentArrivals.shift()
        }
        st.decodeMs.push(decodeMs)
        if (st.decodeMs.length > 2000) st.decodeMs.shift()
        const list = speechRef.current.get(p.turnId) ?? []
        list.push({ pts: p.ptsMs, idx: p.seq, tail: !!(p.flags & FLAG_TAIL), bmp })
        list.sort((a, b) => a.pts - b.pts)
        speechRef.current.set(p.turnId, list)
      }
    })
    window.__avatarStats = statsRef.current // test hook
    return () => { offJson(); offBinary() }
  }, [conv, voice])

  useEffect(() => {
    let raf = 0
    let lastSrc = null
    let lastSpeechIdx = null
    let fade = null                             // { snap: OffscreenCanvas, t0, dur }
    const draw = () => {
      raf = requestAnimationFrame(draw)
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      const now = performance.now()
      const st = statsRef.current
      const idle = idleRef.current
      const frameMs = 1000 / fpsRef.current
      let frame = null
      let next = null                           // the following frame, blended in by `alpha`
      let alpha = 0
      const engine = voice.engineRef.current
      const turn = engine?.playingTurn
      const list = turn && !doneTurnsRef.current.has(turn) ? speechRef.current.get(turn) : null
      const ended = !!(turn && engine?.endedTurns.has(turn))
      if (engine && turn && list?.length) {
        let audible = engine.audibleMs(turn)
        if (ended && audible !== null) {
          // a closing tail that arrives after the audio ended plays from its start (its first frames are
          // the mouth closing), held on the last spoken frame until then
          if (!tailShiftRef.current.has(turn)) {
            const firstTail = list.find((x) => x.tail)
            if (firstTail) tailShiftRef.current.set(turn, Math.max(0, audible - firstTail.pts))
          }
          audible -= tailShiftRef.current.get(turn) ?? 0
        }
        if (!st.clock && engine.playCtx) {
          const ts = engine.playCtx.getOutputTimestamp?.()
          st.clock = { outputLatency: engine.playCtx.outputLatency, baseLatency: engine.playCtx.baseLatency,
            currentTime: engine.playCtx.currentTime, tsContext: ts?.contextTime, audible, start: engine.turnStartCtx.get(turn) }
        }
        if (audible !== null && audible >= 0) {
          // newest frame not in the future; free the ones we have passed
          let i = -1
          while (i + 1 < list.length && list[i + 1].pts <= audible) i++
          if (i >= 0) {
            const f = list[i]
            if (!ended) {
              // what is on screen: with in-between frames, the picture between this frame and the next
              const nx = SMOOTH ? list[i + 1] : null
              const shown = nx && nx.pts - f.pts <= 3 * frameMs
                ? f.pts + Math.min(1, Math.max(0, (audible - f.pts) / (nx.pts - f.pts))) * (nx.pts - f.pts)
                : f.pts
              st.drawnSpeech++
              st.avOffsets.push(audible - shown)
              if (st.avOffsets.length > 400) st.avOffsets.shift()
              if (audible - shown > 120) st.late++
            }
            for (const old of list.splice(0, i)) old.bmp.close()
            // the last frame shown: done once its time is over (waiting a little longer if it is not
            // yet the closing tail, which comes after the end of the audio)
            if (ended && list.length === 1 && audible > f.pts + frameMs + (f.tail ? 0 : TAIL_GRACE_MS)) {
              // the reply and its closing tail are over: the idle video continues on the next frame
              doneTurnsRef.current.add(turn)
              st.tails++
              continueIdleAfter(idle, f.idx, now)
              lastSpeechIdx = f.idx
            } else {
              frame = f.bmp
              lastSpeechIdx = f.idx
              const nx = list[1]
              if (SMOOTH && nx && nx.pts - f.pts <= 3 * frameMs) {
                next = nx.bmp
                alpha = Math.min(1, Math.max(0, (audible - f.pts) / (nx.pts - f.pts)))
              }
            }
          }
        }
      }
      let src = frame ? 'speech' : 'idle'
      if (!frame) {
        if (lastSrc === 'speech' && lastSpeechIdx !== null && idle.staleBelow <= lastSpeechIdx) {
          continueIdleAfter(idle, lastSpeechIdx, now)       // interrupted: no stale idle frames either
        }
        frame = nextIdleFrame(idle, now, fpsRef.current, st)
        if (SMOOTH && frame && idle.next) { next = idle.next; alpha = idle.alpha }
        if (!frame && lastSrc === 'speech') { src = 'speech'; frame = null }   // hold the last speech frame
      }
      if (frame) {
        if (canvas.width !== frame.width) { canvas.width = frame.width; canvas.height = frame.height }
        if (lastSrc && src !== lastSrc) {
          // cross-fade from what is on screen now
          const snap = new OffscreenCanvas(canvas.width, canvas.height)
          snap.getContext('2d').drawImage(canvas, 0, 0)
          fade = { snap, t0: now, dur: src === 'speech' ? FADE_TO_SPEECH_MS : FADE_TO_IDLE_MS }
        }
        ctx.drawImage(frame, 0, 0)
        if (next && alpha > 0.02) {
          ctx.globalAlpha = alpha
          ctx.drawImage(next, 0, 0)
          ctx.globalAlpha = 1
        }
        if (fade) {
          const a = (now - fade.t0) / fade.dur
          if (a < 1) {
            ctx.globalAlpha = 1 - a
            ctx.drawImage(fade.snap, 0, 0)
            ctx.globalAlpha = 1
          } else {
            fade = null
          }
        }
        lastSrc = src
        if (window.__avatarTrace) traceFrame(window.__avatarTrace, canvas, src, turn ?? 0, ended)
      }
      // forget finished turns
      for (const [id, l] of speechRef.current) {
        if (doneTurnsRef.current.has(id) || (engine?.endedTurns.has(id) && id !== turn)) {
          l.forEach((f) => f.bmp.close())
          speechRef.current.delete(id)
        }
      }
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [voice])

  const available = conv.session?.avatar_available
  return (
    <section className="stage" aria-label="Avatar">
      {available ? (
        <canvas ref={canvasRef} data-testid="avatar-canvas" width={size?.[0] ?? 512} height={size?.[1] ?? 512} />
      ) : (
        <div className="stage-placeholder">Avatar disabled (start with profile “avatar”)</div>
      )}
    </section>
  )
}

// After speech: idle playback resumes at the frame after `idx`; older idle frames are stale.
function continueIdleAfter(idle, idx, now) {
  idle.staleBelow = idx + 1
  idle.pos = idx + 1
  idle.at = now
  while (idle.buf.length && idle.buf[0].idx < idle.staleBelow) releaseIdle(idle, idle.buf.shift())
  if (idle.shown) { idle.shown.close(); idle.shown = null }   // stale: never shown again
}

// Close an idle frame unless it is the one on screen (that one is closed when it is replaced).
function releaseIdle(idle, f) {
  if (f.bmp !== idle.shown) f.bmp.close()
}

// The idle frame to show now. The playback position advances at the frame rate, a little faster
// when many frames are buffered and not at all when none is, so it stays ~2-4 frames behind the
// newest frame (a jitter buffer). Returns null when there is nothing to show yet.
function nextIdleFrame(idle, now, fps, st) {
  const buf = idle.buf
  idle.next = null
  if (!buf.length) return idle.shown
  if (idle.pos === null) { idle.pos = buf[0].idx; idle.at = now }
  const newest = buf[buf.length - 1].idx
  const ahead = newest - idle.pos
  // gentle: a faster rate shows as skipped frames (measured: 1.3x made blinks look like jumps)
  const rate = ahead <= 0 ? 0 : ahead < 2 ? 0.95 : ahead <= 5 ? 1 : ahead <= 12 ? 1.1 : 1.2
  idle.pos += ((now - idle.at) * fps / 1000) * rate
  idle.at = now
  if (buf[0].idx > idle.pos + 10 || idle.pos > newest + 10) idle.pos = buf[0].idx   // the worker jumped: resync
  let i = -1
  while (i + 1 < buf.length && buf[i + 1].idx <= idle.pos) i++
  if (i < 0) { st.idleHolds++; return idle.shown }
  for (const old of buf.splice(0, i)) releaseIdle(idle, old)
  if (idle.shown && idle.shown !== buf[0].bmp) idle.shown.close()
  idle.shown = buf[0].bmp
  if (buf.length > 1 && buf[1].idx - buf[0].idx <= 3) {      // in-between position towards the next frame
    idle.next = buf[1].bmp
    idle.alpha = Math.min(1, Math.max(0, (idle.pos - buf[0].idx) / (buf[1].idx - buf[0].idx)))
  }
  return idle.shown
}

// Test hook (window.__avatarTrace = [] before the page loads): for every drawn frame, the share of
// pixels that changed visibly since the previous drawn frame, on a 128x128 grayscale copy. A jump
// (stale frame, pose jump, mouth snapping back) shows as a spike well above normal frame-to-frame motion.
let traceCanvas = null
let tracePrev = null
function traceFrame(trace, canvas, src, turn, ended) {
  if (!traceCanvas) traceCanvas = new OffscreenCanvas(128, 128)
  const tctx = traceCanvas.getContext('2d', { willReadFrequently: true })
  tctx.drawImage(canvas, 0, 0, 128, 128)
  const px = tctx.getImageData(0, 0, 128, 128).data
  const gray = new Uint8Array(128 * 128)
  for (let i = 0; i < gray.length; i++) gray[i] = (px[4 * i] * 77 + px[4 * i + 1] * 150 + px[4 * i + 2] * 29) >> 8
  let changed = 0
  let sum = 0
  if (tracePrev) {
    for (let i = 0; i < gray.length; i++) {
      const d = Math.abs(gray[i] - tracePrev[i])
      sum += d
      if (d > 16) changed++
    }
  }
  const same = tracePrev && sum === 0
  tracePrev = gray
  if (same) return                       // the same frame drawn again (screen refresh faster than video)
  const entry = { t: performance.now(), src, turn, ended, changed: changed / gray.length, mean: sum / gray.length }
  trace.push(entry)
  if (trace.length > 20000) trace.splice(0, 5000)
  // snapshots of the frames around big jumps (window.__avatarShots = []), to look at them
  const shots = window.__avatarShots
  if (shots && entry.changed > 0.02 && shots.length < 40) {
    if (trace.prevShot) shots.push({ ...trace.prevShot, before: true })
    shots.push({ t: entry.t, src, changed: entry.changed, jpg: canvas.toDataURL('image/jpeg', 0.85) })
  }
  if (shots) trace.prevShot = { t: entry.t, src, changed: entry.changed, jpg: canvas.toDataURL('image/jpeg', 0.85) }
  // filmstrip of the last 60 drawn frames (window.__avatarFilm = [], frozen by window.__avatarFilmStop)
  const film = window.__avatarFilm
  if (film && !window.__avatarFilmStop) {
    film.push({ t: entry.t, src, ended, changed: entry.changed, jpg: canvas.toDataURL('image/jpeg', 0.9) })
    if (film.length > 60) film.shift()
  }
}

function dropTurnsUpTo(map, turnId) {
  for (const [id, list] of map) {
    if (id <= turnId) { list.forEach((f) => f.bmp.close()); map.delete(id) }
  }
}
