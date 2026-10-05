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
const GRAIN = typeof location === 'undefined' || !new URLSearchParams(location.search).has('nograin')
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
  const statsRef = useRef({ idle: 0, speech: 0, drawnSpeech: 0, late: 0, avOffsets: [], arrivals: [], clock: null, startGaps: [],
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
  }, [conv.onJson, conv.onBinary, voice.engineRef])   // stable: conv and voice are new objects every render

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
          // A frame shows the middle of its 50 ms (its audio is centred there, avatar/musetalk/lipsync.py).
          // In-between frames blend from one middle to the next; blending from frame start to frame
          // start showed every mouth shape half a frame early. Free the frames we have passed.
          const at = SMOOTH ? audible - frameMs / 2 : audible
          let i = -1
          while (i + 1 < list.length && list[i + 1].pts <= at) i++
          if (i < 0 && list[0].pts <= audible) i = 0                // first half of the first frame
          if (i >= 0) {
            const f = list[i]
            const nx = SMOOTH ? list[i + 1] : null
            const blend = nx && nx.pts - f.pts <= 3 * frameMs ? Math.min(1, Math.max(0, (at - f.pts) / (nx.pts - f.pts))) : 0
            if (!ended) {
              // the moment of the audio that is on screen (0 = the picture is on time)
              const shown = f.pts + (blend ? blend * (nx.pts - f.pts) : 0) + frameMs / 2
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
              if (blend) {
                next = list[1].bmp
                alpha = blend
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
      // a closed ImageBitmap has no size: never draw one (the picture holds for a frame instead of the
      // draw throwing); counted, so a test notices if it happens again
      if (frame && !frame.width) { st.detached = (st.detached ?? 0) + 1; frame = null }
      if (next && !next.width) { st.detached = (st.detached ?? 0) + 1; next = null }
      if (frame) {
        if (canvas.width !== frame.width) { canvas.width = frame.width; canvas.height = frame.height }
        if (lastSrc === 'idle' && src === 'speech' && lastSpeechIdx !== null && idle.shownIdx != null) {
          // how far the first lip-synced frame is from the idle frame on screen (loop frames): with the
          // livelier v3 head motion a few frames apart is a visible jump of the head
          const gap = Math.round((lastSpeechIdx - idle.shownIdx) * 10) / 10
          st.startGaps.push(gap)
          conv.connRef.current?.sendJson({ type: 'avatar_start', gap })   // the worker learns the offset
        }
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
        if (GRAIN) drawGrain(ctx, canvas.width, canvas.height, now)
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
    // not [voice]: voice is a new object every render, and each restart of this loop forgot the picture on
    // screen (lastSrc) and skipped the cross-fade into speech - a hard cut when she started talking
  }, [voice.engineRef])

  const available = conv.session?.avatar_available
  const canTalk = conv.status === 'open' && conv.session?.voice_available !== false && !conv.waiting
  return (
    <section className="stage" aria-label="Avatar">
      {available ? (
        <canvas ref={canvasRef} data-testid="avatar-canvas" width={size?.[0] ?? 512} height={size?.[1] ?? 512} />
      ) : conv.session ? (
        <div className="stage-placeholder">Avatar disabled (start with profile “avatar”)</div>
      ) : null}
      {conv.waiting && (
        <div className="stage-waiting" role="status" data-testid="waiting">
          <strong>{conv.session?.persona ?? 'She'} is talking with another customer right now.</strong>
          <span>You're number {conv.waiting} in line. This page connects you automatically - please stay.</span>
        </div>
      )}
      {conv.status === 'ended (inactive)' && (
        <div className="stage-waiting" role="status" data-testid="ended-inactive">
          <strong>The conversation ended after a while without activity.</strong>
          <button type="button" onClick={() => location.reload()}>Talk again</button>
        </div>
      )}
      {canTalk && (
        <div className="stage-voice">
          {voice.micHint && (
            <p className="stage-hint" data-testid="mic-hint">
              Your Bluetooth headset's microphone makes voices sound like a phone call.{' '}
              <button type="button" onClick={() => voice.useMic(voice.micHint.deviceId)}>Use {voice.micHint.label}</button>{' '}
              <button type="button" className="secondary" onClick={voice.dismissMicHint}>Keep</button>
            </p>
          )}
          {voice.error && <p className="stage-error" role="alert" data-testid="stage-error">{voice.error}</p>}
          {voice.active ? (
            <span className={`listening ${voice.speechDetected ? 'hot' : ''}`} data-testid="listening">
              <span className="dot" />
              {voice.speechDetected ? 'Hearing you…' : 'Listening – just speak'}
            </span>
          ) : (
            // the microphone only starts on a tap (browsers require it): make that impossible to miss
            <button type="button" className="talk" data-testid="talk" onClick={voice.start} disabled={voice.starting}>
              {voice.starting ? 'Allow the microphone…' : voice.error ? 'Try again' : `🎤 Tap to talk to ${conv.session?.persona ?? 'her'}`}
            </button>
          )}
        </div>
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
  // stale: never shown again - but if it is still queued (a reply that started a little behind the idle
  // on screen and was cut short) it is released with the queue, or it would be drawn after close()
  // ("drawImage: the image source is detached", 7x in a 10-minute soak)
  if (idle.shown && !idle.buf.some((f) => f.bmp === idle.shown)) idle.shown.close()
  idle.shown = null
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
  idle.shownIdx = idle.pos           // the loop position on screen (speech joins it: startGaps)
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

// Camera grain: a real camera's picture is never perfectly clean and still - faint sensor noise changes
// every frame. Without it the background is a frozen, too-clean image, one of the cues that makes the
// avatar look synthetic (owner: "must look 100% realistic"). Painted into the avatar canvas at low
// opacity, a new noise pattern ~25 times a second; off with ?nograin. (A separate canvas with a CSS
// blend mode made the browser re-composite everything every frame: 6.9 % of the lip-synced frames
// were drawn late in a soak, 0.2 % without it.)
let grainTiles = null
let grainOffset = [0, 0]
let grainAt = 0
function drawGrain(ctx, w, h, now) {
  if (!grainTiles) grainTiles = Array.from({ length: 4 }, () => noiseTile(128))
  if (now - grainAt > 40) {
    grainAt = now
    grainOffset = [Math.floor(Math.random() * 128), Math.floor(Math.random() * 128),
      grainTiles[Math.floor(Math.random() * grainTiles.length)]]
  }
  const [dx, dy, tile] = grainOffset
  if (!tile) return
  ctx.save()
  ctx.globalAlpha = 0.06
  ctx.globalCompositeOperation = 'overlay'
  ctx.fillStyle = ctx.createPattern(tile, 'repeat')
  ctx.translate(-dx, -dy)
  ctx.fillRect(0, 0, w + dx, h + dy)
  ctx.restore()
}

// Grey sensor-like noise (luminance only, roughly Gaussian) on a mid-grey base, for the overlay blend.
function noiseTile(size) {
  const c = document.createElement('canvas')
  c.width = c.height = size
  const ctx = c.getContext('2d')
  const img = ctx.createImageData(size, size)
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 128 + ((Math.random() + Math.random() + Math.random() - 1.5) * 70)
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v
    img.data[i + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return c
}
