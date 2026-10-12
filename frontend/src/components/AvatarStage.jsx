import { useEffect, useRef, useState } from 'react'
import { H264Stream } from '../lib/h264.js'
import { FLAG_KEY, FLAG_TAIL, KIND_IDLE_H264, KIND_IDLE_VIDEO, KIND_VIDEO, KIND_VIDEO_H264, packetKind,
  parsePacket } from '../lib/protocol.js'

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
// Playout delay at the start of replies (a jitter buffer). Over the internet the first lip-synced frames
// arrived after their audio: through the tunnel 205-465 ms late in 3 of 5 replies (2026-10-05), since the
// renderer only keeps pace with the audio at the start. Frames late in one reply hold the next reply's
// first audio that much longer (useVoice; the worker starts its frames as much later); a reply with every
// frame early lowers it again. On this PC it stays 0.
const MAX_START_HOLD_MS = 400
const REMOTE_START_HOLD_MS = 150
const LOCAL_PAGE = typeof location === 'undefined' || ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)
// A reply's frames are ImageBitmaps (~1 MB each at 512x512, 4 MB at 1024x1024): bounded, so frames of a
// reply whose audio never plays (or never ends) cannot pile up for the whole session.
const MAX_TURN_SECONDS = 20     // frames kept per reply, as seconds of video
const KEEP_TURN_FRAMES = 2      // replies whose frames are kept besides the one being heard (the newest)
const FORCED_HOLD_MS = (() => {         // ?hold=300: begin with this hold (tests, a known slow network)
  const v = typeof location === 'undefined' ? null : new URLSearchParams(location.search).get('hold')
  const ms = v === null ? NaN : Number(v)
  return Number.isFinite(ms) && ms >= 0 ? Math.min(MAX_START_HOLD_MS, ms) : null
})()

// worst: how far the latest-arriving frame of the reply came after its audio (ms; null = not measured)
function nextStartHold(hold, worst) {
  if (worst === null) return hold
  if (worst > 80) return Math.min(MAX_START_HOLD_MS, Math.round(hold + worst))
  if (worst < 0) return Math.max(0, hold - 50)
  return hold
}

export default function AvatarStage({ conv, voice }) {
  const canvasRef = useRef(null)
  const idleRef = useRef({ buf: [], pos: null, at: 0, staleBelow: -1, shown: null, staleRun: 0, restarted: false })
  const speechRef = useRef(new Map())        // turnId -> [{pts, idx, bmp}] sorted by pts
  const doneTurnsRef = useRef(new Set())     // turns whose frames (and closing tail) have been shown
  const tailShiftRef = useRef(new Map())     // turnId -> ms the closing tail is played later than its pts
  const fpsRef = useRef(20)
  const h264Ref = useRef({ idle: null, turns: new Map() })   // H.264 decoders (lib/h264.js)
  const jitterRef = useRef({ turn: 0, worst: null, done: 0, hold: 0 })   // start hold (nextStartHold)
  const rejectUpToRef = useRef(0)
  const statsRef = useRef({ idle: 0, speech: 0, drawnSpeech: 0, late: 0, avOffsets: [], arrivals: [], clock: null, startGaps: [],
    arrivedLate: 0, arrivedTimed: 0, recentArrivals: [], decodeMs: [], tails: 0, idleHolds: 0 })
  const [size, setSize] = useState(null)

  // only refs inside: the draw loop's copy from the first render stays valid
  const setStartHold = (ms) => {
    jitterRef.current.hold = ms
    statsRef.current.startHold = ms
    voice.startHoldRef.current = ms
    conv.connRef.current?.sendJson({ type: 'start_hold', ms })
  }
  const settleTurn = (turn) => {
    const j = jitterRef.current
    if (turn !== j.turn) return
    const next = nextStartHold(j.hold, j.worst)
    j.done = turn
    j.turn = 0
    statsRef.current.worstArrivals = [...(statsRef.current.worstArrivals ?? []), j.worst === null ? null : Math.round(j.worst)].slice(-50)
    if (next !== j.hold) setStartHold(next)
  }

  useEffect(() => {
    const offJson = conv.onJson((msg) => {
      // 'avatar': the avatar worker (re)connected during the session - its size, fps and frame numbers
      if ((msg.type === 'session' || msg.type === 'avatar') && msg.avatar?.size) setSize(msg.avatar.size)
      if ((msg.type === 'session' || msg.type === 'avatar') && msg.avatar?.fps) fpsRef.current = msg.avatar.fps
      if (msg.type === 'avatar') {
        resetIdle(idleRef.current)
        h264Ref.current.idle?.close()
        h264Ref.current.idle = null
      }
      if (msg.type === 'session') {                 // a new session: turn ids start again
        resetIdle(idleRef.current)                  // and the idle frame numbers may too (another worker)
        Object.assign(jitterRef.current, { turn: 0, worst: null, done: 0 })
        rejectUpToRef.current = 0
        doneTurnsRef.current.clear()
        tailShiftRef.current.clear()
        dropTurnsUpTo(speechRef.current, Infinity)
        for (const d of h264Ref.current.turns.values()) d.close()
        h264Ref.current.turns.clear()
        setStartHold(FORCED_HOLD_MS ?? (LOCAL_PAGE ? 0 : Math.max(REMOTE_START_HOLD_MS, jitterRef.current.hold)))
      }
      if (msg.type === 'interrupt') {
        rejectUpToRef.current = Math.max(rejectUpToRef.current, msg.turn_id)
        dropTurnsUpTo(speechRef.current, msg.turn_id)
        for (const [id, d] of h264Ref.current.turns) if (id <= msg.turn_id) { d.close(); h264Ref.current.turns.delete(id) }
      }
    })
    // decoded frames (JPEG or H.264) into the idle buffer or the reply's frame list
    const addIdle = (idx, bmp) => {
      statsRef.current.idle++
      const idle = idleRef.current
      if (idx < idle.staleBelow || (idle.pos !== null && idx < idle.pos - 2 * fpsRef.current)) {
        // Stale (from before the reply that just ended) - or a restarted avatar worker, which numbers its
        // frames from 0 again: every frame was dropped and the idle video froze until the next reply
        // ended. Stragglers are at most ~1 s of video (the server drops more, outbox.py); a longer run of
        // "stale" frames is a new numbering, and the idle video starts over with it.
        if (++idle.staleRun < 2 * fpsRef.current) { bmp.close(); return }
        resetIdle(idle)
        statsRef.current.idleRestarts = (statsRef.current.idleRestarts ?? 0) + 1
      }
      idle.staleRun = 0
      if (idle.buf.some((f) => f.idx === idx)) { bmp.close(); return }
      idle.buf.push({ idx, bmp })
      idle.buf.sort((a, b) => a.idx - b.idx)
      while (idle.buf.length > 40) releaseIdle(idle, idle.buf.shift())
    }
    const addSpeech = (p, bmp, decodeMs) => {
      if (p.turnId <= rejectUpToRef.current || doneTurnsRef.current.has(p.turnId)) { bmp.close(); return }
      statsRef.current.speech++
      {
        const eng = voice.engineRef.current
        const audibleNow = eng?.playingTurn === p.turnId ? eng.audibleMs(p.turnId) : null
        const st = statsRef.current
        if (st.arrivals.length < 600) st.arrivals.push(audibleNow === null ? null : Math.round(audibleNow - p.ptsMs))
        const j = jitterRef.current
        if (p.turnId > j.done && p.turnId !== j.turn) {
          if (j.turn && p.turnId > j.turn) settleTurn(j.turn)       // the reply before was interrupted
          if (p.turnId > j.turn) Object.assign(j, { turn: p.turnId, worst: null })
        }
        if (audibleNow !== null && p.turnId === j.turn) j.worst = Math.max(j.worst ?? -Infinity, audibleNow - p.ptsMs)
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
        if (list.length >= MAX_TURN_SECONDS * fpsRef.current) {    // enough waiting (its audio does not play)
          st.capped = (st.capped ?? 0) + 1
          bmp.close()
          return
        }
        list.push({ pts: p.ptsMs, idx: p.seq, tail: !!(p.flags & FLAG_TAIL), bmp })
        list.sort((a, b) => a.pts - b.pts)
        speechRef.current.set(p.turnId, list)
        dropOldTurns(speechRef.current, voice.engineRef.current?.playingTurn)
      }
    }
    // H.264: one decoder for the idle video, one per reply (each reply is its own stream)
    const h264 = h264Ref.current
    const onH264 = (p) => {
      const st = statsRef.current
      st.codec = 'h264'
      const key = !!(p.flags & FLAG_KEY)
      if (p.kind === KIND_IDLE_H264) {
        if (!h264.idle || h264.idle.broken) {
          h264.idle?.close()
          h264.idle = new H264Stream((m, bmp) => addIdle(m.idx, bmp), () => { st.h264Errors = (st.h264Errors ?? 0) + 1 },
            fpsRef.current)
        }
        if (!h264.idle.feed(p.payload, key, { idx: p.seq })) st.h264Waiting = (st.h264Waiting ?? 0) + 1
        return
      }
      if (p.turnId <= rejectUpToRef.current || doneTurnsRef.current.has(p.turnId)) return
      let dec = h264.turns.get(p.turnId)
      if (!dec || dec.broken) {
        dec?.close()
        for (const [id, d] of h264.turns) if (id < p.turnId) { d.close(); h264.turns.delete(id) }   // older replies
        dec = new H264Stream((m, bmp) => addSpeech(m, bmp, performance.now() - m.fed),
          () => { st.h264Errors = (st.h264Errors ?? 0) + 1 }, fpsRef.current)
        h264.turns.set(p.turnId, dec)
      }
      if (!dec.feed(p.payload, key, { turnId: p.turnId, ptsMs: p.ptsMs, seq: p.seq, flags: p.flags, fed: performance.now() })) {
        st.h264Waiting = (st.h264Waiting ?? 0) + 1
      }
    }
    const offBinary = conv.onBinary(async (buffer) => {
      const kind = packetKind(buffer)
      if (kind !== KIND_IDLE_H264 && kind !== KIND_VIDEO_H264 && kind !== KIND_IDLE_VIDEO && kind !== KIND_VIDEO) return
      const p = parsePacket(buffer)                   // (audio is not copied for nothing)
      if (p.kind === KIND_IDLE_H264 || p.kind === KIND_VIDEO_H264) { onH264(p); return }
      if (p.kind === KIND_VIDEO && (p.turnId <= rejectUpToRef.current || doneTurnsRef.current.has(p.turnId))) return
      statsRef.current.codec = 'jpeg'
      const tDecode = performance.now()
      const bmp = await createImageBitmap(new Blob([p.payload], { type: 'image/jpeg' }))
      const decodeMs = performance.now() - tDecode
      if (p.kind === KIND_IDLE_VIDEO) addIdle(p.seq, bmp)
      else addSpeech(p, bmp, decodeMs)
    })
    // shown again after being hidden: the server sent no video meanwhile and continues from keyframes
    // (session._set_visible) - the decoders start over and wait for them
    const onVisibility = () => {
      if (document.hidden) return
      h264.idle?.close()
      h264.idle = null
      for (const d of h264.turns.values()) d.close()
      h264.turns.clear()
    }
    document.addEventListener('visibilitychange', onVisibility)
    window.__avatarStats = statsRef.current // test hook
    return () => {
      document.removeEventListener('visibilitychange', onVisibility)
      offJson()
      offBinary()
      h264.idle?.close()
      h264.idle = null
      for (const d of h264.turns.values()) d.close()
      h264.turns.clear()
    }
  }, [conv.onJson, conv.onBinary, voice.engineRef])   // stable: conv and voice are new objects every render

  useEffect(() => {
    let raf = 0
    let lastSrc = null
    let lastSpeechIdx = null
    let fade = null                             // { snap: canvas, t0, dur }
    const draw = () => {
      raf = requestAnimationFrame(draw)
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      const now = performance.now()
      const st = statsRef.current
      const idle = idleRef.current
      if (idle.restarted) {                       // new idle numbering: the last reply's frame number means nothing
        idle.restarted = false
        lastSpeechIdx = null
      }
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
            if (firstTail) keepRecent(tailShiftRef.current.set(turn, Math.max(0, audible - firstTail.pts)))
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
              keepRecent(doneTurnsRef.current.add(turn))
              settleTurn(turn)
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
        if (lastSrc === 'idle' && src === 'speech' && lastSpeechIdx !== null && idle.shownIdx != null) {
          // how far the first lip-synced frame is from the idle frame on screen (loop frames): with the
          // livelier v3 head motion a few frames apart is a visible jump of the head
          const gap = Math.round((lastSpeechIdx - idle.shownIdx) * 10) / 10
          st.startGaps.push(gap)
          if (st.startGaps.length > 200) st.startGaps.shift()
          conv.connRef.current?.sendJson({ type: 'avatar_start', gap })   // the worker learns the offset
        }
        if (lastSrc && src !== lastSrc) {
          // cross-fade from what is on screen now (copied before a resize below clears the canvas)
          const snap = newCanvas(canvas.width, canvas.height)
          snap.getContext('2d').drawImage(canvas, 0, 0)
          fade = { snap, t0: now, dur: src === 'speech' ? FADE_TO_SPEECH_MS : FADE_TO_IDLE_MS }
        }
        // the canvas has the frame's own resolution (CSS scales it to the stage); anything of another
        // size (a blended frame, the fade snapshot) is scaled to it with the best filter
        if (canvas.width !== frame.width || canvas.height !== frame.height) {
          canvas.width = frame.width
          canvas.height = frame.height
        }
        const w = canvas.width
        const h = canvas.height
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'          // (a resize resets the context: set every frame)
        ctx.drawImage(frame, 0, 0, w, h)
        if (next && alpha > 0.02) {
          ctx.globalAlpha = alpha
          ctx.drawImage(next, 0, 0, w, h)
          ctx.globalAlpha = 1
        }
        if (fade) {
          const a = (now - fade.t0) / fade.dur
          if (a < 1) {
            ctx.globalAlpha = 1 - a
            ctx.drawImage(fade.snap, 0, 0, w, h)
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

// Idle playback starts over (a new session, or a restarted worker numbering its frames from 0): the
// buffer is dropped and nothing is stale any more; the frame on screen stays until the next one replaces it.
function resetIdle(idle) {
  for (const f of idle.buf) releaseIdle(idle, f)
  idle.buf = []
  idle.pos = null
  idle.staleBelow = -1
  idle.staleRun = 0
  idle.restarted = true
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
  if (!traceCanvas) traceCanvas = newCanvas(128, 128)
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

// Frames are kept for the reply being heard and the newest KEEP_TURN_FRAMES replies: a reply whose audio
// never played or never ended was otherwise freed only when the page closed.
function dropOldTurns(map, playing) {
  if (map.size <= KEEP_TURN_FRAMES) return
  const old = [...map.keys()].sort((a, b) => a - b).slice(0, -KEEP_TURN_FRAMES)
  for (const id of old) {
    if (id === playing) continue
    map.get(id).forEach((f) => f.bmp.close())
    map.delete(id)
  }
}

// per-turn sets and maps: the recent turns only (a long session added one entry per reply)
function keepRecent(coll) {
  if (coll.size > 32) coll.delete(coll.keys().next().value)
}

// OffscreenCanvas where there is one (not in Safari before 16.4: the fade into speech threw there)
function newCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  return c
}

// Camera grain: a real camera's picture is never perfectly clean and still - faint sensor noise changes
// every frame. Without it the background is a frozen, too-clean image, one of the cues that makes the
// avatar look synthetic (owner: "must look 100% realistic"). Painted into the avatar canvas at low
// opacity, a new noise pattern ~25 times a second; off with ?nograin. (A separate canvas with a CSS
// blend mode made the browser re-composite everything every frame: 6.9 % of the lip-synced frames
// were drawn late in a soak, 0.2 % without it.) The patterns are made once per tile (not every frame).
let grainTiles = null
let grainPatterns = []
let grainCtx = null
let grainOffset = [0, 0, -1]
let grainAt = 0
function drawGrain(ctx, w, h, now) {
  if (!grainTiles) grainTiles = Array.from({ length: 4 }, () => noiseTile(128))
  if (ctx !== grainCtx) {                       // patterns belong to a context (a new canvas: made again)
    grainCtx = ctx
    grainPatterns = []
  }
  if (now - grainAt > 40) {
    grainAt = now
    grainOffset = [Math.floor(Math.random() * 128), Math.floor(Math.random() * 128),
      Math.floor(Math.random() * grainTiles.length)]
  }
  const [dx, dy, i] = grainOffset
  if (i < 0) return
  grainPatterns[i] ??= ctx.createPattern(grainTiles[i], 'repeat')
  if (!grainPatterns[i]) return
  ctx.save()
  ctx.globalAlpha = 0.06
  ctx.globalCompositeOperation = 'overlay'
  ctx.fillStyle = grainPatterns[i]
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
