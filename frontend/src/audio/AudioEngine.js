// Microphone capture + low-latency assistant playback (both via AudioWorklet).
import micWorkletUrl from './mic-worklet.js?url'
import playerWorkletUrl from './player-worklet.js?url'
import { MIC_SAMPLE_RATE, TTS_SAMPLE_RATE } from '../lib/protocol.js'
import { preferredMic } from './microphones.js'

export function pcm16ToFloat(pcm16Buffer) {
  const pcm = new Int16Array(pcm16Buffer)
  const f32 = new Float32Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768
  return f32
}

export class AudioEngine extends EventTarget {
  constructor() {
    super()
    this.micCtx = null
    this.playCtx = null
    this.player = null
    this.stream = null
    this.level = 0
    this.playedMs = 0
    this.playingTurn = 0
    this.turnStartCtx = new Map()
    this.turnGaps = new Map()       // turnId -> { done: s, spans: [[fromCtx, toCtx]] } (the player ran dry)
    this.endedTurns = new Set()
    this._gen = 0                   // flush generation: audio queued before a flush is dropped
  }

  // Must be called from a user gesture (click) so the AudioContexts may start.
  async start(onMicFrame) {
    await this.startPlayback()
    await this.startMic(onMicFrame)
  }

  // Assistant audio only (typed messages are spoken too). Call from a user gesture: the
  // AudioContext is created synchronously, before the first await, so the browser lets it run.
  // A start that failed (the worklet did not load) is undone completely, so the next call tries again:
  // kept half-made, every later reply was silent until the page was reloaded.
  async startPlayback() {
    if (this.playCtx) {
      // suspended meanwhile (phone locked, a call): this runs inside a click, so it may run again
      if (this.playCtx.state !== 'running' && this.playCtx.state !== 'closed') this.playCtx.resume().catch(() => {})
      return this._playbackReady
    }
    const ctx = new AudioContext({ sampleRate: TTS_SAMPLE_RATE, latencyHint: 'interactive' })
    this.playCtx = ctx
    const ready = (async () => {
      try {
        await ctx.audioWorklet.addModule(playerWorkletUrl)
        if (this.playCtx !== ctx) return                  // stopped meanwhile
        const player = new AudioWorkletNode(ctx, 'pcm-player', { outputChannelCount: [1] })
        player.connect(ctx.destination)
        player.port.onmessage = (e) => this._onPlayer(e.data)
        this.player = player
        await ctx.resume()
      } catch (err) {
        if (this.playCtx === ctx) {
          this.playCtx = this.player = this._playbackReady = null
          this._gen++                                      // what waited for it is dropped
        }
        ctx.close().catch(() => {})
        throw err
      }
    })()
    if (this.playCtx === ctx) this._playbackReady = ready   // (not when it already failed: no audioWorklet)
    await ready
  }

  // Run fn once playback has started (the first typed message: the worklet is still loading), unless a
  // flush comes first - queued on the promise, Stop or Mute could not remove it and the interrupted
  // audio still played.
  _whenReady(fn) {
    const gen = this._gen
    this._playbackReady?.then(() => { if (gen === this._gen) fn() }, () => {})
  }

  get micOn() {
    return Boolean(this.stream)
  }

  // Call from the click, before any await (like startPlayback): the microphone's AudioContext is created
  // and resumed inside the user gesture, then the permission prompt opens. Created after the prompt (as
  // before), Safari leaves the context suspended - the mic never sends a sound - and an unanswered
  // prompt or a blocked microphone left "Start voice" doing nothing visible. Errors are MicError with a
  // message for the user (micErrorMessage).
  async startMic(onMicFrame, preCtx = null) {
    if (this.stream) { preCtx?.close().catch(() => {}); return }
    if (!navigator.mediaDevices?.getUserMedia) {
      preCtx?.close().catch(() => {})
      throw new MicError(window.isSecureContext ? 'NoMediaDevices' : 'NotSecure')
    }
    let ctx = preCtx ?? newMicContext()
    ctx.resume().catch(() => {})                    // inside the click
    try {
      try {
        const deviceId = preferredMic()
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1,
            ...(deviceId ? { deviceId: { ideal: deviceId } } : {}) },     // the one chosen in this browser, if still there
        })
      } catch (err) {
        throw new MicError(err?.name || 'Error', err?.message)
      }
      // 16 kHz where the browser allows it; otherwise its own rate and the worklet resamples to 16 kHz
      // (Firefox throws for a 16 kHz context with a 48 kHz microphone; some Safari versions ignore the rate).
      let source
      try {
        source = ctx.createMediaStreamSource(this.stream)
      } catch {
        await ctx.close().catch(() => {})
        ctx = new AudioContext({ latencyHint: 'interactive' })
        source = ctx.createMediaStreamSource(this.stream)
      }
      this.micCtx = ctx
      await ctx.audioWorklet.addModule(micWorkletUrl)
      const capture = new AudioWorkletNode(ctx, 'mic-capture')
      const track = this.stream.getAudioTracks()[0]
      const s = track?.getSettings?.() ?? {}
      // reported to the server log: what the browser really does with the microphone
      this.micInfo = { contextRate: ctx.sampleRate, deviceRate: s.sampleRate ?? null, channels: s.channelCount ?? null,
        echoCancellation: s.echoCancellation ?? null, noiseSuppression: s.noiseSuppression ?? null,
        autoGainControl: s.autoGainControl ?? null, device: track?.label ?? '', playbackRate: this.playCtx?.sampleRate ?? null,
        userAgent: navigator.userAgent }
      this.micPeak = 0
      capture.port.onmessage = (e) => {
        this.level = e.data.level
        this.micPeak = Math.max(this.micPeak, e.data.level)
        onMicFrame(e.data.pcm)
      }
      source.connect(capture)
      await Promise.race([ctx.resume(), new Promise((r) => setTimeout(r, 3000))])
      if (ctx.state !== 'running') throw new MicError('Suspended', `audio context ${ctx.state}`)
    } catch (err) {
      this.stream?.getTracks().forEach((t) => t.stop())
      await ctx.close().catch(() => {})
      this.micCtx = this.stream = null
      throw err instanceof MicError ? err : new MicError(err?.name || 'Error', err?.message)
    }
  }

  async stopMic() {
    const ctx = this.micCtx
    this.stream?.getTracks().forEach((t) => t.stop())
    this.micCtx = this.stream = null
    this.level = 0
    await ctx?.close().catch(() => {})
  }

  // Another microphone: its AudioContext is created here, before the first await (inside the click) -
  // created after stopping the old one, Safari left it suspended ("The browser did not start the audio").
  async restartMic(onMicFrame) {
    const ctx = newMicContext()
    ctx.resume().catch(() => {})
    await this.stopMic()
    await this.startMic(onMicFrame, ctx)
  }

  // Everything off (the page goes away): microphone, both AudioContexts.
  async stop() {
    const ctx = this.playCtx
    this.playCtx = this.player = this._playbackReady = null
    this._gen++
    await this.stopMic()
    await ctx?.close().catch(() => {})
  }

  // PCM16 mono 24 kHz from the server.
  push(turnId, pcm16Buffer) {
    this.pushFloat(turnId, pcm16ToFloat(pcm16Buffer))
  }

  // Samples at 24 kHz (PCM converted, or the decoded Opus voice: lib/opus.js); the array is transferred.
  pushFloat(turnId, samples) {
    if (!this.player) {
      // playback still starting (first typed message): queue until the worklet exists
      this._whenReady(() => this.pushFloat(turnId, samples))
      return
    }
    let f32 = samples
    const rate = this.playCtx.sampleRate
    if (rate !== TTS_SAMPLE_RATE) {
      // the browser ignored the requested 24 kHz: resample, or the voice plays too fast and too high
      const out = new Float32Array(Math.round((f32.length * rate) / TTS_SAMPLE_RATE))
      const step = TTS_SAMPLE_RATE / rate
      for (let i = 0; i < out.length; i++) {
        const x = i * step
        const k = Math.floor(x)
        const a = f32[k] ?? 0
        out[i] = a + ((f32[k + 1] ?? a) - a) * (x - k)
      }
      f32 = out
    }
    this.player.port.postMessage({ type: 'push', turnId, samples: f32 }, [f32.buffer])
  }

  endTurn(turnId) {
    if (!this.player) {
      this._whenReady(() => this.endTurn(turnId))
      return
    }
    this.player.port.postMessage({ type: 'end', turnId })
  }

  // A new session (a reconnect): the server numbers its turns from 1 again, so the per-turn
  // bookkeeping starts over - with the old ids kept, the new replies' audio was taken for old turns.
  resetTurns() {
    this._gen++
    this.player?.port.postMessage({ type: 'reset' })
    this.turnStartCtx.clear()
    this.turnGaps.clear()
    this.endedTurns.clear()
    this.playingTurn = 0
    this.playedMs = 0
  }

  flush() {
    this._gen++
    this.player?.port.postMessage({ type: 'flush' })
    if (this.playingTurn) remember(this.endedTurns, this.playingTurn)
    this.playingTurn = 0
    this.playedMs = 0
  }

  // Position (ms) on a turn's audio timeline that is audible right now, or null: the time since its first
  // sample, less the gaps the player ran dry (worklet 'dry'/'resumed'). Frame pts and the server's
  // played_ms are on that timeline - with wall time only, a 300 ms underrun left the lips 300 ms ahead
  // of the voice for the rest of the reply.
  audibleMs(turnId) {
    const start = this.turnStartCtx.get(turnId)
    if (start === undefined || !this.playCtx) return null
    const ts = this.playCtx.getOutputTimestamp?.()
    const nowCtx = ts && ts.contextTime > 0
      ? ts.contextTime + (performance.now() - ts.performanceTime) / 1000
      : this.playCtx.currentTime - (this.playCtx.outputLatency || 0)
    let silent = 0
    const g = this.turnGaps.get(turnId)
    if (g) {
      // a gap is subtracted only as far as it is audible yet (the worklet runs ahead by the output latency)
      while (g.spans.length && g.spans[0][1] <= nowCtx) {
        const [from, to] = g.spans.shift()
        g.done += to - from
      }
      silent = g.done
      for (const [from, to] of g.spans) if (nowCtx > from) silent += Math.min(nowCtx, to) - from
    }
    return (nowCtx - start - silent) * 1000
  }

  _onPlayer(msg) {
    if (msg.type === 'started') {
      this.playingTurn = msg.turnId
      remember(this.turnStartCtx, msg.turnId, msg.ctxTime)
      remember(this.turnGaps, msg.turnId, { done: 0, spans: [] })
    }
    if (msg.type === 'dry') this.turnGaps.get(msg.turnId)?.spans.push([msg.ctxTime, Infinity])
    if (msg.type === 'resumed') {
      const last = this.turnGaps.get(msg.turnId)?.spans.at(-1)
      if (last && last[1] === Infinity) last[1] = Math.max(last[0], msg.ctxTime)
    }
    if (msg.type === 'ended' || msg.type === 'flushed') {
      const id = msg.turnId ?? this.playingTurn
      if (id) remember(this.endedTurns, id)
    }
    if (msg.playedMs !== undefined) this.playedMs = msg.playedMs
    this.dispatchEvent(new CustomEvent('player', { detail: msg }))
  }
}

// Per-turn bookkeeping keeps the recent turns only (one entry per reply grew for the whole session).
const KEEP_TURNS = 16
function remember(coll, id, value) {
  if (coll instanceof Map) {
    coll.delete(id)                   // re-inserted: the newest
    coll.set(id, value)
  } else {
    coll.delete(id)
    coll.add(id)
  }
  if (coll.size > KEEP_TURNS) coll.delete(coll.keys().next().value)
}

// 16 kHz where the browser takes the option (old Safari throws for it): the browser's own rate otherwise.
function newMicContext() {
  try {
    return new AudioContext({ sampleRate: MIC_SAMPLE_RATE, latencyHint: 'interactive' })
  } catch {
    return new AudioContext({ latencyHint: 'interactive' })
  }
}

// A microphone that could not start: `name` is the browser's error (NotAllowedError, ...) or ours.
export class MicError extends Error {
  constructor(name, detail = '') {
    super(micErrorMessage(name))
    this.name = name
    this.detail = detail
  }
}

const APPLE = typeof navigator !== 'undefined' && /iPhone|iPad|iPod|Macintosh/.test(navigator.userAgent)

// What the user sees, and what to do about it.
export function micErrorMessage(name) {
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return APPLE
        ? 'The microphone is blocked for this page. Tap "aA" in the address bar → Website Settings → Microphone → Allow (or Settings → Safari → Microphone), then press Start voice again.'
        : 'The microphone is blocked for this page. Click the icon left of the address (🔒 or ⓘ) → Microphone → Allow, then press Start voice again.'
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return 'No microphone was found. Connect one, or check that it is turned on in the system settings (Windows: Settings → Privacy → Microphone), then press Start voice again.'
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'The microphone could not be opened: another app may be using it, or the system blocks it. Close other apps that use the microphone and press Start voice again.'
    case 'NoMediaDevices':
      return 'This browser cannot use the microphone here. Open the link in Chrome, Edge or Safari directly (not inside another app such as a chat app).'
    case 'NotSecure':
      return 'The microphone needs a secure (https) link. Open the https:// link you were given.'
    case 'Suspended':
      return 'The browser did not start the audio. Press Start voice again.'
    default:
      return `The microphone could not start (${name}). Press Start voice again, or try another browser.`
  }
}
